"""
Collects the full traded option chain per session, not just four chosen legs.

The existing collector picks the four strategy legs at fetch time, which fixes
the structure permanently into the data. Deribit lists only a narrow strike band
on daily expiries, so the intended 0.85x / 1.15x wings frequently do not exist
and the chosen legs land wherever the chain happens to reach. Storing the whole
chain instead lets every candidate structure be built offline from one pull.

For each session this records every instrument on the target expiry that traded
in the entry window, with the trade nearest to session open as its entry price.

Writes data/chain_df.csv:
    session_start, session_close, session_type, session_dow, spot_at_open,
    expiry, instrument_name, strike, option_type, moneyness,
    entry_price, entry_iv, entry_underlying, entry_gap_hours, n_trades

    python collect_chain.py
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

import options_data_collection as odc

START = "2021-01-01"
END = "2026-09-08"
CHUNK = 60
OUT = "data/chain_df.csv"
CHECKPOINT = "data/chain_partial.csv"


def target_expiry(trades):
    """Nearest expiry at or after session close, ties broken by trade count.

    Mirrors the rule in options_data_collection.identify_legs so the chain is
    anchored to the same daily contract the original study used.
    """
    counts = trades["instrument_name"].value_counts()
    best = {}
    for name, cnt in counts.items():
        meta = odc.parse_instrument(name)
        exp = meta.get("expiry")
        if pd.isna(exp):
            continue
        best[exp] = best.get(exp, 0) + int(cnt)
    return best or None


def collect(sessions, btc):
    rows = []
    for _, s in sessions.iterrows():
        s_start, s_close = s["session_start"], s["session_close"]
        try:
            spot = float(btc.loc[s_start, "close"])
        except KeyError:
            i = btc.index.get_indexer([s_start], method="nearest")[0]
            spot = float(btc.iloc[i]["close"])

        trades = odc.fetch_trades(s_start, s_start + dt.timedelta(hours=odc.ENTRY_WINDOW_HOURS))
        if trades.empty:
            continue

        counts = target_expiry(trades)
        if not counts:
            continue
        eligible = {e: c for e, c in counts.items() if e.date() >= s_close.date()}
        if not eligible:
            continue
        exp = min(
            eligible,
            key=lambda e: (abs((e.date() - s_close.date()).days), -eligible[e]),
        )

        trades = trades.copy()
        meta = trades["instrument_name"].map(odc.parse_instrument)
        trades["strike"] = [m.get("strike") for m in meta]
        trades["option_type"] = [m.get("option_type") for m in meta]
        trades["expiry"] = [m.get("expiry") for m in meta]
        trades = trades[trades["expiry"] == exp].dropna(subset=["strike"])
        if trades.empty:
            continue

        trades["gap"] = (trades["timestamp"] - s_start).abs()
        trades = trades.sort_values("gap")

        for name, g in trades.groupby("instrument_name", sort=False):
            first = g.iloc[0]
            rows.append(
                {
                    "session_start": s_start,
                    "session_close": s_close,
                    "session_type": s["session_type"],
                    "session_dow": s["session_dow"],
                    "spot_at_open": spot,
                    "expiry": exp,
                    "instrument_name": name,
                    "strike": float(first["strike"]),
                    "option_type": first["option_type"],
                    "moneyness": float(first["strike"]) / spot,
                    "entry_price": float(first["price"]),
                    "entry_iv": float(first["iv"]) / 100.0 if pd.notna(first.get("iv")) else np.nan,
                    "entry_underlying": float(first.get("index_price", np.nan)),
                    "entry_gap_hours": first["gap"].total_seconds() / 3600.0,
                    "n_trades": int(len(g)),
                }
            )
    return pd.DataFrame(rows)


def main():
    btc = pd.read_csv("data/btc_prices.csv", index_col=0, parse_dates=True)
    btc.index = pd.to_datetime(btc.index, utc=True)
    btc = btc[~btc.index.duplicated(keep="first")]

    sessions = odc.generate_session_windows(START, END)

    done = set()
    if os.path.exists(CHECKPOINT):
        part = pd.read_csv(CHECKPOINT)
        part["session_start"] = pd.to_datetime(part["session_start"], utc=True)
        done = set(part["session_start"])
        sessions = sessions[~sessions["session_start"].isin(done)]
        print(f"resuming, {len(done)} sessions already stored", flush=True)

    sessions = sessions.reset_index(drop=True)
    print(f"collecting {len(sessions)} sessions", flush=True)

    for i in range(0, len(sessions), CHUNK):
        chunk = sessions.iloc[i : i + CHUNK]
        got = collect(chunk, btc)
        if got.empty:
            continue
        got.to_csv(CHECKPOINT, mode="a", header=not os.path.exists(CHECKPOINT), index=False)
        print(
            f"  {chunk['session_start'].min():%Y-%m-%d}..{chunk['session_start'].max():%Y-%m-%d}"
            f"  +{len(got)} rows  ({got['session_start'].nunique()} sessions)",
            flush=True,
        )

    chain = pd.read_csv(CHECKPOINT)
    chain["session_start"] = pd.to_datetime(chain["session_start"], utc=True)
    chain["session_close"] = pd.to_datetime(chain["session_close"], utc=True)
    chain["expiry"] = pd.to_datetime(chain["expiry"], utc=True)
    chain = chain.sort_values(["session_start", "option_type", "strike"]).reset_index(drop=True)
    chain.to_csv(OUT, index=False)

    per = chain.groupby("session_start").size()
    print(
        f"\nwrote {OUT}: {len(chain)} rows, {chain['session_start'].nunique()} sessions\n"
        f"  instruments per session: median {per.median():.0f}, "
        f"p10 {per.quantile(0.1):.0f}, p90 {per.quantile(0.9):.0f}\n"
        f"  moneyness reach: p1 {chain['moneyness'].quantile(0.01):.3f}x, "
        f"p99 {chain['moneyness'].quantile(0.99):.3f}x",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
