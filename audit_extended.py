"""
Full audit of the VRP strategies on the extended 2021-2026 sample.

Runs on data/options_df_ext.csv (produced by collect_extended.py) and reports:

  1. Coverage and data quality by year, including how well the OTM strangle
     legs hit their 85% / 115% moneyness targets in the thin early chains.
  2. Signal rules, gross and net of Deribit taker fees, on the full sample and
     on the later half.
  3. Where the published thresholds sit in the parameter grid.
  4. Walk-forward, with the burn-in swept so the sensitivity is visible.

    python audit_extended.py
"""

import numpy as np
import pandas as pd

import walkforward as wf

wf.OPTIONS_CSV = "data/options_df_ext.csv"
wf.SESSIONS_CSV = "data/sessions_ext.csv"

# Deribit options taker fee: 0.0003 BTC per contract, capped at 12.5% of the
# option premium. In-the-money settlement: 0.00015 BTC, capped at 12.5% of
# intrinsic. Applied per leg.
FEE_PER_CONTRACT_BTC = 0.0003
FEE_PREMIUM_CAP = 0.125
SETTLE_PER_CONTRACT_BTC = 0.00015
SETTLE_INTRINSIC_CAP = 0.125


def build_pnl_with_fees(options, btc):
    """Per-session P&L for both structures, gross and net of taker fees."""
    spot_at_expiry = btc[["time", "close"]].rename(
        columns={"time": "expiry", "close": "spot_at_expiry"}
    )
    legs = options.merge(spot_at_expiry, on="expiry", how="left")
    legs = legs.dropna(subset=["spot_at_expiry"])

    intrinsic = np.where(
        legs["option_type"] == "C",
        np.maximum(legs["spot_at_expiry"] - legs["strike"], 0.0),
        np.maximum(legs["strike"] - legs["spot_at_expiry"], 0.0),
    )
    entry_usd = legs["entry_price"] * legs["spot_at_open"]

    fee = np.minimum(
        FEE_PER_CONTRACT_BTC * legs["spot_at_open"], FEE_PREMIUM_CAP * entry_usd
    ) + np.where(
        intrinsic > 0,
        np.minimum(
            SETTLE_PER_CONTRACT_BTC * legs["spot_at_expiry"],
            SETTLE_INTRINSIC_CAP * intrinsic,
        ),
        0.0,
    )

    leg_pnl_long = intrinsic - entry_usd
    is_atm = legs["leg"].str.startswith("atm").to_numpy()

    legs["gross_short_bf"] = leg_pnl_long * np.where(is_atm, -1.0, 1.0)
    legs["gross_long_str"] = leg_pnl_long * np.where(is_atm, 1.0, 0.0)
    legs["net_short_bf"] = legs["gross_short_bf"] - fee
    legs["net_long_str"] = legs["gross_long_str"] - np.where(is_atm, fee, 0.0)

    cols = ["gross_short_bf", "gross_long_str", "net_short_bf", "net_long_str"]
    return legs.groupby("session_start")[cols].sum().reset_index()


def coverage_report(options, feat):
    print("=" * 78)
    print("COVERAGE AND DATA QUALITY")
    print("=" * 78)
    o = options.copy()
    o["year"] = o["session_start"].dt.year
    o["moneyness"] = o["strike"] / o["spot_at_open"]

    per_session = o.groupby("session_start").size()
    complete = set(per_session[per_session == 4].index)

    rows = []
    for year, g in o.groupby("year"):
        sess = g["session_start"].nunique()
        full = len(set(g["session_start"]) & complete)
        otm_c = g.loc[g["leg"] == "otm_call", "moneyness"]
        otm_p = g.loc[g["leg"] == "otm_put", "moneyness"]
        rows.append(
            {
                "year": year,
                "sessions": sess,
                "all 4 legs": full,
                "quality ok %": 100.0 * g["data_quality_ok"].mean(),
                "max gap h": g["entry_gap_hours"].max(),
                "otm_call m": otm_c.median() if len(otm_c) else np.nan,
                "otm_put m": otm_p.median() if len(otm_p) else np.nan,
                "|call err|": (otm_c - 1.15).abs().median() if len(otm_c) else np.nan,
                "|put err|": (otm_p - 0.85).abs().median() if len(otm_p) else np.nan,
            }
        )
    df = pd.DataFrame(rows).set_index("year")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.to_string())
    print(
        "\n  otm_call m / otm_put m are median realised moneyness "
        "(targets 1.150 and 0.850);\n  the err columns are the median absolute "
        "miss, which shows how coarse each year's strike grid was.\n"
    )


def evaluate(feat, pnl, mask, sig, col, label):
    m = mask & sig
    s = pnl.loc[m, col]
    span = feat.loc[mask, "session_start"]
    years = max((span.max() - span.min()).days / 365.25, 1e-9)
    sh, se, n = wf.sharpe(s, years)
    if not n:
        return f"  {label:44s}  no trades"
    return (
        f"  {label:44s} n={n:5d}  Sharpe={sh:6.2f} ±{se:.2f}  "
        f"t={sh / se:5.2f}  mean=${s.mean():8.1f}"
    )


def main():
    btc, options, sessions = wf.load_inputs()
    feat = wf.build_session_features(btc, options, sessions)
    pnl_all = build_pnl_with_fees(options, btc)

    feat = feat.merge(pnl_all, on="session_start", how="inner")
    feat = feat.sort_values("session_start").reset_index(drop=True)
    feat = feat.dropna(subset=["vrp"]).reset_index(drop=True)
    pnl = feat[["gross_short_bf", "gross_long_str", "net_short_bf", "net_long_str"]]

    coverage_report(options, feat)

    N = len(feat)
    span = feat["session_start"]
    years = (span.max() - span.min()).days / 365.25
    half = N // 2
    later = np.zeros(N, dtype=bool)
    later[half:] = True
    allm = np.ones(N, dtype=bool)
    later_years = (
        span.iloc[N - 1] - span.iloc[half]
    ).days / 365.25

    print("=" * 78)
    print("SAMPLE")
    print("=" * 78)
    print(f"  sessions with a VRP signal : {N}")
    print(f"  span                       : {span.min():%Y-%m-%d} to {span.max():%Y-%m-%d} ({years:.2f}y)")
    print(f"  Sharpe standard error      : ±{1 / np.sqrt(years):.2f}  (was ±0.58 on 3.00y)")
    print(f"  later half                 : {span.iloc[half]:%Y-%m-%d} onward ({later_years:.2f}y, ±{1 / np.sqrt(later_years):.2f})\n")

    v = feat["vrp"].to_numpy()
    ivr = feat["iv_rank"].to_numpy()
    rules = [
        ("zero-parameter, VRP > 0", v > 0, "short"),
        ("published, VRP > 0.05 & IVR > 40", (v > 0.05) & (ivr > 40), "short"),
        ("always on, every session", np.ones(N, bool), "short"),
        ("zero-parameter, VRP < 0", v < 0, "long"),
        ("published, VRP < -0.05 & IVR < 60", (v < -0.05) & (ivr < 60), "long"),
        ("always on, every session", np.ones(N, bool), "long"),
    ]

    for scope, mask, tag in (("FULL SAMPLE", allm, ""), ("LATER HALF", later, "")):
        print("=" * 78)
        print(f"{scope}  —  gross, then net of Deribit taker fees")
        print("=" * 78)
        for gross_net in ("gross", "net"):
            print(f"  [{gross_net}]")
            for label, sig, side in rules:
                col = (
                    f"{gross_net}_short_bf" if side == "short" else f"{gross_net}_long_str"
                )
                print(evaluate(feat, pnl, mask, sig, col, f"{side:5s} {label}"))
            print()

    print("=" * 78)
    print("PARAMETER GRID, gross, full sample")
    print("=" * 78)
    pnl_grid = feat[["gross_short_bf", "gross_long_str"]].rename(
        columns={"gross_short_bf": "pnl_short_butterfly", "gross_long_str": "pnl_long_straddle"}
    )
    for st in (1, 2):
        for side in ("short", "long"):
            g = wf.grid_distribution(feat, pnl_grid, st, side)
            g = g.drop_duplicates(subset=list(wf.ACTIVE_PARAMS[(st, side)]))
            pub = wf.evaluate(feat, pnl_grid, st, wf.DECK_PARAMS[st])[side]["sharpe"]
            pct = 100.0 * (g["sharpe"] <= pub).mean()
            print(
                f"  S{st} {side:5s}  {len(g):2d} sets  min={g['sharpe'].min():5.2f}  "
                f"median={g['sharpe'].median():5.2f}  max={g['sharpe'].max():5.2f}  |  "
                f"published={pub:5.2f} ({pct:.0f}th pct)"
            )

    print("\n" + "=" * 78)
    print("WALK-FORWARD, gross, burn-in swept")
    print("=" * 78)
    print(f"{'burn-in':>9} {'OOS span':>9} {'S2 short':>12} {'S2 long':>12} {'S1 short':>12} {'S1 long':>12}")
    orig = wf.BURN_IN_SESSIONS
    for bi in (365, 550, 730, 1095, 1460):
        if bi >= N - 60:
            continue
        wf.BURN_IN_SESSIONS = bi
        sp = feat["session_start"].iloc[bi:]
        yrs = (sp.max() - sp.min()).days / 365.25
        cells = []
        for st in (2, 1):
            for side in ("short", "long"):
                o, _ = wf.walk_forward(feat, pnl_grid, st, side)
                sh, _, n = wf.sharpe(o, yrs)
                cells.append(f"{sh:6.2f}({n:4d})")
        print(f"{bi:>9} {yrs:>8.2f}y {cells[0]:>12} {cells[1]:>12} {cells[2]:>12} {cells[3]:>12}")
    wf.BURN_IN_SESSIONS = orig


if __name__ == "__main__":
    main()
