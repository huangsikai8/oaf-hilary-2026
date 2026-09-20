"""
Compares every constructible structure on the same sessions and signals.

Deribit lists only a narrow strike band on daily expiries, so the intended
0.85x / 1.15x wings usually do not exist. That makes the choice of structure a
free parameter, and picking the best-performing one after the fact inflates the
result the same way picking the best threshold does. This script therefore
reports the whole table rather than a winner, and estimates how much the
best-of-N figure is inflated by the act of choosing it.

Structures (short side; long side is the mirror):
    straddle          short ATM call + ATM put, no wings          2 legs
    bf_nearest        + long wings at the nearest listed strike   4 legs
    bf_widest         + long wings at the outermost traded strike 4 legs
    bf_delta10        + long wings nearest 10-delta               4 legs

Requires data/chain_df.csv from collect_chain.py.

    python compare_structures.py
"""

import math

import numpy as np
import pandas as pd
import audit_extended as ax
import walkforward as wf

wf.OPTIONS_CSV = "data/options_df_ext.csv"
wf.SESSIONS_CSV = "data/sessions_ext.csv"

CHAIN = "data/chain_df.csv"
DAY = 1.0 / 365.0
TARGET_DELTA = 0.10
N_BOOT = 2000
RNG = np.random.default_rng(7)


def _norm_cdf(x):
    """Standard normal CDF, vectorised, without a SciPy dependency."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    out[ok] = 0.5 * (1.0 + np.vectorize(math.erf)(x[ok] / math.sqrt(2.0)))
    return out


def bs_delta(spot, strike, iv, t=DAY):
    """Black-Scholes delta at zero rates. Returns call delta; put = call - 1."""
    iv = np.asarray(iv, dtype=float)
    iv = np.where((iv <= 0) | ~np.isfinite(iv), np.nan, iv)
    d1 = (np.log(np.asarray(spot, float) / np.asarray(strike, float))
          + 0.5 * iv**2 * t) / (iv * np.sqrt(t))
    return _norm_cdf(d1)


def load_chain():
    ch = pd.read_csv(CHAIN)
    ch["session_start"] = pd.to_datetime(ch["session_start"], utc=True)
    ch["expiry"] = pd.to_datetime(ch["expiry"], utc=True)
    ch = ch[ch["entry_gap_hours"] <= 2.0]
    ch = ch[ch["entry_iv"].between(0.01, 5.0) | ch["entry_iv"].isna()]
    call_delta = bs_delta(ch["spot_at_open"], ch["strike"], ch["entry_iv"])
    ch["delta"] = np.where(ch["option_type"] == "C", call_delta, call_delta - 1.0)
    return ch


def pick_legs(g):
    """For one session's chain, return the leg selections for every structure."""
    calls = g[g["option_type"] == "C"]
    puts = g[g["option_type"] == "P"]
    if calls.empty or puts.empty:
        return None
    spot = g["spot_at_open"].iloc[0]

    atm_c = calls.loc[(calls["strike"] - spot).abs().idxmin()]
    atm_p = puts.loc[(puts["strike"] - spot).abs().idxmin()]

    out = {"atm_c": atm_c, "atm_p": atm_p, "spot": spot}

    # nearest listed strike to the intended 1.15x / 0.85x
    otm_c = calls.loc[(calls["strike"] - spot * 1.15).abs().idxmin()]
    otm_p = puts.loc[(puts["strike"] - spot * 0.85).abs().idxmin()]
    out["near_c"], out["near_p"] = otm_c, otm_p

    # outermost traded strike on each side
    out["wide_c"] = calls.loc[calls["strike"].idxmax()]
    out["wide_p"] = puts.loc[puts["strike"].idxmin()]

    # nearest to 10-delta, among strikes outside the money
    cc = calls[(calls["delta"].notna()) & (calls["strike"] > spot)]
    pp = puts[(puts["delta"].notna()) & (puts["strike"] < spot)]
    out["d10_c"] = (
        cc.loc[(cc["delta"] - TARGET_DELTA).abs().idxmin()] if not cc.empty else None
    )
    out["d10_p"] = (
        pp.loc[(pp["delta"] + TARGET_DELTA).abs().idxmin()] if not pp.empty else None
    )
    return out


def leg_pnl(leg, spot_exp, sign):
    """USD P&L and fee for one leg held to expiry, premium converted at entry."""
    k, prem, spot0 = leg["strike"], leg["entry_price"], leg["spot_at_open"]
    intr = max(spot_exp - k, 0.0) if leg["option_type"] == "C" else max(k - spot_exp, 0.0)
    entry_usd = prem * spot0
    fee = min(ax.FEE_PER_CONTRACT_BTC * spot0, ax.FEE_PREMIUM_CAP * entry_usd)
    if intr > 0:
        fee += min(ax.SETTLE_PER_CONTRACT_BTC * spot_exp, ax.SETTLE_INTRINSIC_CAP * intr)
    return sign * (intr - entry_usd), fee


STRUCTURES = {
    "straddle": [("atm_c", -1), ("atm_p", -1)],
    "bf_nearest": [("atm_c", -1), ("atm_p", -1), ("near_c", 1), ("near_p", 1)],
    "bf_widest": [("atm_c", -1), ("atm_p", -1), ("wide_c", 1), ("wide_p", 1)],
    "bf_delta10": [("atm_c", -1), ("atm_p", -1), ("d10_c", 1), ("d10_p", 1)],
}


def build_pnl(chain, btc):
    spot_exp = (
        btc[["time", "close"]]
        .rename(columns={"time": "expiry", "close": "spot_at_expiry"})
        .set_index("expiry")["spot_at_expiry"]
    )
    rows = []
    for s_start, g in chain.groupby("session_start", sort=True):
        sel = pick_legs(g)
        if sel is None:
            continue
        exp = g["expiry"].iloc[0]
        if exp not in spot_exp.index:
            continue
        se = float(spot_exp.loc[exp])
        rec = {"session_start": s_start, "spot_at_expiry": se}
        for name, legs in STRUCTURES.items():
            if any(sel.get(k) is None for k, _ in legs):
                rec[f"{name}_gross"] = np.nan
                rec[f"{name}_net"] = np.nan
                continue
            gross = fees = 0.0
            for key, sign in legs:
                p, f = leg_pnl(sel[key], se, sign)
                gross += p
                fees += f
            rec[f"{name}_gross"] = gross
            rec[f"{name}_net"] = gross - fees
        rec["near_c_m"] = sel["near_c"]["moneyness"]
        rec["wide_c_m"] = sel["wide_c"]["moneyness"]
        rec["d10_c_m"] = sel["d10_c"]["moneyness"] if sel["d10_c"] is not None else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def stats(series, years):
    s = series.dropna()
    sh, se, n = wf.sharpe(s, years)
    if not n:
        return None
    trimmed = s.sort_values()
    keep_hi = trimmed.iloc[:-5] if n > 10 else trimmed
    keep_lo = trimmed.iloc[5:] if n > 10 else trimmed
    return {
        "n": n,
        "sharpe": sh,
        "se": se,
        "t": sh / se,
        "mean": s.mean(),
        "median": s.median(),
        "win": 100 * (s > 0).mean(),
        "ex_best5": wf.sharpe(keep_hi, years)[0],
        "ex_worst5": wf.sharpe(keep_lo, years)[0],
    }


def main():
    btc, options, sessions = wf.load_inputs()
    feat = wf.build_session_features(btc, options, sessions)[
        ["session_start", "session_type", "vrp", "iv_rank"]
    ]
    chain = load_chain()
    print(
        f"chain: {len(chain)} rows, {chain['session_start'].nunique()} sessions, "
        f"{chain['session_start'].min():%Y-%m-%d}..{chain['session_start'].max():%Y-%m-%d}"
    )

    pnl = build_pnl(chain, btc)
    df = feat.merge(pnl, on="session_start", how="inner").dropna(subset=["vrp"])
    df = df.sort_values("session_start").reset_index(drop=True)
    span = df["session_start"]
    years = (span.max() - span.min()).days / 365.25
    print(f"usable: {len(df)} sessions, {years:.2f}y, Sharpe SE ±{1 / np.sqrt(years):.2f}\n")

    print("Realised wing moneyness (call side, target 1.150):")
    for c, lbl in [("near_c_m", "nearest-to-1.15"), ("wide_c_m", "widest traded"), ("d10_c_m", "10-delta")]:
        print(f"  {lbl:16s} median {df[c].median():.3f}x   p90 {df[c].quantile(0.9):.3f}x")

    v, ivr = df["vrp"].to_numpy(), df["iv_rank"].to_numpy()
    signals = {
        "VRP>0 (no params)": v > 0,
        "VRP>.05 & IVR>40": (v > 0.05) & (ivr > 40),
        "always on": np.ones(len(df), bool),
    }

    print(f"\n{'=' * 100}\nSHORT VOL — every structure x every signal, net of fees\n{'=' * 100}")
    hdr = f"{'signal':20s} {'structure':12s} {'n':>5} {'gross':>7} {'net':>7} {'t(net)':>7} {'mean$':>8} {'med$':>8} {'win%':>6} {'net ex-worst5':>14}"
    print(hdr)
    results = {}
    for sname, sig in signals.items():
        for st in STRUCTURES:
            g = stats(df.loc[sig, f"{st}_gross"], years)
            nt = stats(df.loc[sig, f"{st}_net"], years)
            if nt is None:
                continue
            results[(sname, st)] = nt["sharpe"]
            print(
                f"{sname:20s} {st:12s} {nt['n']:5d} {g['sharpe']:7.2f} {nt['sharpe']:7.2f} "
                f"{nt['t']:7.2f} {nt['mean']:8.0f} {nt['median']:8.0f} {nt['win']:6.1f} "
                f"{nt['ex_worst5']:14.2f}"
            )
        print()

    # ---- how much does "take the best" inflate the number? ----
    print("=" * 100)
    print("COST OF PICKING THE BEST STRUCTURE")
    print("=" * 100)
    cols = [f"{st}_net" for st in STRUCTURES]
    sig = signals["VRP>.05 & IVR>40"]
    sub = df.loc[sig, cols].dropna()
    n = len(sub)
    ann = np.sqrt(n / years)
    M = sub.to_numpy()                      # sessions x structures

    def sharpe_cols(a):
        sd = a.std(axis=0, ddof=1)
        return np.where(sd > 0, a.mean(axis=0) / np.where(sd > 0, sd, 1) * ann, np.nan)

    obs = dict(zip(cols, sharpe_cols(M)))
    best_col = max(obs, key=obs.get)
    best_i = cols.index(best_col)

    draws = np.empty((N_BOOT, len(cols)))
    for b in range(N_BOOT):
        draws[b] = sharpe_cols(M[RNG.integers(0, n, n)])

    parts = ", ".join(f"{c.replace('_net', '')}={v:.2f}" for c, v in obs.items())
    print(f"  n common sessions            : {n}")
    print(f"  observed Sharpe by structure : {parts}")
    print(f"  best observed                : {obs[best_col]:.2f}  ({best_col.replace('_net','')})")
    print(f"  mean across structures       : {np.nanmean(list(obs.values())):.2f}")
    print(
        f"  bootstrap E[max of 4]        : {np.nanmean(draws.max(axis=1)):.2f}   "
        f"E[single structure]: {np.nanmean(draws):.2f}   "
        f"inflation from choosing: "
        f"{np.nanmean(draws.max(axis=1)) - np.nanmean(draws):+.2f} Sharpe"
    )
    lo, hi = np.nanpercentile(draws[:, best_i], [2.5, 97.5])
    print(f"  95% CI on the best structure : [{lo:.2f}, {hi:.2f}]")
    if lo <= 0:
        print("  -> the best structure's confidence interval includes zero.")


if __name__ == "__main__":
    main()
