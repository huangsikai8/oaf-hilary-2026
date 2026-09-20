"""
The selected strategy, end to end.

    Short the ATM straddle on a BTC daily expiry whenever implied volatility
    exceeds the day-of-week adjusted realised-volatility forecast.

No tuned parameters: the entry threshold is zero, the natural point at which
implied equals forecast realised. Nothing about the rule can be fitted, which is
why it is the one carried forward out of the several tested.

Prints every figure quoted in the write-up.

    python strategy.py
"""

import numpy as np
import pandas as pd

import audit_extended as ax
import compare_structures as cs
import walkforward as wf

HALF_SPREAD = 0.041          # measured half-spread, share of premium, per leg
RNG = np.random.default_rng(3)


def build():
    """Features, signal and P&L for every session, on three cost bases."""
    btc, options, sessions = wf.load_inputs()
    feat = wf.build_session_features(btc, options, sessions)
    chain = cs.load_chain()

    spot_exp = (
        btc[["time", "close"]]
        .rename(columns={"time": "expiry", "close": "se"})
        .set_index("expiry")["se"]
    )

    rows = []
    for s_start, g in chain.groupby("session_start", sort=True):
        sel = cs.pick_legs(g)
        if sel is None:
            continue
        exp = g["expiry"].iloc[0]
        if exp not in spot_exp.index:
            continue
        se = float(spot_exp.loc[exp])

        gross = fees = slip = prem = 0.0
        for key in ("atm_c", "atm_p"):
            leg = sel[key]
            p, f = cs.leg_pnl(leg, se, -1)          # short both legs
            gross += p
            fees += f
            entry_usd = leg["entry_price"] * leg["spot_at_open"]
            prem += entry_usd
            slip += HALF_SPREAD * entry_usd
        rows.append(
            {
                "session_start": s_start,
                "gross": gross,
                "net_fees": gross - fees,
                "net_all": gross - fees - slip,
                "premium": prem,
                "fee": fees,
                "spot": sel["spot"],
            }
        )

    d = feat.merge(pd.DataFrame(rows), on="session_start", how="inner")
    return d.dropna(subset=["vrp"]).sort_values("session_start").reset_index(drop=True)


def stats(x, years):
    x = pd.Series(x).dropna()
    sh, se, n = wf.sharpe(x, years)
    if not n:
        return None
    return {
        "n": n, "sharpe": sh, "se": se, "t": sh / se,
        "mean": x.mean(), "median": x.median(),
        "win": 100 * (x > 0).mean(), "total": x.sum(),
    }


def line(label, s):
    if s is None:
        return f"  {label:34s}  no trades"
    return (
        f"  {label:34s} n={s['n']:5d}  Sharpe {s['sharpe']:6.2f}  t {s['t']:5.2f}  "
        f"mean ${s['mean']:7.0f}  med ${s['median']:7.0f}  win {s['win']:4.1f}%"
    )


def main():
    d = build()
    sig = d["vrp"] > 0
    span = d["session_start"]
    years = (span.max() - span.min()).days / 365.25
    t = d.loc[sig]

    print("=" * 84)
    print("SAMPLE")
    print("=" * 84)
    print(f"  sessions available     : {len(d)}")
    print(f"  span                   : {span.min():%Y-%m-%d} to {span.max():%Y-%m-%d} ({years:.2f}y)")
    print(f"  signal fires           : {int(sig.sum())}  ({100*sig.mean():.1f}% of sessions)")
    print(f"  Sharpe standard error  : +/-{1/np.sqrt(years):.2f}\n")

    print("=" * 84)
    print("HEADLINE, three cost bases")
    print("=" * 84)
    for col, lab in [("gross", "gross"), ("net_fees", "net of exchange fees"),
                     ("net_all", "net of fees + half-spread")]:
        print(line(lab, stats(t[col], years)))

    print("\n  baselines, same structure:")
    for col, lab in [("gross", "always on, gross"), ("net_fees", "always on, net fees")]:
        print(line(lab, stats(d[col], years)))
    print(line("signal OFF (VRP<0), gross", stats(d.loc[~sig, "gross"], years)))

    print("\n" + "=" * 84)
    print("STABILITY")
    print("=" * 84)
    h = len(d) // 2
    for lab, m in [("first half", np.arange(len(d)) < h), ("second half", np.arange(len(d)) >= h)]:
        sub = d.loc[m & sig]
        yy = (d.loc[m, "session_start"].max() - d.loc[m, "session_start"].min()).days / 365.25
        print(line(f"{lab} gross", stats(sub["gross"], yy)))
    print()
    d["yr"] = span.dt.year
    print(f"  {'year':>6} {'n':>5} {'gross Sh':>9} {'gross $':>9} {'net $':>8} {'fee %':>7} {'med IV':>7}")
    for y, g in d[sig].groupby("yr"):
        yy = max((g.session_start.max() - g.session_start.min()).days / 365.25, .1)
        sh = wf.sharpe(g["gross"], yy)[0]
        fee_pct = 100 * g["fee"].mean() / g["gross"].mean() if g["gross"].mean() else np.nan
        print(f"  {y:>6} {len(g):5d} {sh:9.2f} {g['gross'].mean():9.0f} "
              f"{g['net_fees'].mean():8.0f} {fee_pct:6.0f}% {100*g['atm_iv'].median():6.0f}%")

    print("\n" + "=" * 84)
    print("TAIL SENSITIVITY  (Sharpe is the wrong statistic for a short-vol book)")
    print("=" * 84)
    x = t["gross"].sort_values()
    for k in (0, 1, 3, 5, 10):
        keep = x.iloc[k:] if k else x
        sh = wf.sharpe(keep, years)[0]
        print(f"  excluding the {k:2d} worst trades : Sharpe {sh:5.2f}   "
              f"(those {k} cost ${-x.iloc[:k].sum():,.0f})" if k else
              f"  all trades                   : Sharpe {sh:5.2f}")
    print(f"\n  worst 5 trades: {[f'${v:,.0f}' for v in x.head(5)]}")
    print(f"  best 5 trades : {[f'${v:,.0f}' for v in x.tail(5)]}")
    print(f"  bottom 1% of trades account for {100*x.head(max(1,len(x)//100)).sum()/x.sum():.0f}% of total P&L")

    print("\n" + "=" * 84)
    print("DAY OF WEEK, gross")
    print("=" * 84)
    d["dow"] = span.dt.day_name()
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        g = d[(d.dow == day) & sig]
        if len(g) < 5:
            continue
        yy = (g.session_start.max() - g.session_start.min()).days / 365.25
        s = stats(g["gross"], yy)
        print(f"  {day:10s} n={s['n']:4d}  Sharpe {s['sharpe']:5.2f}  mean ${s['mean']:6.0f}  win {s['win']:4.1f}%")

    print("\n" + "=" * 84)
    print("EQUITY CURVE  (cumulative gross P&L, sampled for plotting)")
    print("=" * 84)
    eq = t[["session_start", "gross", "net_fees"]].copy()
    eq["cg"] = eq["gross"].cumsum()
    eq["cn"] = eq["net_fees"].cumsum()
    step = max(1, len(eq) // 40)
    pts = eq.iloc[::step]
    print("  " + ", ".join(f"[{r.session_start:%Y-%m},{r.cg:.0f},{r.cn:.0f}]" for r in pts.itertuples()))
    run_max = eq["cg"].cummax()
    dd = (eq["cg"] - run_max)
    print(f"\n  peak cumulative gross ${eq['cg'].max():,.0f}   final ${eq['cg'].iloc[-1]:,.0f}")
    print(f"  max drawdown ${dd.min():,.0f}  ({100*dd.min()/max(run_max.max(),1):.0f}% of peak)")


if __name__ == "__main__":
    main()
