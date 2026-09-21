"""
Parameter search and walk-forward validation for the premium-filtered straddle.

The strategy sells the ATM straddle when the expected premium is positive AND
the straddle premium is rich relative to its own recent history:

    VRP_hat > 0   and   premium/spot  >  q-th percentile over the last W sessions

W and q are the only fitted quantities. The filter itself is not a discovered
pattern - it follows from the fee schedule. Deribit charges a near-fixed amount
per contract, so the edge has to clear a floor and only fat-premium sessions are
worth trading. The search decides where that floor sits.

A trailing *percentile* rather than an absolute level is deliberate: an absolute
threshold is an implied-vol level in disguise and spends most of the sample
waiting for a high-vol regime. The percentile adapts, so the rule keeps trading.

Prints the full-sample grid, the walk-forward results at several burn-ins, and
the untuned control each is measured against.

Parameters are selected on net-of-fee performance. The resulting out-of-sample
trades are then scored on gross, net-of-fee and spread-adjusted P&L, so all
three headline figures come from one strategy and one set of trades.

    python optimise.py
"""

import numpy as np
import pandas as pd

import strategy as S
import walkforward as wf

LOOKBACKS = (60, 120, 250)
CUTOFFS = (0.3, 0.4, 0.5, 0.6, 0.7)
GRID = [(w, q) for w in LOOKBACKS for q in CUTOFFS]

REBALANCE = 60          # sessions between refits
MIN_TRAIN_TRADES = 40
BURN_INS = (365, 550, 730)


def prepare():
    d = S.build().reset_index(drop=True)
    d["prem_pct"] = d["premium"] / d["spot"]
    for w in LOOKBACKS:
        # percentile of today's premium within the trailing window, lagged one
        # session so the current observation never enters its own benchmark
        d[f"q{w}"] = d["prem_pct"].rolling(w, min_periods=w // 2).rank(pct=True).shift(1)
    return d


def fires(d, w, q):
    return (d["vrp"].to_numpy() > 0) & (d[f"q{w}"].to_numpy() > q)


def sharpe(x, years):
    x = pd.Series(x).dropna()
    if len(x) < 2 or years <= 0 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / x.std(ddof=1) * np.sqrt(len(x) / years)


def span_years(dates):
    return (dates.max() - dates.min()).days / 365.25


def full_sample_grid(d, col="net_fees"):
    years = span_years(d["session_start"])
    print(f"{'':10s}" + "".join(f"{q:>9.1f}" for q in CUTOFFS))
    for w in LOOKBACKS:
        row = f"  W={w:<6d}"
        for q in CUTOFFS:
            m = fires(d, w, q)
            row += f"{sharpe(d.loc[m, col], years):9.2f}"
        print(row)
    base = sharpe(d.loc[d["vrp"] > 0, col], years)
    print(f"\n  untuned control, no premium filter: {base:.2f} "
          f"({int((d['vrp'] > 0).sum())} trades)")


SCORE_ON = ("gross", "net_fees", "net_all")


def walk_forward(d, burn_in, objective="net_fees"):
    """Expanding window: choose (W, q) on prior data, apply to the next block.

    Parameters are selected on `objective` only. The trades that selection
    produces are then scored on every column in SCORE_ON, so the cost bases are
    sensitivities applied to one strategy rather than three different adaptive
    strategies. Returns the same trades under each basis, their dates, and the
    parameters chosen per block.
    """
    n = len(d)
    signals = {(w, q): fires(d, w, q) for w, q in GRID}
    oos = {c: [] for c in SCORE_ON}
    dates, chosen = [], []

    for start in range(burn_in, n, REBALANCE):
        end = min(start + REBALANCE, n)
        train, test = np.arange(start), np.arange(start, end)
        train_years = span_years(d["session_start"].iloc[train])

        best, best_sh = None, -np.inf
        for params, sig in signals.items():
            series = d[objective].to_numpy()[train][sig[train]]
            if len(series) < MIN_TRAIN_TRADES:
                continue
            s = sharpe(series, train_years)
            if np.isfinite(s) and s > best_sh:
                best, best_sh = params, s
        if best is None:
            best = (250, 0.7)

        sig = signals[best]
        fired = sig[test]
        for c in SCORE_ON:
            oos[c].extend(d[c].to_numpy()[test][fired])
        dates.extend(d["session_start"].to_numpy()[test][fired])
        chosen.append(best)

    return ({c: np.array(v) for c, v in oos.items()},
            pd.to_datetime(dates), chosen)


def main():
    d = prepare()
    years = span_years(d["session_start"])
    print("=" * 78)
    print(f"FULL-SAMPLE GRID, net of exchange fees   ({len(d)} sessions, {years:.2f}y)")
    print("=" * 78)
    full_sample_grid(d)

    print("\n" + "=" * 78)
    print("WALK-FORWARD   parameters chosen on prior data only, refit every "
          f"{REBALANCE} sessions")
    print("=" * 78)
    for burn in BURN_INS:
        oy = span_years(d["session_start"].iloc[burn:])
        print(f"\n  burn-in {burn} sessions "
              f"({d['session_start'].iloc[burn]:%Y-%m}) | OOS {oy:.2f}y | SE +/-{1/np.sqrt(oy):.2f}")
        oos, dates, chosen = walk_forward(d, burn, objective="net_fees")
        labels = {"gross": "gross", "net_fees": "net fees", "net_all": "fees+spread"}
        for col in SCORE_ON:
            base = d[col].to_numpy()[burn:][d["vrp"].to_numpy()[burn:] > 0]
            s = sharpe(oos[col], oy)
            print(f"    {labels[col]:11s} tuned {s:5.2f}  t {s * np.sqrt(oy):5.2f}  "
                  f"(n={len(oos[col]):4d})   untuned {sharpe(base, oy):5.2f} (n={len(base):4d})")
        counts = pd.Series(chosen).value_counts()
        print(f"    parameters: {len(counts)} distinct, most common {counts.index[0]} "
              f"in {counts.iloc[0]}/{len(chosen)} blocks")
        by_year = pd.Series(1, index=dates).groupby(dates.year).size()
        print(f"    OOS trades by year: {dict(by_year)}")


if __name__ == "__main__":
    main()
