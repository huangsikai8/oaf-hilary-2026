"""
Walk-forward evaluation of the BTC short-dated options VRP strategies.

Rebuilds the signal pipeline from raw inputs so that every feature at session t
uses only information available at t, then evaluates both strategies under an
expanding-window walk-forward in which the signal thresholds are re-selected on
past data only and applied forward.

Strategy definitions (as specified in the pitch deck):

    sigma_imp        = 0.5 * (IV_atm_call + IV_atm_put)
    sigma_realised   = RV_1d,t * mu_DOW(t) / mu_DOW(t-1)
    VRP              = sigma_imp - sigma_realised
    z_VRP            = (VRP - mu_VRP,30d) / sigma_VRP,30d

    Strategy 1  short iron butterfly when z_VRP >  1
                long  straddle       when z_VRP < -1
    Strategy 2  short iron butterfly when VRP >  0.05 and IV rank > 40
                long  straddle       when VRP < -0.05 and IV rank < 60

Run after dataCollection.py and options_data_collection.py.

    python walkforward.py
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

HOURS_PER_YEAR = 8760

RV_WINDOW = 24                  # hours in the trailing realised-vol window
DOW_HIST_WINDOW = 12            # same-day-of-week observations in mu_DOW
DOW_MIN_PERIODS = 4
ALL_HIST_WINDOW = 24 * 60       # hourly observations in the all-days baseline
ALL_MIN_PERIODS = 24 * 14

IV_RANK_WINDOW = "364D"
IV_RANK_MIN_PERIODS = 20

# Published parameters, kept as the reference point.
DECK_PARAMS = {
    1: {"z_window": 30, "z_threshold": 1.0},
    2: {"vrp_threshold": 0.05, "ivr_short": 40.0, "ivr_long": 60.0},
}

# Candidate grids searched at each walk-forward refit, keyed by (strategy, side).
# Each grid varies only the parameters that affect that side, so no candidate is
# duplicated and the grid distribution is not diluted by ties.
_Z_GRID = [
    {"z_window": w, "z_threshold": z}
    for w in (20, 30, 45, 60)
    for z in (0.75, 1.0, 1.25, 1.5)
]

GRIDS = {
    (1, "short"): _Z_GRID,
    (1, "long"): _Z_GRID,
    (2, "short"): [
        {"vrp_threshold": v, "ivr_short": s, "ivr_long": DECK_PARAMS[2]["ivr_long"]}
        for v in (0.025, 0.05, 0.075, 0.10)
        for s in (30.0, 40.0, 50.0, 60.0)
    ],
    (2, "long"): [
        {"vrp_threshold": v, "ivr_short": DECK_PARAMS[2]["ivr_short"], "ivr_long": l}
        for v in (0.025, 0.05, 0.075, 0.10)
        for l in (40.0, 50.0, 60.0, 70.0)
    ],
}

# Parameters that actually drive each side, used when reporting selections.
ACTIVE_PARAMS = {
    (1, "short"): ("z_window", "z_threshold"),
    (1, "long"): ("z_window", "z_threshold"),
    (2, "short"): ("vrp_threshold", "ivr_short"),
    (2, "long"): ("vrp_threshold", "ivr_long"),
}

BURN_IN_SESSIONS = 365          # first year reserved for the initial fit
REBALANCE_SESSIONS = 30         # refit roughly monthly
MIN_TRAIN_TRADES = 20           # minimum in-sample trades to trust a candidate


# ----------------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------------

OPTIONS_CSV = "data/options_df.csv"
SESSIONS_CSV = "data/sessions.csv"


def load_inputs():
    btc = pd.read_csv("data/btc_prices.csv")
    btc = btc.rename(columns={btc.columns[0]: "time"})
    btc["time"] = pd.to_datetime(btc["time"], utc=True)
    btc = btc.sort_values("time").reset_index(drop=True)
    if "log_returns" not in btc.columns:
        btc["log_returns"] = np.log(btc["close"] / btc["close"].shift(1))

    options = pd.read_csv(OPTIONS_CSV)
    options["session_start"] = pd.to_datetime(options["session_start"], utc=True)
    options["expiry"] = pd.to_datetime(options["expiry"], utc=True)

    sessions = pd.read_csv(SESSIONS_CSV)
    sessions["session_start"] = pd.to_datetime(sessions["session_start"], utc=True)

    return btc, options, sessions


def build_rv_benchmark(btc):
    """Day-of-week adjusted realised volatility, using past information only.

    mu_DOW is an average of *forward* one-day realised vol over past
    observations. A forward-looking series observed at time u spans returns
    u+1 .. u+24, so it is only fully realised at u+24. Every average is
    therefore lagged by at least 24 hours before it enters a feature.
    """
    btc = btc.copy()
    sq = btc["log_returns"] ** 2

    btc["rv_1d"] = np.sqrt(sq.rolling(RV_WINDOW).mean()) * np.sqrt(HOURS_PER_YEAR)

    # Forward one-day realised vol: returns t+1 .. t+24.
    btc["rv_1d_forward"] = (
        np.sqrt(sq.shift(-1).rolling(24).mean().shift(-23)) * np.sqrt(HOURS_PER_YEAR)
    )

    btc["hour"] = btc["time"].dt.hour
    btc["dow"] = btc["time"].dt.dayofweek

    # Same (day-of-week, hour) group: one step back is 168 hours, already realised.
    btc["mu_dow"] = (
        btc.groupby(["dow", "hour"])["rv_1d_forward"]
        .transform(
            lambda x: x.shift(1)
            .rolling(DOW_HIST_WINDOW, min_periods=DOW_MIN_PERIODS)
            .mean()
        )
    )

    # All-days baseline: consecutive hourly observations, so lag a full 24 hours.
    btc["mu_all"] = (
        btc["rv_1d_forward"]
        .shift(24)
        .rolling(ALL_HIST_WINDOW, min_periods=ALL_MIN_PERIODS)
        .mean()
    )

    btc["dow_multiplier"] = btc["mu_dow"] / btc["mu_all"]
    btc["rv_benchmark"] = (
        btc["rv_1d"] * btc["dow_multiplier"] / btc["dow_multiplier"].shift(24)
    )

    return btc[["time", "close", "rv_1d", "rv_benchmark"]]


def build_session_features(btc, options, sessions):
    rv = build_rv_benchmark(btc).rename(columns={"time": "session_start"})

    atm = (
        options[options["leg"].isin(["atm_call", "atm_put"])]
        .groupby("session_start")["entry_iv"]
        .mean()
        .rename("atm_iv")
    )

    feat = sessions.merge(atm, on="session_start", how="inner")
    feat = feat.merge(
        rv[["session_start", "rv_1d", "rv_benchmark"]], on="session_start", how="left"
    )
    feat = feat.sort_values("session_start").reset_index(drop=True)

    # IV rank against the trailing 52 weeks of the same session type. The current
    # session's own IV is observed at entry, so it is included in the range.
    feat["iv_rank"] = np.nan
    for stype, idx in feat.groupby("session_type").groups.items():
        sub = feat.loc[idx].set_index("session_start")["atm_iv"]
        roll = sub.rolling(IV_RANK_WINDOW, min_periods=IV_RANK_MIN_PERIODS)
        low, high = roll.min(), roll.max()
        rank = np.where(
            (high - low).abs() < 1e-12, 50.0, (sub - low) / (high - low) * 100.0
        )
        feat.loc[idx, "iv_rank"] = rank

    feat["vrp"] = feat["atm_iv"] - feat["rv_benchmark"]
    return feat


def add_zscore(feat, window):
    """VRP z-score against the trailing `window` sessions, excluding the current one."""
    roll = feat["vrp"].rolling(window, min_periods=max(10, window // 2))
    mean = roll.mean().shift(1)
    std = roll.std(ddof=1).shift(1)
    return np.where(std.abs() < 1e-12, np.nan, (feat["vrp"] - mean) / std)


# ----------------------------------------------------------------------------
# Per-session P&L for each structure, independent of the signal
# ----------------------------------------------------------------------------

def build_session_pnl(options, btc):
    """P&L in USD for one unit of each structure, for every session.

    The entry premium is quoted in BTC and is received (or paid) at entry, so it
    is converted at the spot price at session open. The exit value is the USD
    intrinsic value at expiry.
    """
    spot_at_expiry = (
        btc[["time", "close"]]
        .rename(columns={"time": "expiry", "close": "spot_at_expiry"})
    )

    legs = options.merge(spot_at_expiry, on="expiry", how="left")
    if legs["spot_at_expiry"].isna().any():
        missing = legs.loc[legs["spot_at_expiry"].isna(), "expiry"].unique()
        raise ValueError(f"Missing BTC prices for expiries: {missing}")

    intrinsic = np.where(
        legs["option_type"] == "C",
        np.maximum(legs["spot_at_expiry"] - legs["strike"], 0.0),
        np.maximum(legs["strike"] - legs["spot_at_expiry"], 0.0),
    )
    entry_usd = legs["entry_price"] * legs["spot_at_open"]
    legs["leg_pnl_long"] = intrinsic - entry_usd

    is_atm = legs["leg"].str.startswith("atm")

    # Short iron butterfly: short the ATM straddle, long the OTM strangle.
    butterfly_sign = np.where(is_atm, -1.0, 1.0)
    # Long straddle: the two ATM legs only.
    straddle_sign = np.where(is_atm, 1.0, 0.0)

    legs["pnl_short_butterfly"] = legs["leg_pnl_long"] * butterfly_sign
    legs["pnl_long_straddle"] = legs["leg_pnl_long"] * straddle_sign

    pnl = legs.groupby("session_start")[
        ["pnl_short_butterfly", "pnl_long_straddle"]
    ].sum()
    return pnl.reset_index()


# ----------------------------------------------------------------------------
# Signals and performance
# ----------------------------------------------------------------------------

def signals_for(feat, strategy, params):
    """Boolean short/long entry masks for a strategy under one parameter set."""
    if strategy == 1:
        z = add_zscore(feat, params["z_window"])
        thr = params["z_threshold"]
        short = pd.Series(z > thr, index=feat.index).fillna(False)
        long = pd.Series(z < -thr, index=feat.index).fillna(False)
    elif strategy == 2:
        v, ivr = feat["vrp"], feat["iv_rank"]
        short = (v > params["vrp_threshold"]) & (ivr > params["ivr_short"])
        long = (v < -params["vrp_threshold"]) & (ivr < params["ivr_long"])
        short, long = short.fillna(False), long.fillna(False)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return short.to_numpy(), long.to_numpy()


def sharpe(pnl, years):
    """Annualised Sharpe of a per-trade P&L series, scaled by realised trade frequency.

    Returns (sharpe, standard_error, n_trades). The standard error of an
    annualised Sharpe scaled this way is approximately 1/sqrt(years), and so is
    governed by the length of the sample rather than the number of trades.
    """
    pnl = pd.Series(pnl).dropna()
    n = len(pnl)
    if n < 2 or years <= 0 or pnl.std(ddof=1) == 0:
        return np.nan, np.nan, n
    ann = np.sqrt(n / years)
    return pnl.mean() / pnl.std(ddof=1) * ann, 1.0 / np.sqrt(years), n


def evaluate(feat, pnl, strategy, params, mask=None):
    """Sharpe, mean P&L and win rate for both sides over the selected sessions."""
    short, long = signals_for(feat, strategy, params)
    if mask is not None:
        short, long = short & mask, long & mask

    span = feat["session_start"]
    if mask is not None and mask.any():
        span = span[mask]
    years = max((span.max() - span.min()).days / 365.25, 1e-9)

    out = {}
    for side, sel, col in (
        ("short", short, "pnl_short_butterfly"),
        ("long", long, "pnl_long_straddle"),
    ):
        series = pnl.loc[sel, col]
        sh, se, n = sharpe(series, years)
        out[side] = {
            "sharpe": sh,
            "se": se,
            "n": n,
            "mean_pnl": series.mean() if n else np.nan,
            "win_rate": (series > 0).mean() if n else np.nan,
            "total_pnl": series.sum() if n else 0.0,
        }
    return out


# ----------------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------------

def walk_forward(feat, pnl, strategy, side):
    """Expanding-window walk-forward over the parameter grid for one side.

    At each rebalance the parameters are chosen on sessions strictly before the
    block, then applied to the block. Returns the concatenated out-of-sample
    P&L and the parameter selected for each block.
    """
    col = "pnl_short_butterfly" if side == "short" else "pnl_long_straddle"
    n_sessions = len(feat)

    # Precompute signals for every candidate once, then slice by block.
    candidates = []
    for params in GRIDS[(strategy, side)]:
        short, long = signals_for(feat, strategy, params)
        candidates.append((params, short if side == "short" else long))

    oos_pnl, oos_dates, chosen = [], [], []

    for start in range(BURN_IN_SESSIONS, n_sessions, REBALANCE_SESSIONS):
        end = min(start + REBALANCE_SESSIONS, n_sessions)
        train_idx = np.arange(start)
        test_idx = np.arange(start, end)

        train_span = feat["session_start"].iloc[train_idx]
        train_years = max((train_span.max() - train_span.min()).days / 365.25, 1e-9)

        best_i, best, best_sh = None, None, -np.inf
        for i, (params, sig) in enumerate(candidates):
            series = pnl.iloc[train_idx].loc[sig[train_idx], col]
            if len(series) < MIN_TRAIN_TRADES:
                continue
            sh, _, _ = sharpe(series, train_years)
            if np.isfinite(sh) and sh > best_sh:
                best_i, best, best_sh = i, params, sh

        if best_i is None:
            best, best_sh = DECK_PARAMS[strategy], np.nan
            _, sig = signals_for(feat, strategy, best)[:: 1 if side == "long" else -1]
        else:
            sig = candidates[best_i][1]

        fired = test_idx[sig[test_idx]]
        oos_pnl.extend(pnl.iloc[fired][col].tolist())
        oos_dates.extend(feat["session_start"].iloc[fired].tolist())
        chosen.append(
            {
                "block_start": feat["session_start"].iloc[start],
                "train_sessions": len(train_idx),
                "train_sharpe": best_sh,
                "n_trades": len(fired),
                **best,
            }
        )

    return pd.Series(oos_pnl, index=pd.DatetimeIndex(oos_dates)), pd.DataFrame(chosen)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def grid_distribution(feat, pnl, strategy, side):
    """In-sample Sharpe for every parameter set in the grid.

    Locates the published parameters within the distribution of everything that
    could have been tried, which is the size of the selection bias.
    """
    col = "pnl_short_butterfly" if side == "short" else "pnl_long_straddle"
    span = feat["session_start"]
    years = max((span.max() - span.min()).days / 365.25, 1e-9)

    rows = []
    for params in GRIDS[(strategy, side)]:
        short, long = signals_for(feat, strategy, params)
        sig = short if side == "short" else long
        sh, _, n = sharpe(pnl.loc[sig, col], years)
        rows.append({"sharpe": sh, "n": n, **params})
    return pd.DataFrame(rows).dropna(subset=["sharpe"])


def fmt_side(res):
    if not res["n"]:
        return "   no trades"
    return (
        f"n={res['n']:4d}  Sharpe={res['sharpe']:6.2f} "
        f"(+/-{res['se']:.2f})  mean=${res['mean_pnl']:8.1f}  "
        f"win={100 * res['win_rate']:5.1f}%  total=${res['total_pnl']:>10,.0f}"
    )


def main():
    btc, options, sessions = load_inputs()
    feat = build_session_features(btc, options, sessions)
    pnl = build_session_pnl(options, btc)

    feat = feat.merge(pnl, on="session_start", how="inner")
    feat = feat.sort_values("session_start").reset_index(drop=True)
    pnl = feat[["pnl_short_butterfly", "pnl_long_straddle"]]

    span = feat["session_start"]
    print(
        f"Sessions: {len(feat)}  "
        f"{span.min():%Y-%m-%d} to {span.max():%Y-%m-%d}  "
        f"({(span.max() - span.min()).days / 365.25:.2f} years)"
    )

    oos_mask = np.zeros(len(feat), dtype=bool)
    oos_mask[BURN_IN_SESSIONS:] = True
    oos_span = feat.loc[oos_mask, "session_start"]
    print(
        f"Walk-forward period: {oos_span.min():%Y-%m-%d} to {oos_span.max():%Y-%m-%d} "
        f"({oos_mask.sum()} sessions), refit every {REBALANCE_SESSIONS} sessions\n"
    )

    for strategy in (1, 2):
        params = DECK_PARAMS[strategy]
        print(f"{'=' * 78}\nSTRATEGY {strategy}   published parameters: {params}\n{'=' * 78}")

        full = evaluate(feat, pnl, strategy, params)
        print("  Published parameters, full sample (in-sample):")
        for side in ("short", "long"):
            print(f"    {side:5s} {fmt_side(full[side])}")

        held = evaluate(feat, pnl, strategy, params, mask=oos_mask)
        print("\n  Published parameters, walk-forward period only (no refitting):")
        for side in ("short", "long"):
            print(f"    {side:5s} {fmt_side(held[side])}")

        print("\n  Walk-forward, parameters reselected on past data only:")
        for side in ("short", "long"):
            oos, chosen = walk_forward(feat, pnl, strategy, side)
            years = max((oos_span.max() - oos_span.min()).days / 365.25, 1e-9)
            sh, se, n = sharpe(oos, years)
            res = {
                "sharpe": sh,
                "se": se,
                "n": n,
                "mean_pnl": oos.mean() if n else np.nan,
                "win_rate": (oos > 0).mean() if n else np.nan,
                "total_pnl": oos.sum() if n else 0.0,
            }
            print(f"    {side:5s} {fmt_side(res)}")

            keys = list(ACTIVE_PARAMS[(strategy, side)])
            stability = chosen[keys].astype(str).agg(" ".join, axis=1).value_counts()
            print(
                f"          parameters selected across {len(chosen)} blocks: "
                f"{len(stability)} distinct, most common "
                f"'{stability.index[0]}' in {stability.iloc[0]}"
            )
            chosen.to_csv(f"data/wf_params_s{strategy}_{side}.csv", index=False)
            oos.rename("pnl").to_csv(f"data/wf_pnl_s{strategy}_{side}.csv")

        print("\n  Full-sample Sharpe across the whole parameter grid:")
        for side in ("short", "long"):
            grid = grid_distribution(feat, pnl, strategy, side)
            grid = grid.drop_duplicates(subset=list(ACTIVE_PARAMS[(strategy, side)]))
            published = full[side]["sharpe"]
            pct = 100.0 * (grid["sharpe"] <= published).mean()
            print(
                f"    {side:5s} {len(grid):3d} sets  "
                f"min={grid['sharpe'].min():5.2f}  median={grid['sharpe'].median():5.2f}  "
                f"max={grid['sharpe'].max():5.2f}  |  published={published:5.2f} "
                f"({pct:.0f}th percentile)"
            )
        print()

    print(
        f"Note: with {(span.max() - span.min()).days / 365.25:.2f} years of data the "
        f"standard error on an annualised Sharpe is about "
        f"{1 / np.sqrt((span.max() - span.min()).days / 365.25):.2f}, "
        "regardless of trade count."
    )


if __name__ == "__main__":
    main()
