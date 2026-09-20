# BTC Options Volatility Risk Premium — Oxford Alpha Fund (Hilary 2026)

Does a volatility risk premium exist in short-dated BTC options, and can it be
traded after costs?

**Short answer.** The premium exists and is statistically significant. It is
marginal after realistic execution cost, and the strategy that harvests it has
been losing money in 2026.

---

## The strategy

**Premium-filtered short straddle on Deribit daily expiries.**

At 08:00 UTC, for the expiry 24 hours out:

1. Take the ATM call and put — strike closest to spot at open — priced from the
   trade nearest 08:00 within a two-hour window.
2. Compute the expected variance premium, `VRP̂ = σ_imp − σ̂_realised`, where
   `σ_imp` is the average of the two ATM implied vols and `σ̂_realised` is a
   day-of-week adjusted forecast built from trailing hourly realised vol.
3. **Enter only if both hold:**
   - `VRP̂ > 0` — implied exceeds the forecast.
   - The straddle premium, as a share of spot, is above its **q-th percentile
     over the trailing W sessions**.
4. Sell both legs. Hold to expiry. No hedge, no stop, no wings.

`W` and `q` are the only fitted quantities. Everything else is fixed in advance.

### Why a premium filter

It falls out of the fee schedule rather than out of the data. Deribit charges a
near-fixed amount per contract, so the fee is a roughly constant **0.074% of
spot** while the edge scales with implied vol. When vol is low the fee consumes
the entire edge — 97% of it in 2026. The implication is mechanical: only trade
when the premium is large enough to absorb a fixed cost.

A *trailing percentile* rather than an absolute level is deliberate. An absolute
threshold is an implied-vol level in disguise: the 2.6%-of-spot variant put 187
of its 303 trades in 2021 and 5 in 2026. The percentile adapts to the regime and
trades 18–74 times in every year of the sample.

---

## Headline result

**Walk-forward, net of exchange fees, out of sample: Sharpe 1.14, t = 2.45**,
over 4.60 years and 348 trades. Parameters re-selected every 60 sessions by grid
search on strictly prior data — 29 refits.

| Basis | Sharpe | t | Note |
|---|---|---|---|
| Walk-forward, net of exchange fees | **1.14** | **2.45** | the number to quote |
| Same window, filter switched off | 0.46 | — | untuned control |
| Walk-forward, after fees **and** measured spread | 0.70 | 1.49 | not significant |
| Same, filter switched off | −0.18 | — | negative |

The result is insensitive to the walk-forward start: 1.14 / 1.12 / 1.15 at
burn-ins of 365 / 550 / 730 sessions. The 365 figure is quoted because it gives
the longest out-of-sample span, not the best number — 730 is marginally higher.

**Exchange fees are included in the 1.14** — entry fees and settlement fees. What
is not included is the spread and the full execution model. See *Limits* below.

---

## Method

| Step | Choice |
|---|---|
| Search | Exhaustive grid, 15 candidates: `W ∈ {60, 120, 250}` × `q ∈ {0.3…0.7}` |
| Objective | Net-of-fee Sharpe on the training slice |
| Validation | Expanding-window walk-forward, refit every 60 sessions |
| Control | Same strategy, filter removed — a zero-parameter rule that cannot be overfitted |

All 15 grid cells beat the unfiltered control, spanning 1.06 to 1.76 in sample.
The presented configuration (`W=250, q=0.70`) is deliberately not the peak: a
result that survives across the whole surface is a property of the idea, whereas
a lone peak is a fitted one.

Annualisation is by realised trade frequency, `Sharpe = mean/sd × √(N/years)`.
The standard error is `≈ 1/√years` — set by sample **length**, not trade count,
so ±0.42 in sample and ±0.47 out. Trading more often buys no confidence.

---

## Data

| Source | Coverage |
|---|---|
| Coinbase Exchange — hourly BTC-USD | 2021-01-01 → 2026-09-09, 49,843 bars |
| Deribit public trade history — full traded chain per session | 2,076 sessions, 37,096 rows |

The original study ran 2023–2025. Deribit serves daily expiries from January
2021, which nearly doubles the sample and cuts the Sharpe standard error from
±0.58 to ±0.42.

**Deribit lists only a narrow strike band on daily expiries.** Probing every
strike from 0.60× to 1.60× spot on the 12APR23 expiry returns 20 call strikes
spanning 0.863×–1.079×. The ±15% iron condor described in the source literature
is **not constructible** on this venue — a finding in its own right, and the
reason this study trades a bare straddle.

---

## Limits

**Execution.** Exchange fees are modelled. Bid-ask is measured (median effective
spread 8.3% of premium, from aggressor-tagged trades on 245 instruments) and
applied as a sensitivity, taking the result to 0.70. Not modelled at all:
slippage beyond that spread, market impact, fill probability, partial fills,
legging risk.

**Portfolio.** No capital, margin, position sizing, liquidation, daily
mark-to-market or return on collateral. This is dollar P&L on one contract, so
it is **not a return-based Sharpe**, and the maximum drawdown is uninterpretable
without a collateral base. The specific hazard left unmodelled: Deribit margin
on short options rises with volatility, i.e. exactly when the position is losing.

**Decay.** 2026 is negative — 40 trades, −$54 each. First half of the sample
2.30, second half 1.38.

**Tail.** Short volatility with no wings. Skew −1.43, worst trade −$4,809 net
against a median gain of $364.

**Inherited parameters.** The 08:00 session boundary, two-hour entry window,
24-hour RV window, 12-observation day-of-week window and choice of daily expiry
were fixed once on the original sample and never re-selected. They outnumber the
two parameters fitted here, and nothing bounds their contribution.

---

## Corrections to the original study

Three defects were found and fixed in the rebuilt pipeline. The original scripts
are unmodified and still contain them.

1. **Look-ahead in the realised-vol benchmark.** A forward-looking average was
   lagged one hour when it needed 24, putting up to 23 hours of a session's own
   outcome inside the feature predicting it.
2. **Entry premium converted at the wrong spot.** Premium is quoted in BTC and
   received at entry, so it converts at spot at open — the original used spot at
   expiry, a price unknown at trade time.
3. **Degenerate early IV rank.** A one-observation minimum let the earliest
   sessions rank against a zero-width range.

The deck also states the day-of-week adjustment as `μ_DOW(t)/μ_DOW(t−1)` while
the code normalises each factor by an all-days baseline first. The two differ by
a factor with median 0.99991 and agree on the sign of `VRP̂` in 98.45% of
sessions. Documented, not yet reconciled in code.

---

## Repository

### Current pipeline

```
collect_chain.py        full traded option chain per session  -> data/chain_df.csv
collect_extended.py     four-leg legs, 2021-2026              -> data/options_df_ext.csv
walkforward.py          leak-corrected features, walk-forward machinery
strategy.py             the strategy end to end, all reported figures
optimise.py             grid search + walk-forward validation
audit_extended.py       coverage, fee model, null baselines
compare_structures.py   straddle vs three butterfly variants + bootstrap
```

### Original study — retained, unmodified

```
dataCollection.py            Coinbase spot ingest
options_data_collection.py   Deribit leg identification
iv.py / iv2.py               IV rank, VRP, signal generation
signals.py / signals2.py     threshold and z-score signals
trade_construction.ipynb     original backtest
```

---

## Running it

```bash
pip install -r requirements.txt

python dataCollection.py     # hourly spot
python collect_chain.py      # full chain, ~35 min, checkpoints every 60 sessions
python strategy.py           # the strategy and its results
python optimise.py           # grid search and walk-forward
```

Collection is network-bound at roughly one session per second and resumes from
its checkpoint if interrupted. The analysis scripts run offline in seconds.

---

## Authors

Henry Huang · Harik Sodhi · Alphonsus Neo · Sikai Huang · Ishwar Karthik
