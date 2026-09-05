<!doctype html>
# WP001 — Walk-Forward Cross-Validation Baseline

First statistically-backed validation of the Bayesian hierarchical football model: a 35-window walk-forward cross-validation across six EPL seasons, evaluated with proper scoring rules instead of ad-hoc metrics. This is the notebook's supporting commentary and results, pulled out so `wp001_walkforward_cv.ipynb` stays a lean, runnable pipeline.

## What this is

Rolling-window cross-validation: train on rounds 1–N, predict round N+1, repeat for many different N. Each window is a genuinely out-of-sample test — the model never sees the round it's predicting.

- **Data**: EPL, 2020-21 through 2025-26 (6 seasons, 208 total rounds, 5 relegation/promotion boundaries)
- **Config**: `clip_theta=5.0`, `center_team_strength=False`, `use_dixon_coles=True`, `use_xG=True`
- **Windowing**: training starts a few rounds into the second season (so every window straddles a relegation/promotion boundary), stepping forward 5 rounds at a time → **35 windows**, `test_window=1` (each window predicts exactly one round ahead)
- **Execution**: each window fits and evaluates in its own subprocess (`scripts/run_cv_window.py`), checkpointing to `cv_checkpoint.pkl` after every window — a hung or crashed window gets retried on the next run instead of losing the whole sweep (see "Why subprocess isolation" below)

## Results

| Metric | Value |
|---|---|
| MAE | 0.913 ± 0.153 goals/match |
| Mean LL improvement over naive | 1.27 ± 1.57 (bootstrap 95% CI: [0.78, 1.79]) |
| Windows individually beating naive on LL | 29/35 (83%) |
| Pooled RPS, 401 test matches | Model: 0.198 vs. Naive: 0.234 |
| **RPS improvement over naive** | **0.036, 95% CI [0.026, 0.046] — excludes zero** |

**The RPS result is the headline finding**: it's the first properly-powered, statistically significant result in this project's validation history. Everything measured before this (single small holdouts, unpooled log-likelihood comparisons) sat inside its own noise band — this doesn't.

### Calibration (pooled across all 401 CV test matches)

| Predicted P(home win) | n | Predicted | Actual |
|---|---|---|---|
| 0.0-0.1 | 3 | 0.11 | 0.00 |
| 0.1-0.2 | 53 | 0.20 | 0.15 |
| 0.2-0.4 | 118 | 0.31 | 0.33 |
| 0.4-0.5 | 130 | 0.43 | 0.48 |
| 0.5-0.6 | 72 | 0.56 | 0.63 |
| 0.6-0.8 | 22 | 0.68 | 0.86 |
| 0.8-0.9 | 3 | 0.81 | 1.00 |

Well calibrated for close-to-even predictions (0.2-0.4 bucket: predicted 0.31, actual 0.33 — spot on). Systematically **under-confident** once it already favors a home win — a growing gap across three consecutive buckets (0.4-0.5 → 0.5-0.6 → 0.6-0.8), not a single noisy bin. Not yet root-caused; leading hypothesis is over-shrinkage of `home_adv` or attack/defence for stand-out teams. Small-n bins (0.0-0.1, 0.8-0.9, n=3 each) shouldn't be over-read.

## Known limitations / open questions

- **xG and Dixon-Coles are enabled together, un-ablated.** This result doesn't say how much either feature individually contributes — could be mostly xG, mostly Dixon-Coles, or mostly just having 6 seasons of data instead of 1-2. A follow-up ablation (xG-only, Dixon-Coles-only, neither) would answer this.
- **Not benchmarked against real bookmaker odds.** "Beats naive" is a real, statistically-supported claim. "Competitive with the market" is a different, unanswered question — no odds data exists in this repo.
- **The home-win-confidence calibration gap** (above) is unresolved.
- **Pooled RPS naive baseline is computed over the whole CV dataset** (all seasons, train+test rounds combined) for simplicity, not the strictly leakage-free per-window training-only average that the per-window LL comparison (`ll_naive` in `run_cv_window.py`) already correctly uses. In practice the two give nearly identical numbers here (league scoring rate is stable across seasons), and any bias this introduces runs *against* the model (a baseline with slight foresight looks better, not worse), so it doesn't inflate the headline result — but it's worth knowing if replicating this exactly.

## Why subprocess isolation for the CV loop

Earlier attempts ran all 35 windows sequentially inside one long-lived Jupyter kernel and reliably stalled partway through (once past window 14, CPU dropped to ~9% — a genuine hang, not just a slow NUTS tree search). The likely cause: each window rebuilds the model at a different shape (`n_teams`/`n_time` both grow every window), so `nuts_sampler="numpyro"` JIT-recompiles from scratch every time — accumulating 20-30 different compiled JAX programs, thread-pool state, and memory inside one process eventually breaks something that a single one-off fit never triggers. Clearing JAX's cache and forcing garbage collection between windows didn't fix it. Running each window as a disposable subprocess does: nothing has the chance to accumulate, because there's no "across windows" left inside a single process for it to accumulate in.

## Metrics glossary

- **MAE (Mean Absolute Error)** — average goals-off per prediction. Lower is better. Only judges goal counts, not who wins.
- **Log-Likelihood (LL)** — how probable the model considered what actually happened, summed across matches. Always negative; closer to zero is better; only comparisons (not the raw number) mean anything. `exp(difference)` gives a "Bayes Factor" — treat anything under ~20 as weak evidence, not proof.
- **RPS (Ranked Probability Score)** — the metric this project trusts most for match predictions. Scores win/draw/loss probabilities, and unlike LL it knows outcomes are *ordered* (predicting "draw" when away won is a smaller miss than predicting "home win" when away won). 0 = perfect, ~0.2 ≈ solid football-forecasting model.
- **Calibration** — "is 70% actually 70%?" Bucket predictions by stated probability, compare to actual outcome rate in that bucket. Separate from accuracy; essential for anything betting-related.
- **Bootstrap confidence interval** — resamples results many times to see how much an "improvement" number wanders. If the range excludes zero, it's likely real; if it includes zero, it's not yet distinguishable from noise.

## Reproducing

```bash
cd work_products/wp001_walkforward_cv_baseline
jupyter lab wp001_walkforward_cv.ipynb
```

Run top to bottom. `cv_checkpoint.pkl` and `cv_shared_data.pkl` in this folder already hold the completed 35-window run above — re-running the CV cell will detect them and report "35/35 windows already completed" rather than re-running everything. Delete `cv_checkpoint.pkl` to force a clean restart.
