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

> This is the third and (so far) final version of these numbers. Two earlier versions were superseded by bugs found and fixed in `scripts/run_cv_window.py`: (1) predictions silently omitted the `beta_xG * log(xG)` term even when trained with `use_xG=True`; (2) a separately-rebuilt team→ID mapping could silently swap two teams' learned values; (3) `use_dixon_coles=True` fits `rho_dc` during training, but nothing evaluating predictions applied it — log-likelihood and RPS/calibration were computed as if Dixon-Coles were always off. All three are now fixed by routing training and prediction through one shared module, `src/football_model/model/predict.py` (see its docstring). This run reflects all three fixes.

| Metric | Value |
|---|---|
| MAE | 0.921 ± 0.149 goals/match |
| Mean LL improvement over naive | 1.60 ± 1.92 (bootstrap 95% CI: [0.99, 2.23]) |
| Windows individually beating naive on LL | 28/35 (80%) |
| Pooled RPS, 401 test matches | Model: 0.196 vs. Naive: 0.234 |
| **RPS improvement over naive** | **0.038, 95% CI [0.026, 0.050] — excludes zero** |

**The RPS result is the headline finding**: it's the first properly-powered, statistically significant result in this project's validation history. Everything measured before this (single small holdouts, unpooled log-likelihood comparisons) sat inside its own noise band — this doesn't. Both the LL-improvement bootstrap CI and the RPS-improvement bootstrap CI exclude zero, so this isn't a fluke of one lucky window. Worth noting: the Dixon-Coles fix moved individual windows' LL by a real amount (e.g. window 8: 5.94 → 6.06) but left pooled RPS essentially unchanged (0.1963 either way) — its correction is concentrated on a few rare low-score cells, enough to matter to LL, not enough to move an RPS average computed across 401 matches.

Worth being upfront about the other side of that: **7 of 35 windows (20%) individually score *worse* than the naive baseline** on log-likelihood (windows 10, 11, 17, 22, 23, 29, 32 — range across all windows is -2.09 to +6.06). The model wins on average, clearly, but "wins every single week" is not an accurate claim to make from this data — window-to-window variance is real and non-trivial.

### Calibration (pooled across all 401 CV test matches)

| Predicted P(home win) | n | Predicted | Actual |
|---|---|---|---|
| 0.0-0.1 | 7 | 0.10 | 0.00 |
| 0.1-0.2 | 64 | 0.20 | 0.17 |
| 0.2-0.4 | 96 | 0.32 | 0.35 |
| 0.4-0.5 | 114 | 0.44 | 0.46 |
| 0.5-0.6 | 65 | 0.56 | 0.52 |
| 0.6-0.8 | 40 | 0.67 | 0.75 |
| 0.8-0.9 | 14 | 0.78 | 0.93 |

(A 0.9-1.0 bucket with n=1 is omitted as uninformative.) Calibration is reasonably tight through the middle of the range (0.1-0.2 through 0.5-0.6 all sit within ~4 points of their bucket's actual frequency). The same **under-confidence at high home-win probability** shows up again in the two top buckets (0.6-0.8: predicted 0.67 vs. actual 0.75; 0.8-0.9: predicted 0.78 vs. actual 0.93) — consistent with every prior version of this analysis. Leading hypothesis is still over-shrinkage of `home_adv` or attack/defence for stand-out teams; still not root-caused.

## Known limitations / open questions

- **xG and Dixon-Coles ablation is now underway**: [WP002](../wp002_xg_dc_ablation/) isolates how much each contributes individually (both/neither/xG-only/DC-only) — this result alone doesn't say how much either feature adds vs. just having 6 seasons of data instead of 1-2.
- **Not benchmarked against real bookmaker odds.** "Beats naive" is a real, statistically-supported claim. "Competitive with the market" is a different, unanswered question — no odds data exists in this repo.
- **The home-win-confidence calibration gap** (above) is unresolved.
- **Live data source, not a frozen snapshot**: `get_understat_data` fetches current Understat data on every run, so re-running this notebook later (once more of the season has played out, or if historical match data gets corrected upstream) can shift these numbers slightly even with nothing in the model or code changed — that's the data source, not non-determinism in training (each window's `pm.sample` uses a fixed seed).
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

Run top to bottom. `cv_checkpoint.pkl` and `cv_shared_data.pkl` in this folder already hold the completed 35-window run above (reflecting all three fixes) — re-running the CV cell will detect them and report "35/35 windows already completed." Delete `cv_checkpoint.pkl` to force a clean restart.
