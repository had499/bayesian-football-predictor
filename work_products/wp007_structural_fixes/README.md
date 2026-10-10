# WP007 — Structural fixes: rigid dynamics, promoted teams, xG, the Dixon-Coles bug, forecasting

**Status: complete.** Full 9-arm × 30-window sweep finished 2026-09-22 (0 divergences, max rhat 1.029). One change survives — the xG measurement channel — and it's smaller and less robust than the diagnostics that motivated it suggested. Everything else in the ladder (posterior-averaged prediction, an explicit intercept, random-walk dynamics, the promoted-team offset, and Dixon-Coles even with its tau bug fixed) came back indistinguishable from noise. See "Results" below.

## Why this work product exists

WP004–WP006 ruled out the cheap fixes: recalibration recovers ~2% of the gap to Pinnacle, loosening priors and per-team sigma are within noise. The remaining gap (+0.0139 RPS behind Pinnacle) is *resolution*. A diagnostic pass over 12 seasons of Understat data (2014–2025, 4,560 matches) pointed at structure, not knobs:

| # | Diagnostic (computed on the data, not the model) | Finding |
|---|---|---|
| D1/D2 | Real year-on-year change in team attack (noise-corrected) vs. what the prior allows | ≈0.12 real (rough estimate) vs. ≈0.038 prior-implied; even WP005's loosest sigma reaches only ≈0.094. The limit is the *structure* (static `att_0` + AR(1) reverting to it, ~20-round half-life), not one sigma value |
| D9 | Newly promoted sides vs. continuing teams (33 team-seasons) | −0.37 attack, −0.26 defence; the model gives them `att_0 ~ N(0, 0.2)`, i.e. an average team |
| D4/D7/D8 | Is xG a better strength signal than goals? | First-half xG predicts second-half goals better than first-half goals (r 0.73 vs 0.68 attack, 0.62 vs 0.55 defence). But `goals ~ Poisson(xG)` would over-disperse (xG CV² ≈ 0.32), so xG needs its own observation model |
| D5 | Does the Dixon-Coles tau keep the joint distribution normalised? | **No — bug.** `tau(1,0)`/`tau(0,1)` used the wrong side's rate; the joint sums to `1 + ρ(λ−μ)²e^(−λ−μ)` (checked numerically against that formula) |
| D6 | Overall goal level under `center_team_strength=True` | No free parameter: implied 1.14 home / 1.00 away goals vs. ~1.54 / ~1.26 actual (the validated runs override to `center_team_strength=False`, so default and validated config were different models) |

## What changed

**Bug fixes (unconditional):**
* `dixon_coles_tau` (PyTensor, `model/components.py`, and numpy mirror, `model/predict.py`): swapped cells fixed to Dixon & Coles (1997) eq. 4. New regression tests assert the joint sums to exactly 1. **Consequence for earlier work:** WP001/WP002's `both` and `dc_only` arms were trained with the improper tau, so WP002's "Dixon-Coles does nothing" conclusion should be re-examined (arm `S2_dc_only_fixed` here does exactly that).

**New, opt-in `ModelConfig` options (defaults reproduce every earlier config):**
* `dynamics="random_walk"` (+ `rw_init_scale`): no static level, ρ fixed at 1.
* `use_promoted_effect` (+ `promoted_effect_sd`): a learned level shift applied once, in the round a team flips inactive→active, then carried by the walk (rather than a first-season offset that would have to vanish abruptly in season two). Implemented in `ar1_team_process(entry_shift=…)`.
* `use_intercept` (+ `intercept_mu/sd`): explicit log-rate intercept.
* `use_xg_channel` (+ `xg_channel_weight`, shape prior): realised match xG as a second, tempered **Gamma** measurement of the same latent rate (`components.xg_measurement_channel`). Needs the new `ModelData.xG_obs_home/away`.

**Forecasting (`model/predict.py`):** `forecast_latent`, `posterior_lambdas`, `posterior_outcome_probs`, `posterior_goals_loglik`, `scoreline_grid`. These score the posterior *average* of per-draw probabilities and roll the latent state forward through the forecast horizon (innovation variance, the season-opener sigma multiplier, promotion entry shift, re-centering over that round's active set). The old path scored posterior *means* at the frozen last trained state.

**Harness:** `scripts/run_cv_window.py` gains `--predict-mode plugin|posterior` (default `plugin` = WP001–006 behaviour), sampler flags, per-match `home/away/round/p_*/rps/ll_goals`, and per-window `param_means`/divergences/max-rhat. Config overrides now win over harness defaults (previously a duplicate key raised `TypeError`). `prepare_model_data`'s xG-baseline loop is vectorised (bit-identical output, verified against the original on real data).

## The experiment (as designed)

Cumulative ladder — one change per rung, all scored on identical test matches:

| Arm | Change |
|---|---|
| `L0_baseline` | Repo config as validated (xG/DC off per WP002), plug-in prediction |
| `L1_posterior` | + posterior-averaged, horizon-aware prediction (same fit as L0) |
| `L2_level` | + intercept, hard centering, home-advantage priors the data can move |
| `L3_randomwalk` | + random-walk strengths |
| `L4_promoted` | + promotion entry shift |
| `L5_xgchannel` | + xG measurement channel (weight 0.5) |
| `L6_dixoncoles` | + (fixed) Dixon-Coles |
| `S1_xgweight1` | L5 with xG weight 1.0 |
| `S2_dc_only_fixed` | L0 + fixed Dixon-Coles only |

Data: 12 EPL seasons of history; test period 2020-21 onward (matches WP001–006); 30 windows, each forecasting the next 4 rounds → **1,314 test matches** (vs. 401 in WP001). Sampler for all arms: 600 draws / 600 tune / 4 chains / target_accept 0.9. Comparisons report both a match-level bootstrap CI and a **window-cluster** bootstrap CI (matches within a window share one fit and aren't independent, so the match-level interval is too narrow; the cluster one is the conservative one).

## Results

Full sweep: 9 arms × 30 windows, 1,314 paired test matches, 2020-21 through 2025-26. All numbers below are RPS unless stated; lower is better; CIs are the conservative window-cluster bootstrap (matches inside one window share a single fit, so a plain match-level CI is too narrow — both are in the notebook, they agree throughout).

### The ladder — one change per rung

| Arm | RPS | Δ vs previous rung | Δ vs L0 (cumulative) | Verdict |
|---|---|---|---|---|
| `L0_baseline` (repo, as validated, plug-in prediction) | 0.2074 | — | — | — |
| `L1_posterior` (+ posterior-averaged, horizon-aware prediction) | 0.2075 | +0.0001 [−0.0002, +0.0004] | +0.0001 | no clear difference |
| `L2_level` (+ intercept, hard centering, wider HFA prior) | 0.2077 | +0.0002 [−0.0002, +0.0006] | +0.0004 | no clear difference |
| `L3_randomwalk` (+ random-walk dynamics) | 0.2075 | −0.0003 [−0.0012, +0.0007] | +0.0001 | no clear difference |
| `L4_promoted` (+ promotion entry shift) | 0.2077 | +0.0002 [−0.0003, +0.0006] | +0.0003 | no clear difference |
| `L5_xgchannel` (+ xG measurement channel, weight 0.5) | **0.2049** | **−0.0028 [−0.0048, −0.0007]** | **−0.0025 [−0.0046, −0.0003]** | **better** |
| `L6_dixoncoles` (+ fixed Dixon-Coles) | 0.2049 | +0.0000 [−0.0002, +0.0002] | −0.0025 [−0.0045, −0.0003] | no clear difference |

**Only `L5_xgchannel` clears the bar.** Every other structural change I hypothesized from the diagnostics — looser dynamics, an identifiable level, the promoted-team offset — moved RPS by less than its own confidence interval at 1,314 matches. That's a real result, just not the one the diagnostics predicted: the *direction* of each diagnostic held up (e.g. `L3_randomwalk`'s fitted `sigma_att` does imply a bigger 1-season strength swing, see "What the model learned" below), but none of them translated into a measurably better forecast.

**Caveat on `L5`'s robustness — substantially revised by WP013.** Tested in isolation (`L0` + xG channel only, `work_products/wp013_xg_channel_robustness/`), the effect holds in **6 of 6** leave-one-season-out folds, and the four other rungs bundled here add nothing (+0.00005 [−0.00085, +0.00096]). The fragility described below is the *bundle's*, not the xG channel's — prefer `L0` + `use_xg_channel=True`. Original caveat, which applies to the `L5` bundle: the −0.0025 to −0.0028 effect excludes zero pooled across all 6 seasons, but **does not survive dropping 2024 alone** (point estimate stays negative, −0.0016, but CI widens to [−0.0038, +0.0008]) — 2024 alone shows a clear effect (−0.0074, CI [−0.0096, −0.0043], see "Stability across seasons" below) and is doing much of the work in the pooled number. Treat this as a real but not yet fully robust effect, not a settled one — it would benefit from more seasons/windows before being trusted the way WP001's RPS result was.

### Side arms

| Comparison | ΔRPS | CI (cluster) | Verdict |
|---|---|---|---|
| DC fixed alone vs. old model (`S2_dc_only_fixed` vs `L0`) | +0.00006 | [−0.0001, +0.0002] | no clear difference |
| DC on top of full model (`L6` vs `L5`) | +0.00002 | [−0.0002, +0.0002] | no clear difference |
| xG weight 1.0 vs 0.5 (`S1_xgweight1` vs `L5`) | −0.00052 | [−0.0019, +0.0009] | no clear difference |

**Dixon-Coles does nothing here even with the tau bug fixed** — WP002's "not worth the complexity" verdict, which was reached under the buggy (improper) tau, holds up under the correct one too. **xG weight 0.5 (the default) is as good as 1.0** — no reason to increase it.

### What the model learned

Posterior means, median across windows (`L0`/`L1`/`L2` share one fit, hence identical numbers):

| | sigma_att | home_mu | promo_att | promo_def | xg_shape | xg_kappa |
|---|---|---|---|---|---|---|
| `L0`/`L1`/`L2` | 0.013 | 0.196–0.216 | — | — | — | — |
| `L3_randomwalk` | 0.012 | 0.217 | — | — | — | — |
| `L4_promoted` | 0.012 | 0.217 | −0.069 | −0.037 | — | — |
| `L5_xgchannel` | 0.014 | 0.224 | −0.048 | −0.057 | 3.05 | 1.01 |

Two things worth noting:
- **`L3_randomwalk`'s fitted sigma implies a 1-season change SD of ≈0.073–0.088** (`sigma_att × sqrt(38 rounds)`) — bigger than `L0`'s prior-implied ≈0.038 and moving toward the ≈0.12 data-based estimate from the diagnostic pass, exactly the direction predicted. It just didn't turn into a better forecast (see ladder above) — the model can move more, but moving more isn't what was limiting it.
- **The promotion shift shrank a lot**: diagnostics on real team-seasons found promoted sides average −0.37 attack / −0.26 defence vs. continuing teams; the model's single learned `promo_att`/`promo_def` (shared across every promoted team) lands at only −0.05 to −0.07. One shared parameter pooled across all promoted teams, constrained by `promoted_effect_sd=0.25`, can't reach the full effect a per-team fit would show — worth trying a wider prior or a per-team version before concluding the idea doesn't work.
- **`xg_shape` (≈3.05) lands almost exactly on the ≈3.11 Gamma shape estimated directly from the data** (diagnostic D8) — the xG channel's own measurement model is well-specified, which is some evidence the improvement it buys is a real information gain, not an artifact of a mis-set prior.

### Calibration

The under-confidence WP001 found at high home-win probability is still there, only slightly improved:

| Predicted P(home win) | `L0` actual | `L5` actual |
|---|---|---|
| 0.62–0.75 | 0.657 | 0.641 |
| 0.75–0.88 | 0.889 | 0.868 |

Both still under-confident in the top bucket (predicted ~0.79–0.80, actual 0.87–0.89); `L5` narrows the gap slightly but doesn't close it. Not root-caused by this work product.

### Where does the xG channel help?

| Subset | n | Δ (L5 − L0) | CI (cluster) |
|---|---|---|---|
| promoted side, season rounds 1–10 | 91 | −0.0007 | [−0.0087, +0.0074] |
| promoted side, round 11+ | 285 | −0.0022 | [−0.0065, +0.0018] |
| no promoted side | 938 | **−0.0028** | **[−0.0049, −0.0005]** |

Counter to what I'd have guessed, the clearest signal is in matches with **no** promoted side, not matches involving one — consistent with `L4_promoted` alone showing no effect. The xG channel is winning on ordinary matches, not disproportionately on the promoted-team edge case it was partly motivated by.

### Stability across seasons

| Season | n | Δ (L5 − L0) | CI (cluster) |
|---|---|---|---|
| 2020 | 223 | +0.0001 | [−0.0054, +0.0068] |
| 2021 | 220 | −0.0017 | [−0.0051, +0.0013] |
| 2022 | 205 | −0.0043 | [−0.0097, +0.0019] |
| 2023 | 228 | −0.0025 | [−0.0078, +0.0031] |
| **2024** | 219 | **−0.0074** | **[−0.0096, −0.0043]** |
| 2025 | 219 | +0.0004 | [−0.0021, +0.0029] |

Only 2024 individually clears zero. The other five seasons all point the same direction (negative, i.e. `L5` better) except 2020/2025 which are flat — no season points the *other* way — but none besides 2024 is individually significant. This is "a real average effect, unevenly distributed," the same texture WP001's original 29/35-windows-beat-naive result had, not "the model reliably wins every season."

### Sampler health

0 divergences and max rhat ≤ 1.029 across every one of the 270 (arm, window) fits — no sampling problems to caveat the results with. `L5`/`L6`/`S1` (the xG-channel arms) run ~30–75% slower per window than the others (up to ~450s/window for `S1`) from the extra Gamma likelihood terms, still well within a practical sweep.

### Pinnacle benchmark: see WP008

football-data.co.uk turned out to be blocked by network policy on the machine this ran on (a corporate proxy block page, not a real 503 from the server), so the notebook's market-comparison section (cell 16) here still reports "PENDING." **The benchmark itself was obtained separately, via a verified GitHub mirror, in [`work_products/wp008_market_benchmark/`](../wp008_market_benchmark/README.md).** Headline: `L5_xgchannel` closes 32% of `L0_baseline`'s gap to Pinnacle (both still reliably worse than the market) and reaches statistical parity with Pinnacle in 2 of 6 seasons — real progress, not the whole answer. See that work product for the full breakdown, including the disagreement-quantile pattern (unchanged from WP003) that accounts for most of what's left.

## Bottom line / recommendation

Adopt **`L5_xgchannel`** (`use_xg_channel=True`, weight 0.5, otherwise `L0`'s config) as the new candidate default — it's the only change here with a measured, if not yet fully season-robust, RPS improvement. Skip Dixon-Coles (confirmed not worth it, bug fixed or not) and the promoted-team offset as currently specified (shared, not per-team). `L1`'s posterior-averaged prediction is philosophically more correct than plug-in (real parameter and forecast-horizon uncertainty) and never worse, so worth keeping on principle even though it didn't move RPS in this data volume — the two ways of scoring should diverge more once predictions reach further beyond the trained window than this harness's 4-round test windows do.

## History: the earlier stalled run

The sweep initially died mid-run on this machine, twice, for two unrelated reasons, both resolved and now recorded for next time:
1. **A sudden ~90× slowdown** appeared after the first ~7 tasks (idle machine, identical leapfrog-step counts, so not a modelling issue) — never root-caused, resolved by simply re-running (see `--no-cxx` note below, which may have been part of it: `PYTENSOR_FLAGS=cxx=` was set globally in the shell during that attempt, which is more fragile across subprocesses than the per-task `--no-cxx` flag `run_cv.py` now supports).
2. **`CalledProcessError` within 2–9s on every task**: PyTensor's C backend can't link on this machine (`clang++: unknown architecture` against the installed SDK) — JAX does the actual sampling regardless, so this only matters for a few small helper compiles. Fixed by adding `--no-cxx` to `run_cv.py`, which sets `PYTENSOR_FLAGS=cxx=` per subprocess.

## Reproducing

```bash
# (behind TLS interception, first: export REQUESTS_CA_BUNDLE=/path/to/ca.pem)
python work_products/wp007_structural_fixes/run_cv.py --workers 4 --no-cxx   # resumable: skips finished (arm, window) tasks
```

Analysis lives in `analysis.py` (bootstrap CIs, RPS, calibration, odds helpers — all unit-tested in `tests/test_wp007_analysis.py`) and `wp007_structural_fixes.ipynb`, which already holds the executed readout this README's numbers were pulled from. To reproduce it, re-run the notebook's cells directly (`jupyter nbconvert --to notebook --execute wp007_structural_fixes.ipynb`) against a fresh `run_cv.py` sweep.

Took ~4.5 hours wall-clock for the full 270 tasks with 4 workers on this machine (mean ~280s/window for the non-xG arms, up to ~450s/window for `S1_xgweight1`). `--no-cxx` sets `PYTENSOR_FLAGS=cxx=` per subprocess (needed if tasks die in seconds with a clang "linker command failed" error — JAX does the actual sampling regardless). `--arms` and `--max-windows` allow a cheaper subset run. `run_cv.py` pins XLA to one single-threaded device per chain to avoid oversubscription when tasks run concurrently.

## Caveats and things not done

* **Pinnacle benchmark not obtained** — see above. Nothing here should be described as "closes the gap to the market" until that's run.
* **`L5_xgchannel`'s effect is not yet season-robust** — see "Stability across seasons" above. More windows/seasons, or a second held-out re-run, would be the natural next check before treating it as settled the way WP001's RPS result was.
* Every ladder rung besides `L5` came back null at this sample size, not proven harmful — a bigger sweep (more windows, or a wider `promoted_effect_sd`/per-team promotion effect) could still turn one up. I wouldn't prioritize re-testing them without a new reason to expect a different result.
* The 0.12 "real year-on-year change" diagnostic estimate that motivated `L3_randomwalk` is a rough, noise-subtracted estimate from 205 team-season pairs.
* Untested ideas from the original list: a market-informed pre-season prior, rest-days/congestion features, per-team sigma on top of a random walk, a per-team (not shared) promotion effect, wiring any of this into `services/predictor` (which still trains the plain config and samples scorelines independently).
* The CV harness's naive baseline (`ll_naive`) uses the home-goals mean for both sides — inherited from WP001 and left unchanged.
* Tests: 90 pass in the repo suite plus 9 for the analysis helpers (run with `PYTENSOR_FLAGS="cxx="` on the machine I used, where the PyTensor C backend couldn't link even on the untouched code — the NUTS smoke tests failed identically before my changes).
