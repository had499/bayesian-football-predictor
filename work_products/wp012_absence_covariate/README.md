# WP012 — Squad absence as a model covariate: the test WP011 couldn't do

**Status: complete. Verdict: NULL, and this time through the channel that matters.** Adding WP011's `absent_drop` as a covariate inside the model changes RPS by **+0.00011, CI [−0.00025, +0.00044]** across 1,314 paired matches — indistinguishable from zero, with a CI tight enough that this is a well-powered null rather than a failure to detect.

This closes the gap WP011 left open. That work product only *correlated* absence against finished predictions, which couldn't see the channel where the feature changes how the latent team strengths are inferred during training. This one puts the feature in the model and measures the forecast.

## What was built

`absent_drop` — the share of a team's recent xG+xA produced by players whose availability has fallen below their own norm (WP011) — now enters each side's log rate as `beta_absence * absence`, plumbed end-to-end:

`ModelData` → `prepare_model_data` → `build_model` → `predict.py` → `run_cv_window.py`

The prediction path matters as much as the model path: a term applied in training but dropped at prediction time is exactly the failure that silently corrupted the xG covariate and Dixon-Coles in earlier work. Two tests cover it, including one asserting the term reaches `lambda` with precisely the value `predict.py` computes.

**Only 30 new fits were needed.** The WP012 data file was verified to produce bit-identical `ModelData` apart from the absence arrays, so WP007's existing `L5_xgchannel` checkpoints are a valid paired baseline (`L5` has `use_absence=False` and ignores those arrays).

## Results

| | Pooled RPS | CI (cluster) |
|---|---|---|
| `L5_xgchannel` | 0.20486 | [0.19511, 0.21442] |
| `L7_absence` | 0.20497 | [0.19511, 0.21447] |

**Paired: +0.00011, CI [−0.00025, +0.00044] — no clear difference.**

Against the fair (pre-close) market benchmark, on the 1,205 matches with Pinnacle opening odds:

| Arm | Gap vs. pre-close |
|---|---|
| `L5_xgchannel` | +0.00392 [+0.00122, +0.00663] |
| `L7_absence` | +0.00419 [+0.00147, +0.00683] |

Unchanged — marginally worse, well within noise. Sampler clean: 0 divergences, max rhat 1.019 across all 30 windows.

## The interesting part: the coefficient comes out backwards

The hypothesis predicts a **negative** `beta_absence` — a side missing its contributors should score less. The model learned **+0.205** (5–95% across windows: +0.080 to +0.354), positive in *every one* of the 30 windows. A 1-SD absence swing implies a **+2.2%** increase in that side's scoring rate.

That is not noise, and it is not what the theory predicts.

It also isn't because the feature is measuring nothing. The *marginal* relationship runs the right way:

| `absent_drop` quartile | mean xG scored |
|---|---|
| low | 1.595 |
| q2 | 1.561 |
| q3 | 1.495 |
| high | 1.479 |

A clean monotonic decline, ~7% across the range, in the predicted direction. So raw absence really does associate with slightly weaker attacking output — the signal exists, it's just small.

**The sign flips once you condition on the model's latents, and I can't confidently explain why.** The most plausible reading is that the attack/defence random walk has *already* absorbed the absence effect through the weaker results it produced, so the covariate is largely redundant and ends up correcting for over-shrinkage rather than adding information. That story is incomplete though: WP011 measured the walk at σ≈0.0143/round, which only moves ~2.9% over a four-match absence — less than the ~7% marginal effect, so the latent shouldn't absorb all of it. `absent_drop` also correlates 0.52 with rotation breadth, so confounding with squad depth is plausible. I'd treat the mechanism as unresolved rather than assert one.

Either way, the forecasting answer doesn't depend on resolving it: the feature does not improve predictions.

## What this settles, and what it doesn't

**Settles**: the specific claim WP011 could not test. With the feature *in* the model, able to reshape the latents during training, it still buys nothing. The correlational null and the in-model null now agree, which is considerably stronger than either alone.

**Doesn't settle**: whether a *better* absence feature would work. The wrong-signed coefficient is itself evidence that `absent_drop` is an imperfect proxy for "this side is weakened" — it likely also picks up rotation, squad depth, and early substitutions in comfortable wins. A cleaner construction (weighting by minutes lost among established starters only, or separating injury from rotation) might behave differently. But two work products have now probed this from opposite directions and both say no, so I would want a specific reason before a third.

**Also unchanged**: the away-favourite asymmetry (WP009) and the concentration of the deficit in high-line-movement matches (WP010) both remain unexplained.

## Where the project stands

Tried and null: recalibration (WP004), prior loosening (WP005), per-team sigma (WP006), random-walk dynamics / intercept / promotion offset / Dixon-Coles (WP007), post-hoc sharpening and rest days (WP009), squad instability correlationally (WP011) and now as a model covariate (WP012).

Tried and positive: the xG measurement channel (WP007), worth −0.0025 RPS and ~32% of the gap to the closing line (WP008).

The model sits at **+0.0039 RPS behind Pinnacle's pre-close line**, at parity on ~75% of matches and in 3 of 6 seasons. Ten distinct ideas have been tested; one worked. That's a reasonable point to stop optimising and consolidate.

## Reproducing

```bash
python work_products/wp012_absence_covariate/prepare_data.py   # joins WP011 features onto df_cv
python work_products/wp012_absence_covariate/run_cv.py --workers 4   # 30 windows, ~42 min
python work_products/wp012_absence_covariate/analyse.py
```

Needs WP011's cached rosters, WP007's `checkpoints/` and `cv_shared_data.pkl`, and WP008's `odds_raw.pkl`.
