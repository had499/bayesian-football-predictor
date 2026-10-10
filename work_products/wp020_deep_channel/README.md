# WP020 — Deep completions as a measurement channel

**Status: complete. Verdict: the best result since the xG channel, but NOT established.** Adding a deep-completion channel on top of the xG channel is worth **−0.00096 RPS [−0.00248, +0.00063]**. The point estimate is the largest improvement found since WP007, and it narrows the gap to Pinnacle's pre-close line from **+0.00409 to +0.00289** — about 29% of what remained. But the CI includes zero, and the effect (−0.00096) is smaller than this design's detection threshold (±0.00156). **This is an under-powered positive, not a win.**

The shipping config stays `L0_baseline` + `use_xg_channel=True` until this is confirmed at higher power.

## Why deep completions, after WP019 closed the shot feed

WP019 established that `(shot count, total xG)` nearly spans the shot feed: every decomposition tried — big/small chances, open/dead ball, on-target xG — correlated ≥0.93 with one of the two, because they are all sums over the same 116,448 shots. A genuinely new channel had to **not be a function of the shot list**.

Understat's `teamsData` carries two such fields. `deep` (passes completed within ~20m of the opposing goal, excluding crosses) measures possession worked into dangerous areas that **did not produce a shot** — structurally invisible to a model whose only observables are goals and shot-derived xG. `ppda` was tested and rejected at the premise stage.

## The prediction, recorded before running

Written into `run_cv.py` and scored automatically by `analyse.py` so it could not be quietly revised:

1. **`H_deep_added` beats the baseline by more than 0.0005 RPS**, point estimate −0.0015
2. **The gain concentrates in the high-line-movement quartile** where WP010 localised the deficit

| | predicted | actual | |
|---|---|---|---|
| (1) size | > 0.0005, pt −0.0015 | **−0.00096** [−0.00248, +0.00063] | **direction and rough size right; CI includes zero** |
| (2) location | top 25% line movement | **middle 50%** (−0.00227 [−0.00436, −0.00002]); top quartile −0.00051 | **WRONG** |

Prediction (2) failing matters: **deep completions do not fix WP010's specific deficit.** The gain is a general improvement spread across ordinary matches, and the high-line-movement concentration remains unexplained — as it has since WP010.

The forecast came from a per-match premise check (n=7,680, leave-one-season-out) that put `deep` at 117% of the xG effect and extrapolated −0.0030 [−0.0062, −0.0006]. I shaded that to −0.0015 for a team-identification confound. The true value sits below even the shaded estimate, so **the framework is directionally useful and quantitatively optimistic** — it over-predicted here and slightly over-predicted WP019's shots too (−0.00044 predicted, +0.00016 measured).

## Results

30 windows, 1,314 paired matches, 0 divergences, max rhat 1.065.

| Arm | Channels | Pooled RPS | Gap vs Pinnacle pre-close |
|---|---|---|---|
| `L0_baseline` | goals | 0.20739 | +0.00650 |
| `S3_xgchannel_only` | goals+xG | 0.20481 | +0.00409 |
| **`H_deep_added`** | goals+xG+deep (w=0.5) | **0.20385** | **+0.00289** |
| `I_deep_only` | goals+deep | 0.20576 | +0.00456 |
| `J_deep_heavy` | goals+xG+deep (w=1.0) | 0.20408 | +0.00304 |

| Comparison | ΔRPS | CI (cluster) |
|---|---|---|
| deep ADDED to xG (H vs A) | **−0.00096** | [−0.00248, +0.00063] |
| deep INSTEAD of xG (I vs A) | +0.00095 | [−0.00146, +0.00339] |
| is deep worth anything at all (I vs L0) | −0.00164 | [−0.00362, +0.00046] |
| heavier deep weight (J vs H) | +0.00022 | [−0.00069, +0.00117] |
| *[reference] the xG channel itself (A vs L0)* | *−0.00258* | *[−0.00414, −0.00095]* |

## Three things that distinguish this from WP019's null

**1. The weight ladder has the right shape.** WP019's shot channel degraded monotonically as it was trusted more. This one improves and then flattens — the signature of a channel with a genuine optimum:

| weight | RPS | |
|---|---|---|
| 0.0 (baseline) | 0.20481 | |
| **0.5** | **0.20385** | best |
| 1.0 | 0.20408 | slight regression |

**2. Better in 5/6 seasons** (2023 alone: −0.00385 [−0.00607, −0.00161], CI excluding zero). WP019's shot channel managed 3/6. The xG channel's bar was 6/6.

**3. Deep alone nearly beats the raw baseline** — `I_deep_only` vs `L0` is −0.00164 [−0.00362, +0.00046], against −0.00091 for shots alone in WP019. It is a stronger stand-alone channel, though still weaker than xG (−0.00258).

## The channel calibrated itself correctly

| | learned | truth in the data |
|---|---|---|
| `deep_kappa` (deep completions per unit rate) | **4.89** | **5.05** |

As with `shot_kappa` (9.33 vs 9.39), this is not a misspecified channel. `sigma_att` rises modestly (0.0170 → 0.0173) at w=0.5 and more at w=1.0 (0.0239, with `rho_att` 0.9928 → 0.9838), consistent with `J` being slightly over-flexed — the same over-trusting signature WP019 found at weight 1.0.

## What this needs before it can be believed

The effect is **below this design's MDE**. 30 windows give a detection threshold of ±0.00156 and the effect is −0.00096, so a null was always the likely outcome even if the effect is real. The BACKLOG has long flagged that a proper test needs ~3x the windows (2017 onward at step 4, ~8–11 hours). **This is now the best-motivated long run in the project's history**: a ~37%-of-xG-channel effect, right ladder shape, 5/6 seasons, and a mechanism that is not a relabelling of anything already in the model.

Second priority: a combined `xG + deep + shots` arm. WP019's shot channel was null alone but its per-match information was not *negative*, and `deep`-on-top-of-shots kept 98% of its value in the premise check, so the three may be complementary.

## Reproducing

```bash
python work_products/wp020_deep_channel/prepare_data.py       # joins `deep` via short_to_title.pkl
python work_products/wp020_deep_channel/run_cv.py --workers 4 # 90 tasks, ~142 min
python work_products/wp020_deep_channel/analyse.py
```

Needs WP007's `cv_shared_data.pkl`/`checkpoints/`, WP013's `checkpoints/`, WP008's `odds_raw.pkl`.
