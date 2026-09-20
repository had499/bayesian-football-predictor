# WP013 — Lineup continuity: does an unusual starting XI predict what the model misses?

**Status: complete. Stage 1 (residual check) missed its bar; stage 2 (continuity wired into the model, full 35-window walk-forward CV) also missed its pre-declared bar, narrowly, with a consistent direction. Result: a lead worth confirming on new matches, not a finding, and nowhere near closing the gap to Pinnacle. See "Stage 2 results". **Update: WP014 tested it on 1,086 fresh held-out matches and did not confirm it (−0.00025 RPS, CI [−0.00097, +0.00046]); see `wp014_continuity_confirmation`.**

## Idea

The model knows team strength but not who is playing. WP008/009 added an attacking-quality deviation (xG+xA per 90) and got no measurable held-out gain. This WP tries a cheaper and different signal: **how usual today's starting XI is for that team**, by unit (defence, midfield, attack, whole XI). It estimates no player quality, so it cannot just restate team strength.

## Feature

`src/football_model/features/continuity_features.py::build_continuity_table`. For each team-match:

- A player's **start share** = fraction of the team's previous 10 league matches he started, in any position (formation changes move a player between slots, not his regularity).
- A unit's **continuity** = mean start share of today's starters in that unit. Units by Understat slot: defence = `GK, DC, DL, DR`; midfield = `DMC/DML/DMR, MC/ML/MR, AMC/AML/AMR`; attack = `FW, FWL, FWR`; plus the whole XI.
- Uses only the team's strictly earlier matches and the confirmed starting XI (never substitutes). Rows are NaN when there are fewer than 5 earlier matches, in a team's first 5 matches of a season (summer transfers make a low share a squad change, not a disruption), and after a gap of more than 150 days (history reset).
- `tests/test_continuity_features.py`, 9 tests, including a truncation test (a later match never changes an earlier row), a check that formation changes don't create false "unusual" players, and season-start and gap handling. I confirmed the tests fail when a leak is deliberately introduced.

Data: the cached WP009 player logs (`player_match_data.pkl`, 66,664 rows, 2020–2026). There is exactly one starting `GK` per team-match, and 4,528 of 4,560 team-matches have exactly 11 starters (16 have 12 and 16 have 10; left as is).

## The check (no retraining)

For each held-out match and each side, the residual is `goals − model λ`, using the WP001 baseline's predictions (401 matches). Regress that residual on each continuity score (standardised, so slopes are goals per standard deviation), with a bootstrap clustered by match. Continuity for the side's own team and for its opponent are both tested.

## Pre-declared primary test

**P1: the opponent's defence continuity vs the side's residual goals.** Hypothesis: when the opponent's back line is unusual, the side scores more than the model expected, so the slope is **negative**. A "hit" requires the 95% CI entirely below zero **and** a negative slope in both halves of the data (split at the median date).

All other slopes (own midfield, own attack, own whole-XI, and the rest) are exploratory and are not findings. A secondary that looks good is a lead to test on new data, not a result.

## What each outcome would mean

- **P1 hits.** Build defence continuity into the model as a covariate (mirrored in `predict.py`, cross-checked bit-for-bit as WP008 did), screen 18 windows, then confirm on all 35, tested against `lineup_loose_combo` on the same matches.
- **P1 misses.** Stop. Do not wire any unit into the model on the strength of a secondary.

## Results

Continuity is available for 87% of team-matches (the rest are NaN by design); on the 401 held-out matches, 702 of 802 side-rows have a valid opponent-defence value. Mean continuity is 0.71 (defence), 0.66 (midfield), 0.64 (attack), 0.68 (whole XI).

**P1: no hit.** Opponent's defence continuity vs the side's residual goals: slope **−0.077 goals per SD, 95% CI [−0.168, +0.013]**, halves −0.069 / −0.083. The direction is the one hypothesised and both halves agree, but the CI includes zero (roughly p ≈ 0.09 two-sided), so it does not meet the declared bar.

| feature | n rows | slope (goals/SD) | 95% CI | half 1 / half 2 |
|---|---|---|---|---|
| opp defence | 702 | −0.077 | [−0.168, +0.013] | −0.069 / −0.083 |
| opp midfield | 702 | −0.003 | [−0.087, +0.087] | +0.004 / −0.011 |
| opp attack | 702 | −0.061 | [−0.148, +0.021] | −0.063 / −0.060 |
| opp whole XI | 702 | −0.079 | [−0.168, +0.009] | −0.066 / −0.092 |
| own defence | 702 | −0.016 | [−0.104, +0.076] | +0.004 / −0.033 |
| own midfield | 702 | +0.055 | [−0.032, +0.142] | +0.039 / +0.072 |
| own attack | 702 | −0.012 | [−0.101, +0.075] | −0.034 / +0.007 |
| own whole XI | 702 | +0.019 | [−0.068, +0.106] | +0.017 / +0.021 |

Exploratory reading, not findings:
- The opponent's *attack* continuity (−0.061) looks almost as strong as its defence continuity, though it has no mechanism for affecting the other side's goals. That suggests whatever signal exists is "the opponent's XI is unusual" in general, not specifically the back line. The features are correlated, so this data cannot separate them.
- Own-side features show nothing; the own-midfield slope (+0.055) has the expected sign but is well inside its CI.
- On `lineup_loose_combo`'s residuals (which already include the attacking-lineup term) the opponent-defence slope is −0.074, CI [−0.165, +0.015]: the attacking covariate does not absorb it.

**Decision (by the rule above): stop, do not wire continuity into the model.** The result is a weak lead in the hypothesised direction, at a size (0.08 goals per SD, about 5% of a typical scoring rate) that this data set cannot confirm or rule out; its own detection limit is about 0.09.

**If it is worth pursuing, the only clean route is more data, not more tests on these 401 matches.** The held-out set is small because each walk-forward window tests only 2 rounds. Re-running the WP001 baseline with longer test spans (same 35 fits, one model) would add several hundred new held-out matches; P1 should then be re-declared and tested on those new matches only, so the 401 that produced this lead are not reused. Cost is about the same as one WP001 rerun, and the feature and notebook here would be reused as they are.

## Stage 2 — continuity inside the model (added after stage 1)

Stage 1's rule said to stop; this stage was added at your request, so treat it as a test of a weak lead rather than a follow-up to a hit. Its own caveat cannot be removed: the idea came from these same 401 held-out matches, so a hit here is not an independent confirmation.

**What was wired in (same pattern as WP008's lineup covariate):**
- `ModelConfig.use_continuity` (default False, so every earlier config is unchanged) and `continuity_beta_sd` (0.1).
- `ModelData.defence_cont_home/away`, filled by `prepare_model_data(continuity_table=...)`, joined on `(team, date)`, defaulting to 0 (neutral) when missing.
- `build_model`: `theta_home += beta_continuity * defence_cont_away` and `theta_away += beta_continuity * defence_cont_home`, that is, a side's back-line continuity affects its **opponent's** scoring. `beta_continuity ~ Normal(0, 0.1)`: the hypothesis has a direction (negative) but the data decides the sign.
- `predict.py`: the same term in `compute_theta`/`predict_match_lambdas`/`predict_rows`. Two new tests compare the numpy prediction against the real PyMC model's lambdas, with deliberately asymmetric home/away continuity so a swapped side would fail.
- `run_cv_window.py`: extracts `beta_continuity`, applies it at prediction, records it per window. It **raises** if `use_continuity` is requested but the shared-data pickle has no `continuity_table`, since the feature would otherwise be all zeros and the arm would silently be the baseline.
- `build_multileague_model` raises `NotImplementedError` for `use_continuity`.
- The feature is standardised with fixed constants (defence continuity mean 0.714, SD 0.140, over 2020–2026): two scalars of look-ahead, immaterial for a covariate with SD 0.14.

**Verification:** 13 new tests (model wiring, the two prediction cross-checks, the data join, the standardiser, the guard). One real end-to-end window (window 25, full sampling, through `run_cv_window.py`) completed cleanly with `beta_continuity = −0.053`: negative as hypothesised, and about the size stage 1's slope implies (−0.077 goals per SD is roughly −0.055 in log rate). A single window says nothing about held-out accuracy.

**Arms** (`wp013_continuity_cv.ipynb`, run by you; 18-window screen, then 35 only if the screen shows something):
- `continuity`: WP001 defaults plus `use_continuity`.
- `continuity_lineup_loose_combo`: WP009's `lineup_loose_combo` plus `use_continuity`.
- seeded, not re-run: `baseline` (WP001) and `lineup_loose_combo` (WP009 full 35-window checkpoint).

**Pre-declared primary comparison:** paired RPS, `continuity − baseline`, 95% bootstrap CI entirely below zero. Secondary: `continuity_lineup_loose_combo − lineup_loose_combo`. Reported alongside: `beta_continuity` per window. The detection limit for a paired RPS difference on about 361 matches is roughly ±0.0007–0.001 (WP011's CIs), so a smaller gain cannot be told apart from noise and a miss means "not large".

**Cost:** about 35 s per window on an idle machine (the single smoke window took 2:48 on a loaded one); 2 arms × 18 windows for the screen at `MAX_WORKERS=3`.

## Stage 2 results

Full 35-window CV, 361 Pinnacle-covered matches (401 held-out), 67.5 min wall time; all four arms have 35/35 windows.

| arm | RPS | gap to Pinnacle | 95% CI | share of baseline's gap closed |
|---|---|---|---|---|
| baseline | 0.1925 | +0.0139 | [+0.0080, +0.0198] | 0% |
| lineup_loose_combo | 0.1911 | +0.0125 | [+0.0067, +0.0185] | 10% |
| **continuity** | 0.1915 | +0.0129 | [+0.0072, +0.0187] | 7.5% |
| continuity_lineup_loose_combo | 0.1902 | +0.0116 | [+0.0059, +0.0175] | 17% |

Pinnacle RPS is 0.1786. The gap CIs overlap heavily; the paired differences are the test.

| paired RPS difference (negative = first is better) | mean | 95% CI | verdict |
|---|---|---|---|
| **PRIMARY: continuity − baseline** | −0.00104 | [−0.00221, +0.00013] | **no hit** (upper bound +0.00013) |
| secondary: continuity_lineup_loose_combo − lineup_loose_combo | −0.00093 | [−0.00204, +0.00020] | no hit (upper bound +0.00020) |
| context: lineup_loose_combo − baseline | −0.00139 | [−0.00275, −0.00001] | WP009's known thin edge |

**Verdict by the declared rule: no hit on the primary or the secondary.** Both are narrow misses in the hypothesised direction.

What points towards a real, small effect:
- Both paired differences are about −0.001 RPS, roughly three-quarters of what WP009's lineup covariate gave, and the primary and secondary agree.
- Both halves of the data agree (primary −0.00078 / −0.00129; secondary −0.00066 / −0.00118); continuity improved 59% of matches.
- `beta_continuity` is negative in 32 of 35 windows (mean −0.033, range −0.053 to +0.025), as hypothesised. It also gets stronger as training data grows: mean −0.012 over the first 12 windows and −0.046 over the last 12, heading towards stage 1's implied −0.055. That is what a real coefficient shrunk by a Normal(0, 0.1) prior looks like when data accumulates.

What points the other way, or limits the claim:
- **Selection.** The idea came from these same 401 matches (stage 1). Ideas found on a data set look better on it than they will on new data, so the true effect is probably smaller than −0.001.
- **The windows are not independent.** Each window's training set contains the earlier ones, so "32 of 35" and the growth in `beta_continuity` are one accumulating data set seen 35 times, not 35 confirmations. A standard error computed across windows would be wrong.
- **The effect sits at the detection limit** (about ±0.001 RPS on 361 matches), so this run cannot tell "small and real" from "noise".
- By calendar year the primary difference is negative in 2021–2024 and +0.0006 in 2025.
- The best arm against the baseline (`continuity_lineup_loose_combo − baseline` = −0.00232, CI [−0.00409, −0.00058]) excludes zero, but it was not pre-declared, it picks the best of four arms after seeing the results, and about 60% of it is WP009's known lineup effect. It is not a hit.

**What this means for the goal (beating the bookmakers):** nothing changes. The best arm is still +0.0116 RPS behind Pinnacle's close (83% of the baseline's gap remains), and WP012 found the model adds no information to that price at any weight. Starting XIs are also public about an hour before kickoff, so the market has the chance to price this information before kickoff (not measured here).

**To confirm or drop it:** the only clean route is new matches. Re-run the WP001 baseline and the two continuity arms with longer test spans (same 35 fits) to add several hundred held-out matches, re-declare the primary, and test on the new matches only. Until then it is a lead, not a result.

## Limits

- 401 matches gives about 700 side-rows after NaN removal, so only effects of roughly 0.09 goals per SD or more are detectable (standard error 0.046). A miss means "not large", not "zero".
- Starting XIs are published about an hour before kickoff, so any use of this feature for betting is limited to that window.
- Continuity cannot separate injury from rotation, and does not measure the quality of the replacement.

## Reproducing

```bash
pytest tests/test_continuity_features.py
cd work_products/wp013_lineup_continuity
jupyter lab wp013_lineup_continuity.ipynb   # stage 1: no sampling, about a minute
jupyter lab wp013_continuity_cv.ipynb        # stage 2: walk-forward CV, heavy compute
```
