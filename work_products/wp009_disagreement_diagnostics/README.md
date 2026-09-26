# WP009 — What's actually in the disagreement gap, and can it be sharpened away?

**Status: complete.** WP008 found the model-vs-Pinnacle gap roughly quadruples on the matches where the two disagree most, unchanged by anything WP007 tried. This work product asks *what characterizes those matches* before proposing another structural change — and tests the one fix the diagnosis directly motivates. **Answer: the model is genuinely less sharp than the market, but that under-confidence turns out to be honest, not fixable by post-hoc sharpening — it cancels out exactly (helps on non-upsets, hurts on upsets, net zero). The gap looks like real missing information, not a statistical artifact.**

No new model training in this work product — everything here is a diagnostic pass over WP007/WP008's existing predictions plus one cheap post-hoc test (no retraining needed).

## Setup

All numbers: `L5_xgchannel` (WP007's one validated change) on the 1,205 WP007 test matches with Pinnacle closing odds (same set WP008 used — every number below was cross-checked to reproduce WP008's pooled gap of +0.0055 exactly, confirming the joins are correct).

**A real bug found and fixed along the way**: an early version of this diagnostic's rest-days join produced 2,285 rows instead of 1,205 — a team can legitimately play twice under the same round *number* (the ISO-week-based round scheme groups a rearranged midweek fixture into the same slot as the next round — e.g. Arsenal played both Southampton and Stoke under "round 14" in Dec 2014), so `(team, round)` silently wasn't a unique join key. Fixed by joining on `(team, date)` instead, with a hard uniqueness assertion; `tests/test_wp009_diagnostics.py` has a regression test using that exact Arsenal fixture pair.

## Findings

### 1. The headline: the model beats Pinnacle on upsets, loses badly when the favourite wins

| | n | mean(RPS<sub>model</sub> − RPS<sub>Pinnacle</sub>) | CI (cluster) |
|---|---|---|---|
| Upset (favourite lost) | 540 | **−0.0098** | **[−0.0149, −0.0048] — model better** |
| No upset (favourite won) | 665 | **+0.0179** | **[+0.0143, +0.0216] — model worse** |

This is a far bigger swing than the pooled +0.0055 gap suggests — the pooled number is these two effects netting out, weighted by how often each happens (55% no-upset, 45% upset in this sample). The model isn't uniformly worse than the market; it's specifically worse on the (larger) share of matches that go as expected.

### 2. Why: the model is measurably less sharp (less confident) than Pinnacle

| | mean max-probability (how confident the top pick is) |
|---|---|
| Model | 0.5344 |
| Pinnacle | 0.5430 |

A modest but real ~0.9-point gap. Broken down by what actually happened:

| | mean(model's P on Pinnacle's favourite − Pinnacle's own P) |
|---|---|
| On upsets | −0.0133 |
| On non-upsets | **−0.0235** |

The model is under-confident relative to the market **in both cases**, but nearly twice as under-confident specifically on the matches that go on to vindicate the favourite. Pinnacle's extra sharpness isn't noise — it's disproportionately concentrated on the calls that pay off.

### 3. A specific, distinct pattern: the model under-rates away favourites

| Pinnacle's favourite | n | mean(model_worse) | CI (cluster) |
|---|---|---|---|
| Home | 722 | +0.0028 | [−0.0011, +0.0067] — no clear effect |
| Away | 481 | **+0.0095** | **[+0.0045, +0.0146] — clearly worse** |

On the 481 matches where Pinnacle favours the away side, the model assigns it less win probability than Pinnacle does (0.479 vs. 0.511) and gives the home side correspondingly more (0.280 vs. 0.240) — a consistent ~3–4 point shift toward home specifically when the away side is the stronger team. This is a real, distinguishable pattern, not just "away favourites are rarer" (Pinnacle's own average favourite strength is similar in both splits: 0.565 home-favourite vs. 0.511 away-favourite). It's flagged here as an open lead, not fixed — see "What this doesn't answer."

### 4. Ruled out: rest days, promoted-team involvement, season timing

| Candidate | Result |
|---|---|
| Rest-day differential (days since each side's last league match) | corr(&#124;rest_diff&#124;, model_worse) = **0.007** — no relationship |
| Either side newly promoted | +0.0048 [−0.0008, +0.0107] vs. +0.0058 [+0.0020, +0.0095] for non-promoted — no meaningful split |
| Season round (early vs. late) | mixed, mostly CI-includes-zero; weak, inconsistent — not a real driver |

Rest days/fixture congestion was flagged in WP005's original suggestion list as an untested idea; it's now been tested and doesn't explain any of this. Worth not chasing further without a different rationale.

### 5. The dose-response is real and monotonic

| Disagreement with Pinnacle | n | mean(model_worse) | CI (cluster) |
|---|---|---|---|
| 0.00–0.05 | 638 | +0.0002 | [−0.0016, +0.0020] — flat |
| 0.05–0.10 | 377 | +0.0073 | [+0.0017, +0.0129] |
| 0.10–0.15 | 148 | +0.0090 | [−0.0056, +0.0239] |
| 0.15–0.20 | 31 | +0.0374 | [−0.0070, +0.0788] — wide, small n |
| 0.20+ | 11 | +0.1112 | [+0.0233, +0.1910] — wide, small n |

Confirms WP003/WP008's quantile finding wasn't a threshold artifact — it's a genuine, roughly monotonic relationship between how much the model disagrees with the market and how much worse it does, essentially flat at low disagreement.

## The test this motivates: post-hoc temperature scaling on `L5_xgchannel`

WP004 tested global temperature scaling on the old (pre-WP007) config and found it did nothing for RPS. Given finding #2 above — the model is *measurably* less sharp than the market — it's worth re-testing on the improved model specifically, and worth checking the upset/no-upset split rather than just the pooled number, since #1 shows those two subsets pull in opposite directions.

**Method**: fit `T` (softmax(log p)/T) by log-loss on one half of CV windows, evaluate the RPS change on the other half (2-fold, window-level split — never fits and evaluates on the same window's shared model output). `diagnostics.evaluate_temperature_scaling`, tested in `tests/test_wp009_diagnostics.py`.

| | n | ΔRPS (sharpened − original) | CI (cluster) |
|---|---|---|---|
| **Overall** | 1205 | **+0.0000** | **[−0.0003, +0.0003] — no effect** |
| No upset | 665 | −0.0025 | [−0.0029, −0.0021] — sharpening helps |
| Upset | 540 | +0.0031 | [+0.0024, +0.0038] — sharpening hurts |

Fitted `T` ≈ 0.96–0.98 (mild sharpening) across two different random fold splits — consistent, not sensitive to the particular split.

**This is a clean, informative null result, not an inconclusive one.** Sharpening helps exactly where you'd expect (matches that go as expected) and hurts exactly where you'd expect (upsets), in amounts that cancel almost precisely. That's the signature of genuinely well-calibrated uncertainty, not fixable miscalibration — the model isn't being falsely cautious; it actually doesn't have the information to be more confident, and forcing confidence onto it just moves losses from one bucket to the other. **This replicates WP004's original conclusion, now confirmed on the improved model too**: post-hoc probability scaling is not a lever here, in either direction.

## Bottom line

The remaining gap to Pinnacle isn't a statistical artifact of the model's calibration — three different tests (WP004's original recalibration, this WP's re-test on the improved model, and the upset/no-upset decomposition of that re-test) all point the same way. It looks like the market has real information the model doesn't: something that lets Pinnacle be more confident specifically when the favourite is about to win, and specifically more confident about away favourites. That's consistent with what the repo's own README has flagged as untested from the start — team news the model has no access to (injuries, suspensions, lineups, a manager set to rotate for a cup replay) — not something reachable by tuning `ModelConfig` further.

## Update: WP010 partly walks this back

The conclusion above ("the gap looks like genuine missing information," speculating injuries/lineups) was refined by **WP010**, which benchmarked against Pinnacle's pre-closing line instead of closing. Only **~29% of the gap to closing is late-breaking information** (the confirmed XI / late fitness tests — and upper-bounded, since that window also contains sharp money). The remaining ~71% is a gap at equal *timing* but **not** equal *information*: the opening line already prices squad state, transfer business and manager changes, none of which this model sees at any point. So *historic* lineup data targets that 71%, not the 29% — this does **not** bound the lineup direction. See `work_products/wp010_opening_vs_closing/`.

## What this doesn't answer

* **What the away-favourite pattern actually traces to** — is it home advantage being over-applied for specific teams, a genuine strength-assessment gap for certain away sides, or something else? Flagged, not diagnosed further here.
* **Whether external data (injuries/lineups) would actually close the gap** — plausible given the process of elimination above, but not tested; Understat doesn't carry this data, so testing it would need a new source.
* **A conditional (not global) temperature**, e.g. fit separately for home- vs. away-favourite matches, or informed by the model's own disagreement with a simple market proxy — not attempted; with only 1,205 matches, a finer-grained fit risks overfitting the calibration set the way WP004's isotonic-regression arm did.
* **Whether this pattern holds on `L0_baseline`** (the pre-xG-channel config) too, or is specific to `L5` — not checked; the diagnostic was run only on the model actually worth improving further.

## Reproducing

```bash
python -c "
import sys; sys.path.insert(0, 'work_products/wp009_disagreement_diagnostics')
from diagnostics import build_diagnostic_frame, evaluate_temperature_scaling
j = build_diagnostic_frame('L5_xgchannel')
print(len(j), (j.rps_model - j.rps_pinnacle).mean())  # should print 1205, 0.0055...
"
```

Needs WP007's `cv_shared_data.pkl`/`checkpoints/` and WP008's `odds_raw.pkl` to already exist — this reads their outputs, doesn't retrain or refetch anything. Runs in seconds.
