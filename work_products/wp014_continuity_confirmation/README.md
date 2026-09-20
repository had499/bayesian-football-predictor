# WP014 — Confirming lineup continuity on fresh matches

**Status: complete. Continuity was NOT confirmed on 1,086 fresh held-out matches: primary `continuity − baseline` = −0.00025 RPS, 95% CI [−0.00097, +0.00046] (no hit). The estimate is about a quarter of WP013's −0.00116, which is what selection from the original 401 matches would predict. See "Results".**

## Why

WP013 found a lead: adding the opponent's defence continuity to the model improved RPS by −0.00104 (95% CI [−0.00221, +0.00013]) on the 401 matches WP001's windows hold out. That missed the pre-declared bar narrowly, and the idea came from those same 401 matches, so it cannot be confirmed on them. This WP tests it on matches that were never held out before.

## Design

- **34 new walk-forward windows**, built by `football_model.evaluation.windows.interleaved_windows`. Each trains on rounds 1..T+2 and tests the next **3 rounds**, where T is the corresponding WP001 window's `train_end` (36, 41, ..., 201; the T = 206 window has no rounds left). Test rounds are 39–41, 44–46, ..., 204–206.
- **Disjoint from every match WP001 held out**: WP001 tests rounds T+1; these test T+3..T+5. The function raises if a round would be reused, and `tests/test_windows.py` covers it. New windows do train on the old test rounds, which is ordinary walk-forward use of past data.
- **1,086 new held-out matches** (2.7× WP001's 401) for 34 fits, the same compute as the "double to 70 windows" plan, because a fit is what costs time and a 3-round test costs almost nothing extra. The alternative (34 windows testing one round each) would add only about 390 matches.
- **Deviation from WP001's protocol, deliberately:** predictions are now 1–3 rounds beyond the last trained round instead of 1 (the model holds attack/defence at their final trained value; see `predict_rows`' `max_t`). That makes every arm's absolute RPS a little worse than WP001's numbers, and applies equally to all four arms, so the paired comparison is unaffected. WP014's absolute RPS must not be compared with WP001–WP013's.
- **All four arms are fitted** on the new windows (nothing to seed): `baseline`, `lineup_loose_combo`, `continuity`, `continuity_lineup_loose_combo`, with the same configs as WP013 (`LOOSE` = WP005's `loose_combo`).
- Smoke test: one real window (10 of 34, `continuity`, full sampling, through `run_cv_window.py`) completed with 32 predictions stored for rounds 84–86 and `beta_continuity = −0.035`. Multi-round alignment between the model's rows and the fixtures was checked (identical goals in identical order).

## Pre-declared tests

The comparison uses **all** new held-out matches, since comparing arms needs no odds; the gap to Pinnacle is reported on the Pinnacle-covered subset for context only.

| | comparison (paired RPS, per match, bootstrap) | a "hit" requires |
|---|---|---|
| **Primary** | `continuity − baseline`, new matches only | 95% CI entirely below zero |
| Secondary | `continuity_lineup_loose_combo − lineup_loose_combo`, new matches only | 95% CI entirely below zero |

There is one primary test. The secondary is only interpreted alongside it. Everything else in the notebook (per-window `beta_continuity`, calendar-year splits, the pooled old+new estimate) is exploratory.

## What each outcome means

- **Primary hits.** Continuity is a small, confirmed improvement. Adopt `use_continuity` in the best config, and only then consider extending it (other units, longer history).
- **Primary misses.** Not confirmed. Report the estimate and CI, read them against the power below, and stop wiring continuity variants into the model; the EPL data has almost no unused matches left to test on (about 390 rounds' worth, the T+2 rounds), so further ideas built from these matches cannot be validated on EPL alone.
- **Secondary** only decides whether to prefer `continuity_lineup_loose_combo` over `lineup_loose_combo` as the best-available configuration.

## Power (calculated before running)

Standard error of the paired difference is about 0.00060 at WP013's 361 matches (from its CI), so about **0.00034** at 1,086 matches (assuming the same per-match spread). The chance the primary hits, if the true effect is:

| true RPS effect | power at 361 (WP013) | power at 1,086 (this WP) |
|---|---|---|
| −0.0005 | 13% | 31% |
| −0.00075 | 24% | 59% |
| −0.0010 | 39% | 83% |
| −0.0014 (WP009's lineup effect) | 64% | 98% |

The smallest effect detected 80% of the time is about 0.00096. **A miss therefore rules out effects of roughly −0.001 or larger, but not a real effect near −0.0005**, and the original −0.00104 is likely inflated by having been picked from a look at the data. This is the best EPL allows: WP014 already uses most of the unused matches.

## Results

All four arms completed 34/34 windows (135.5 min wall time); 1,086 new held-out matches, verified against `df_cv`'s own goals and identical across arms. Numbers below were recomputed from the checkpoints and match the notebook.

| arm | RPS on the 1,086 new matches |
|---|---|
| baseline | 0.2059 |
| lineup_loose_combo | 0.2051 |
| continuity | 0.2056 |
| continuity_lineup_loose_combo | 0.2049 |

(Absolute RPS is higher than in WP001–013 because these are different matches and are predicted up to 3 rounds ahead; compare arms with each other only.)

| paired RPS difference (negative = first is better) | mean | 95% CI | better on | verdict |
|---|---|---|---|---|
| **PRIMARY: continuity − baseline** | −0.00025 | [−0.00097, +0.00046] | 49% of matches | **no hit** |
| secondary: continuity_lineup_loose_combo − lineup_loose_combo | −0.00026 | [−0.00095, +0.00042] | 49% | no hit |
| context: lineup_loose_combo − baseline | −0.00073 | [−0.00151, +0.00006] | 55% | not a pre-declared test |

Standard error of the primary is 0.00037 (the pre-run estimate was 0.00034), so the smallest effect this test detects 80% of the time is about 0.0010.

**Reading, against the rule set before running:**
- **Continuity is not confirmed.** By the declared rule that means stop wiring continuity variants into the model, and `lineup_loose_combo` stays the best-available configuration; `use_continuity` stays in the code, off by default.
- **The point estimate is negative but small: −0.00025, about a quarter of the −0.00116 seen on the original 401 matches.** That is the shrinkage you expect when an effect is picked from a look at the data. It is not evidence of zero: the CI runs from −0.00097 to +0.00046, so it **rules out effects of about −0.001 or larger, but not a real effect near −0.0005**, exactly as the power table said it would.
- The same shrinkage shows up for WP009's `lineup_loose_combo`, which was the best of four arms when it was chosen: its edge over baseline halved from −0.00139 to −0.00073 and its CI now includes zero. Expect any improvement of this size found on this data to look smaller on new matches.

**Exploratory, not findings:**
- `beta_continuity` is negative in 31 of 34 windows (mean −0.034), again from nested windows, so not 34 independent confirmations. The model learns the relationship in training and it does not turn into a reliable held-out gain, the same pattern as WP009's lineup covariate.
- The primary difference by calendar year is mixed (2021 +0.0015, 2022 −0.0002, 2023 −0.0016, 2024 −0.0013, 2025 +0.0002, 2026 +0.0024) and by rounds ahead is −0.00055, −0.00048, +0.0003 (about 350 matches each). Consistent with noise.
- **Pooled old + new (1,487 matches, not independent of the original selection):** −0.00050, CI [−0.00110, +0.00012].

**Gap to Pinnacle on the new matches (998 with closing odds; Pinnacle RPS 0.1982):** baseline +0.0074 [+0.0035, +0.0113], best arm (`continuity_lineup_loose_combo`) +0.0063 [+0.0025, +0.0100]. The model is still significantly behind the market. But the baseline's gap here is about half of WP003's +0.0139 on the original 361 matches; the two estimates differ by roughly 1.8 standard errors, so that is not conclusive, and the matches and prediction horizon differ. A reasonable reading is that +0.0139 was on the high side, and the gap over all 1,359 Pinnacle-covered matches is nearer +0.009 (a weighted average of the two, not a separate test).

**Consequences:**
1. Stop building continuity variants. EPL has almost no unused matches left (about 390 rounds' worth, the T+2 rounds), so further ideas cannot be validated on EPL alone.
2. The 998 fresh Pinnacle-covered matches are the first opportunity to re-test WP012's conclusions (the model adds nothing to Pinnacle's price) on data WP012 never saw. Worth doing before treating WP012 as final.
3. For beating the bookmakers, nothing here changes the picture.

## What this cannot show

- **Anything about beating the bookmakers.** WP012 found the model adds nothing to Pinnacle's price, and even the best arm was +0.0116 RPS behind it. A confirmed 0.001 gain would improve the model, not close that gap.
- Whether the effect would help a bet: starting XIs are public about an hour before kickoff, and the market's reaction was not measured.

## Cost

Four arms × 34 windows = 136 fits. WP013 took 67.5 min for 70 fits at `MAX_WORKERS=3` on the laptop, so expect about 2 hours there, less on a desktop that does not throttle. The checkpoints resume, so it can be run in batches.

## Reproducing

```bash
pytest tests/test_windows.py
cd work_products/wp014_continuity_confirmation
jupyter lab wp014_continuity_confirmation.ipynb   # heavy compute
```

`cv_shared_data.pkl` is WP013's shared data (EPL data, `lineup_dev_table`, `continuity_table`) with `windows` replaced by the 34 new ones.
