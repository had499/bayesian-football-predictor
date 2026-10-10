# WP014 — Does npxG sharpen the xG channel? Does xG calibration drift matter?

**Status: complete. Verdict: NULL on both, and both nulls are well-powered.** Removing penalties from the xG measurement channel is worth −0.00014 RPS [−0.00050, +0.00025]; fitting the channel's `kappa` per season is worth +0.00004 [−0.00001, +0.00008]. Neither is distinguishable from zero, and the intervals are tight enough to rule out anything worth ~13% of the remaining gap.

**The shipping config is unchanged: `L0_baseline` + `use_xg_channel=True`** (WP013).

## Why these two were worth testing

They were the only remaining ideas that *sharpened the one mechanism that works* rather than adding a new one:

1. **npxG.** Penalties are ~0.76 xG each and poorly repeatable — winning them has weak season-to-season persistence — so in principle they inject noise into the measurement channel while carrying little signal about team strength. npxG is the standard metric in football analytics for exactly this reason.
2. **Calibration drift.** Measured directly: the xG/goals ratio moves near-monotonically from **0.957 (2014–16) to 1.078 (2023–25)**. The model fits one global `kappa`, so it was using a compromise value wrong at both ends of its own training history.

## A data problem that changed the design

Understat's **match-level xG and its shot feed disagree**: correlation 0.997, but the shot-sum averages 1.4481 against the match aggregate's 1.4254, with a tail of ~1.0 discrepancies. Not own goals — only ~32 per season. Most likely the two were computed at different xG-model versions.

That mattered, because comparing shot-derived npxG against the match-level xG the model currently uses would have confounded *penalties removed* with *different data source*. So a third arm was added purely as a control:

| Arm | Channel observes | Isolates |
|---|---|---|
| `S3_xgchannel_only` (WP013) | match-level xG | the shipping baseline |
| **`D_shotxg`** | shot feed, penalties **included** | the source change alone |
| **`B_npxg`** | shot feed, penalties **removed** | penalties, source held constant |
| **`C_kappa_season`** | match-level xG, `kappa` per season | calibration drift alone |

Without `D`, `B`'s number would have been uninterpretable.

## Results

30 windows, 1,314 paired matches. Penalties confirmed at **6.7% of total xG**, present in **12% of matches**.

| Arm | Pooled RPS | Gap vs Pinnacle pre-close |
|---|---|---|
| `L0_baseline` | 0.20739 | +0.00650 |
| `S3_xgchannel_only` | 0.20481 | +0.00409 |
| `D_shotxg` | 0.20477 | +0.00406 |
| **`B_npxg`** | **0.20463** | **+0.00388** |
| `C_kappa_season` | 0.20485 | +0.00414 |

| Comparison | ΔRPS | CI (cluster) | Verdict |
|---|---|---|---|
| penalties removed, source constant (B vs D) | −0.00014 | [−0.00050, +0.00025] | no clear difference |
| does the source matter at all (D vs A) | −0.00004 | [−0.00016, +0.00007] | no clear difference |
| npxG vs shipping config (B vs A) | −0.00018 | [−0.00055, +0.00022] | no clear difference |
| per-season `kappa` (C vs A) | +0.00004 | [−0.00001, +0.00008] | no clear difference |
| *[reference] the xG channel itself (A vs L0)* | *−0.00258* | *[−0.00414, −0.00095]* | *better* |

Leave-one-season-out on `B_npxg`: null in all six folds. Sampler: 0 divergences; max rhat 1.059 — higher than ideal, worth noting.

**The source-change arm is its own small finding**: swapping match-level xG for the shot feed moves RPS by −0.00004 [−0.00016, +0.00007]. The two sources disagree measurably but the disagreement is immaterial to forecasts. Good to know before anyone spends time reconciling them.

## Why npxG didn't help — my prior reasoning was wrong about the size

I argued penalties are ~7% of xG so removing them should improve signal-to-noise by *more* than 7%. That framing was wrong in two ways, and the fitted parameters show it:

1. **`kappa` absorbs the level change entirely.** Learned `kappa`: **0.957 for npxG vs 1.029 for shot-total** — a 7.0% difference, almost exactly the penalty share. `kappa` is a free scale parameter, so removing a near-constant fraction of xG just rescales it. The channel's *information per match* barely moves.
2. **Penalties occur in only 12% of matches.** For the other 88%, npxG is *identical* to xG — the observation is literally unchanged. So the change touches roughly one observation in eight, and within those it is partly absorbed by (1).

"7% of total xG" is a statement about the level. What the channel needs is signal relative to noise, and on 88% of rows nothing changed at all.

## Why per-season kappa didn't help — as predicted

Flagged before running: `kappa` drift is a **common-mode** error that shifts all teams together, and common-mode shifts largely cancel in home/draw/away probabilities. That's the same reason Dixon-Coles does nothing (WP002/WP007 — a uniform draw-ward nudge). The result is +0.00004 with a CI of width 0.00009 — about as exactly zero as anything measured in this project.

The drift is real and fixing it is defensible on correctness grounds. It buys no accuracy.

## Bottom line

Twelve distinct ideas have now been tested across WP004–WP014; **one works** (the xG measurement channel, −0.0026 RPS, robust in 6/6 leave-one-season-out folds per WP013).

The model sits at **+0.0039 RPS behind Pinnacle's pre-close line**, at parity on ~75% of matches and in 3 of 6 seasons. The two ideas here were the last that targeted the working mechanism rather than adding a new one, and both are now closed with tight intervals.

I would stop optimising. The remaining candidates are all either well-powered nulls already, or require data this pipeline doesn't have.

## What's still genuinely open

* **`L3_randomwalk` and xG-weight are under-powered nulls** (MDE 35% and 52% of the gap respectively). Testing them properly needs ~3× the CV windows — 2017 onward at step 4, ~8–11 hours. Best motivated of anything left, but I'd expect null.
* **The promotion offset is likely mis-specified, not just under-powered** — WP007 fitted −0.06 against a real gap of −0.37, so a per-team version is untested.
* **The away-favourite asymmetry (WP009)** and **the deficit's concentration in high-line-movement matches (WP010)** remain unexplained.
* **`param_means` doesn't capture the season-varying kappa names** (`xg_kappa_mu`/`tau`/`z`), so `C_kappa_season` shows NaN in the learned-parameter table. Cosmetic; the arm itself ran correctly.

## Reproducing

```bash
# ~90 min, resumable; set REQUESTS_CA_BUNDLE/SSL_CERT_FILE behind TLS interception
python work_products/wp014_npxg_decomposition/fetch_shots.py
python work_products/wp014_npxg_decomposition/prepare_data.py    # writes both variants
python work_products/wp014_npxg_decomposition/run_cv.py --workers 4   # 90 tasks, ~92 min
python work_products/wp014_npxg_decomposition/analyse.py
```

Needs WP007's `cv_shared_data.pkl`/`checkpoints/`, WP013's `checkpoints/`, and WP008's `odds_raw.pkl`.
