# WP021 — High-power confirmation run for the deep-completion channel

**Status: complete. Verdict: NOT CONFIRMED.** At 115 windows the deep channel is worth **−0.00083 RPS [−0.00177, +0.00013]** — still null, and more importantly **not robust**: dropping a single season (2016) halves it to −0.00044 [−0.00140, +0.00047]. Adding the shot channel on top **cancels it entirely** (`K_deep_shots` = −0.00000).

**The shipping config remains `L0_baseline` + `use_xg_channel=True`.**

## Why this run existed

WP020 measured −0.00096 [−0.00248, +0.00063] on 30 windows, where the detection threshold was ±0.00156 — so a null was guaranteed whether or not the effect was real. This grid was sized to settle it:

| windows | MDE | CI the WP020 effect would produce | |
|---|---|---|---|
| 30 (WP020) | 0.00156 | [−0.00252, +0.00060] | still null |
| 77 (BACKLOG's plan) | 0.00097 | [−0.00193, +0.00001] | borderline |
| **115 (this run)** | predicted 0.00080 | [−0.00176, −0.00016] | should clear zero |

115 windows = first test season 2016, STEP 3, TEST_LEN 3. `STEP == TEST_LEN` keeps test sets contiguous and **non-overlapping**, which the window-cluster bootstrap requires — it treats windows as independent clusters. 3,780 test matches, 2.9x WP020's. 345 fits, 6.6 h, 0 failures, 0 divergences, max rhat 1.055.

## Results

| Arm | Channels | Pooled RPS | Gap vs Pinnacle pre-close |
|---|---|---|---|
| `S3_xgchannel_only` | goals+xG | 0.19819 | +0.00444 |
| **`H_deep_added`** | goals+xG+deep | **0.19736** | **+0.00359** |
| `K_deep_shots` | goals+xG+deep+shots | 0.19818 | +0.00465 |

| Comparison | ΔRPS | CI (cluster) |
|---|---|---|
| deep ADDED to xG (H vs A) | −0.00083 | [−0.00177, +0.00013] |
| deep + shots together (K vs A) | −0.00000 | [−0.00105, +0.00099] |
| does adding shots to deep help (K vs H) | **+0.00082** | [−0.00005, +0.00170] |

Better in **6/10 seasons**. The xG channel's bar was 6/6.

## The finding that settles it: the gain is not robust

The two strongest seasons were 2016 (−0.00445) and 2023 (−0.00231), and 2016 is the thinnest-training season in the grid. Splitting by how much history the model had:

| Training history | Δ (H vs baseline) | |
|---|---|---|
| thin (rounds 1–187) | **−0.00197 [−0.00357, −0.00042]** | the ONLY band that clears zero |
| middle (1–298) | +0.00051 [−0.00110, +0.00212] | null |
| thick (1–415) | −0.00101 [−0.00247, +0.00041] | null |
| **excluding 2016 entirely** | **−0.00044 [−0.00140, +0.00047]** | **effect halves** |

Two things follow:

1. **An effect that halves when one season of ten is removed is not a robust effect.** WP013's bar for the xG channel was 6/6 leave-one-season-out folds; this is nowhere near it.
2. **The pattern across training size is non-monotone** (−0.00197, +0.00051, −0.00101), so there is no clean "deep substitutes for missing history" mechanism either — it looks like noise with one strong band, not structure. The thick-history band (−0.00101) does match WP020's −0.00096, which is why WP020 saw what it saw; but that band is itself null.

## The shot channel actively cancels it

`K vs H` is **+0.00082 [−0.00005, +0.00170]** — adding WP019's shot channel on top of deep makes things worse, and in the thin-history band it is decisively worse (+0.00197 [+0.00027, +0.00363]). This is a stronger statement than WP019's own null: the shot channel is not merely uninformative, it *displaces* whatever the deep channel contributes. The premise-check reading that `deep`-on-top-of-shots retained 98% of its value did not transfer.

## The channel was correctly specified throughout

| | learned | truth in the data |
|---|---|---|
| `deep_kappa` | **4.84** | **5.05** |

As with `shot_kappa` (9.33 vs 9.39) in WP019, the null is not an artifact of a misspecified channel.

## Two of my own predictions scored

**Power prediction — wrong.** I predicted MDE ±0.00080 from 115 windows; achieved **±0.00095**. The 1/√n scaling assumed equally informative windows, but TEST_LEN dropped from 4 to 3 (fewer matches per window) and the added early seasons are noisier. **When sizing a run, scale by total test matches and window variance, not window count alone.**

**WP020's location claim — refuted.** WP020 found the gain concentrated in the middle 50% of line movement. At 3x the power it is flat across all three bands (top −0.00096, middle −0.00057, bottom −0.00131). That concentration was noise, as was the "high-line-movement" prediction it replaced. WP010's deficit remains unexplained and nothing here touches it.

## Bottom line

Twenty-one work products; **one mechanism works** (the xG measurement channel, −0.0026 RPS, 6/6 seasons). Deep completions are the closest anything has come since, and they do not survive a properly powered test. The model sits at **+0.0044 RPS behind Pinnacle's pre-close line** on this grid.

The measurement-channel direction is now exhausted for the available data: goals, xG, shots and deep are the four observables, and only xG pays.

## Reproducing

```bash
python work_products/wp021_power_confirmation/prepare_data.py    # 115 windows, deep + shots
python work_products/wp021_power_confirmation/run_cv.py --workers 4   # 345 tasks, ~6.6 h
python work_products/wp021_power_confirmation/analyse.py
```

Needs WP007's `cv_shared_data.pkl`, WP008's `odds_raw.pkl`, WP014's `shots/`, WP020's `team_context.pkl` + `short_to_title.pkl`.
Note: `CODE_TO_FD` in `wp007_structural_fixes/analysis.py` gained 7 pre-2020 clubs (CAR, HUD, HUL, MID, QPR, STO, SWA) so the market join works before 2020 — pure additions, no existing key changed.
