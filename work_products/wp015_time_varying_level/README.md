# WP015 — Time-varying home advantage and league scoring level

**Status: complete. Verdict: NULL on both — and the falsification test failed, which is the more useful result.** Letting league-wide home advantage drift is worth +0.00026 RPS [−0.00063, +0.00125]; letting the global scoring level drift is worth +0.00000 [−0.00022, +0.00019]. Neither helps, and fixing home advantage did **not** shrink WP009's away-favourite asymmetry.

## Why this looked like the best remaining idea

The model has free parameters for *levels* (`kappa`, `intercept`, `home_mu`) but almost none for level *drift* — team strengths were the only thing allowed to move over time. Two global parameters looked badly frozen:

- home advantage appeared to range **0.008 (2020, empty stadiums) to 0.300**
- league scoring log-level appeared to range **0.249 to 0.494** (2.57 → 3.28 goals/match)

And unlike `kappa` drift or Dixon-Coles — both common-mode, both cancelling in H/D/A — a wrong home advantage is **directional**: it pushes P(home) up and P(away) down, so it should corrupt outcome discrimination. There was also a corroborating signature: WP009 found the model under-rates away favourites, exactly what an over-estimated HFA produces.

## The design

Built as **one coherent re-parameterization**, not two bolt-ons. `build_model` now *refuses* two combinations, both tested:

- `time_varying_level` + soft centering → two representations of the same level drift, a poorly identified posterior
- `time_varying_level` + `xg_kappa_by_season` → collinear (both mediate latent → observation level)

Three arms on top of WP013's shipping config, so a win would be attributable:

| Arm | Change |
|---|---|
| `E_hfa_only` | league-wide HFA follows a slow walk; per-team deviations kept |
| `F_level_only` | hard centering + drifting global scoring level |
| `G_both` | both together |

## Results

| Arm | Pooled RPS | Δ vs shipping config | CI (cluster) |
|---|---|---|---|
| `S3_xgchannel_only` | 0.20481 | — | — |
| `E_hfa_only` | 0.20507 | +0.00026 | [−0.00063, +0.00125] |
| `F_level_only` | 0.20481 | +0.00000 | [−0.00022, +0.00019] |
| `G_both` | 0.20496 | +0.00015 | [−0.00072, +0.00116] |
| HFA on top of the level fix (G vs F) | | +0.00014 | [−0.00069, +0.00112] |

`F_level_only` is exactly zero on a tight interval. Null in all six leave-one-season-out folds. Gap to Pinnacle pre-close unchanged (+0.00409 → +0.00409 for F).

## The falsification test failed

The check that made this worth running, rather than reading a null RPS and shrugging:

| Arm | home-fav gap | away-fav gap | asymmetry |
|---|---|---|---|
| `S3_xgchannel_only` | +0.00112 | +0.00857 | **+0.00745** |
| `E_hfa_only` | +0.00098 | +0.00955 | +0.00856 |
| `G_both` | +0.00133 | +0.00872 | +0.00738 |

Fixing home advantage **did not shrink the away-favourite asymmetry**. Two independent lines of evidence pointed at HFA and the causal link is dead: WP009's asymmetry is *not* caused by a mis-estimated time-invariant home advantage. It remains unexplained.

## Why it failed — I overstated the defect four-fold

The model learned **almost exactly the right amount of drift, and it didn't help**:

| | per season |
|---|---|
| noise-corrected TRUE HFA variation | **0.069** |
| what the model learned (σ=0.0121/round × √34) | **0.071** |
| noise-corrected true level variation | 0.053 |
| what the model learned (σ=0.0075/round) | 0.044 |

My "0.008 to 0.300" was a raw range across season *estimates*, each carrying **±0.062** of sampling noise from only 380 matches. Noise-corrected, season-to-season HFA SD is **0.069**, not the ~0.29 range implied. I overstated the flaw by roughly 4× by quoting a range instead of subtracting the noise floor — the same correction I *did* apply in WP007 to the year-on-year attack figure, and then walked straight past here.

The 2020 value of 0.008 is a single season's estimate with ±0.062 attached. The COVID effect is likely real but much smaller than the raw number suggests.

## Caveats

* **96 divergences**, against 0 in every previous arm — the extra walks create harder geometry. Results are consistent across arms so I don't think they're corrupted, but I'd want a longer run before leaning on these numbers hard. max rhat 1.043.
* `F_level_only` doubles as the **`L2_level` re-test** the backlog called for (hard centering + a drifting intercept, the parameterization that was previously impossible). It is null, so that backlog item is closed.
* This tests *slow-walk* drift. An abrupt regime change (a COVID indicator variable) is a different specification and untested — though given the noise-corrected variation is small, I would not expect it to pay.

## Reproducing

```bash
python work_products/wp015_time_varying_level/run_cv.py --workers 4   # 90 tasks, ~125 min
python work_products/wp015_time_varying_level/analyse.py
```

Needs WP007's `cv_shared_data.pkl`/`checkpoints/`, WP013's `checkpoints/`, WP008's `odds_raw.pkl`.
