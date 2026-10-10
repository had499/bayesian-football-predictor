# WP013 — Is the xG channel robust, and does it work on its own?

**Status: complete. Two clear positives, and both correct earlier claims in this project's write-ups.**

1. **The xG measurement channel works entirely on its own.** `L0_baseline` + xG channel alone is −0.00258 RPS [−0.00414, −0.00095] vs `L0`. The four other changes bundled into `L5_xgchannel` — posterior-averaged prediction, intercept/hard-centering, random-walk dynamics, promotion offset — add **+0.00005 [−0.00085, +0.00096]**, i.e. nothing.
2. **Stripped of that bundle it is markedly more robust.** Leave-one-season-out, the isolated version holds in **6 of 6** folds; the full `L5` bundle holds in only 5 of 6.

**Recommended config: `L0_baseline` + `use_xg_channel=True`.** Simpler, more robust, tighter CI, same performance.

## Why this needed doing

WP007's ladder was cumulative, so "L0 + xG channel alone" had never been measured — I flagged that gap repeatedly but never closed it. And I had been describing the xG channel as fragile because its effect didn't survive dropping 2024. That was tested on the *bundle*, and 2024 is the worst possible fold by construction, since it was the one individually-significant season. Both claims turn out to be wrong in the model's favour.

## Results

30 windows, 1,314 paired matches, 2020-21 through 2025-26. `S3_xgchannel_only` uses `L0`'s own plugin prediction mode, so the xG channel is the *only* difference from `L0`.

| Arm | Pooled RPS | CI (cluster) |
|---|---|---|
| `L0_baseline` | 0.20739 | [0.19779, 0.21668] |
| `L5_xgchannel` (full bundle) | 0.20486 | [0.19511, 0.21442] |
| **`S3_xgchannel_only`** | **0.20481** | [0.19540, 0.21388] |

### 1. The xG channel does all the work

| Comparison | ΔRPS | CI (cluster) | Verdict |
|---|---|---|---|
| xG-channel-only vs `L0` — **the isolated effect** | **−0.00258** | **[−0.00414, −0.00095]** | better |
| full `L5` bundle vs `L0` — cumulative | −0.00253 | [−0.00459, −0.00034] | better |
| `L5` vs xG-channel-only — **what the other 4 rungs add** | **+0.00005** | **[−0.00085, +0.00096]** | **no clear difference** |

The four extra changes contribute nothing, and they *widen* the interval — variance without signal. This finally answers the question left open since WP007: they are not quietly interacting with the xG channel, they are simply inert.

### 2. Isolated, the effect is robust to dropping any single season

| Fold | `L5` bundle vs `L0` | xG-channel-only vs `L0` |
|---|---|---|
| all seasons | −0.00253 [−0.00459, −0.00034] ✓ | −0.00258 [−0.00414, −0.00095] ✓ |
| drop 2020 | −0.00307 ✓ | −0.00293 ✓ |
| drop 2021 | −0.00271 ✓ | −0.00282 ✓ |
| drop 2022 | −0.00221 ✓ | −0.00229 ✓ |
| drop 2023 | −0.00253 ✓ | −0.00286 ✓ |
| **drop 2024** | −0.00156 **[−0.00381, +0.00079] ✗** | **−0.00195 [−0.00367, −0.00022] ✓** |
| drop 2025 | −0.00312 ✓ | −0.00266 ✓ |

(✓ = CI excludes zero.) **The fragility was the bundle's, not the xG channel's.** The four null changes were adding enough noise to break the 2024 fold; removing them makes all six hold.

### 3. Gap to the market is unchanged

| Arm | Gap vs Pinnacle pre-close |
|---|---|
| `L0_baseline` | +0.00650 [+0.00332, +0.00948] |
| `L5_xgchannel` | +0.00392 [+0.00122, +0.00663] |
| `S3_xgchannel_only` | +0.00409 [+0.00129, +0.00682] |

Statistically indistinguishable from the bundle, so WP008's and WP010's market conclusions carry over to the simpler config unchanged.

Sampler: 0 divergences; max rhat 1.035 — slightly above the 1.01 ideal, acceptable for exploratory CV at 600 draws but worth noting.

## What this changes

- **Ship `L0` + `use_xg_channel=True`**, not `L5`. Same accuracy, fewer moving parts, and the robustness argument is strictly stronger.
- **Retire `dynamics="random_walk"`, `use_promoted_effect`, `use_intercept` and posterior-averaged prediction as *performance* changes.** They are measurably inert here. (Posterior-averaged prediction remains the more *correct* thing to do on principle — it represents real parameter and forecast uncertainty — and it is not harmful; it just doesn't pay at this horizon. Worth keeping for that reason, not for RPS.)
- **The one validated win is more solid than previously written up.** Earlier READMEs describe it as "real but not season-robust." On the isolated config that is no longer accurate, and WP007's README has been corrected.

## Caveats

* The isolation arm uses plugin prediction to match `L0` exactly. That confounds nothing for the isolated comparison, but it means `S3` is not simply "`L5` minus four things" — it is "`L0` plus one thing." The `L5` vs `S3` row is the cleaner read of what the bundle adds.
* Leave-one-season-out reuses the same 30 fits per arm, so folds are not independent refits; it tests sensitivity of the *estimate* to each season, not refit stability.
* max rhat 1.035 on the isolation arm is a little high; a longer run would tighten it, though 0 divergences across 30 windows is reassuring.
* This does not revisit whether a *larger* xG-channel effect is available — weight 0.5 vs 1.0 was already null in WP007 (`S1_xgweight1`).

## Reproducing

```bash
python work_products/wp013_xg_channel_robustness/run_cv.py --workers 4   # 30 windows, ~31 min
python work_products/wp013_xg_channel_robustness/analyse.py
```

Needs WP007's `cv_shared_data.pkl` and `checkpoints/`, and WP008's `odds_raw.pkl`.
