# WP002 — xG / Dixon-Coles Ablation

**Status: all 4 arms complete and fresh — every bug below (1-3) is fixed and reflected in the numbers.**

**Headline finding, final: neither xG nor Dixon-Coles makes a statistically detectable difference, alone or combined.** All 4 arms score within noise of each other on MAE and pooled RPS, and all 5 paired comparisons come back "no" — every 95% CI includes zero. This is the same conclusion the pre-bug-3-fix version of this README reached for Dixon-Coles — but that earlier version reached it via a test that was structurally incapable of detecting a DC effect either way (see bug #3), so it wasn't actually evidence of anything. This version's test can detect a DC effect (it moved individual windows' log-likelihood by a real amount in WP001 — see that README), and still comes back null. That's a meaningfully stronger result than coincidentally landing on the same answer.

## Correction history

Three real bugs were found and fixed in `scripts/run_cv_window.py` since this ablation first ran.

1. **Missing xG term at prediction time**: predictions never included the learned `beta_xG · log(xG)` term, even when a model was trained with `use_xG=True` — training fit `attack`/`defense` assuming part of the signal was carried by xG, then prediction silently dropped that term and scored using only `attack`/`defense`/`home_adv`. This made any xG-enabled arm look worse than it should, independent of whether xG is actually useful — it's why an earlier version of this README reported `both`/`xg_only` as the two *worst* arms. **Affected all 4 arms** (the bug was in shared code, not xG-specific), fixed.
2. **Swapped team IDs**: the script rebuilt its own team→ID mapping via `pd.unique()` on the raw (unsorted) dataframe, while `prepare_model_data` (used for both training and the xG lookup) sorts by datetime first — since `pd.unique()`'s output order depends on row order, this silently assigned two teams (Nottingham Forest, Bournemouth) swapped IDs between the two mappings, corrupting predictions for any test match involving either team. **Affected all 4 arms equally** (had nothing to do with `use_xG`), fixed by deleting the script's hand-rolled lookups entirely and reading everything (`t_idx`/`team_idx`/`opp_idx`/`xG`) straight off the `ModelData` `prepare_model_data` produces — see `predict_rows` in `src/football_model/model/predict.py`.
3. **Dixon-Coles never applied at evaluation time**: `dixon_coles_adjustment` only ever ran *during training*, as a `pm.Potential` shaping the posterior — nothing downstream (the script's log-likelihood, or this notebook's RPS/calibration) ever applied the actual `tau(goals; lambda, rho)` correction it exists for. Every `dc_only`/`both` result was scored as if Dixon-Coles were off, silently, regardless of the setting — the paired comparisons involving DC (`both vs neither`, `DC's marginal contribution`, `DC alone vs neither`) were therefore measuring almost nothing by construction, not genuinely testing whether DC helps. **Affects only `dc_only`/`both`** — `neither`/`xg_only` never touch `rho_dc` and are unaffected. Fixed via `dixon_coles_log_correction`/`dc_outcome_probs` in `src/football_model/model/predict.py`, which both the script and this notebook's `outcome_probs_from_lambda` now route through instead of a plain independent-Poisson grid.

All three bugs are fixed and reflected in every number below — `both`/`dc_only` were re-run from scratch after bug #3's fix (`neither`/`xg_only` never touch `rho_dc` and didn't need re-running). Verified the fix itself before trusting these results: full test suite passes (52 tests, including a cross-check of the numpy `dixon_coles_tau` against `components.py`'s actual PyTensor version), and a spot-check window showed the correction moves LL by a small but real, non-zero amount (5.94 → 6.06 for that one window) before the full re-run confirmed the same pattern at scale.

## Method

Same 35-window walk-forward CV as WP001 (identical data, identical windows — reused directly from `wp001_walkforward_cv_baseline/cv_shared_data.pkl`, so this is an apples-to-apples comparison, not a re-fetch). Four arms, only the feature flags differ:

| Arm | `use_xG` | `use_dixon_coles` |
|---|---|---|
| `both` (= WP001's result, reused, not re-run) | ✅ | ✅ |
| `neither` | ❌ | ❌ |
| `xg_only` | ✅ | ❌ |
| `dc_only` | ❌ | ✅ |

`both` is copied directly from WP001's checkpoint (identical config, identical data — same script, same arguments) rather than re-running 35 windows we already have fresh. The other 3 arms run from inside `wp002_ablation.ipynb` itself (the first code cell), each window in its own subprocess (same isolation approach as WP001, for the same reason — see WP001's README).

### Why paired comparisons, not just per-arm vs. naive

All 4 arms are scored on the *identical* set of test matches (same windows, same rounds). That means predictions can be paired match-for-match across arms, rather than comparing four unpaired distributions — a paired bootstrap on the RPS difference cancels out match-to-match difficulty/noise that affects every arm equally, giving a much more sensitive test for a single feature's marginal effect than comparing each arm to naive separately. Verified the pairing is valid before running the real comparison: match order for a given window is deterministic (depends only on the data, not on model config), confirmed by re-running window 1 under two different configs and checking the goals sequence comes back identical.

## Results

All 4 arms: 35/35 windows, 401 pooled test matches each (identical matches across arms — same data, same windows).

### Per-arm summary

| Arm | MAE | Mean LL improvement over naive (95% CI) | Pooled RPS |
|---|---|---|---|
| `both` | 0.921 | 1.60 [0.99, 2.23] | 0.19629 |
| `neither` | 0.921 | 1.53 [0.93, 2.15] | 0.19639 |
| `xg_only` | 0.921 | 1.57 [0.97, 2.19] | 0.19636 |
| `dc_only` | 0.921 | 1.56 [0.95, 2.20] | 0.19633 |
| naive (same 401 matches) | — | — | 0.2341 |

All 4 arms clearly beat naive (0.234) on RPS, by essentially the same margin. But **the 4 arms remain indistinguishable from each other**: MAE agrees to 3 decimal places, and pooled RPS spans a range of just 0.0001 — far smaller than sampling noise on 401 matches. This now includes a genuinely DC-aware `both`/`dc_only` (bug #3's fix applied) — the near-equivalence isn't an artifact of DC being silently ignored anymore.

### Paired comparisons (same 401 matches, matched match-for-match — see README's "Why paired comparisons")

| Comparison | Mean RPS improvement | 95% CI | Significant? |
|---|---|---|---|
| Both vs. neither (combined effect) | +0.00011 | [−0.0005, 0.0008] | no |
| xG's marginal effect on top of DC (both vs. dc_only) | +0.00004 | [−0.0004, 0.0005] | no |
| DC's marginal effect on top of xG (both vs. xg_only) | +0.00008 | [−0.0004, 0.0005] | no |
| xG alone vs. neither | +0.00003 | [−0.0004, 0.0005] | no |
| DC alone vs. neither | +0.00006 | [−0.0004, 0.0005] | no |

Every single comparison's CI includes zero — none of the five ways of slicing "does this feature help" clears the bar. Unlike the pre-bug-3-fix version of this table, DC's rows here are a genuine test: `dixon_coles_log_correction`/`dc_outcome_probs` are actually applied, and WP001 already showed this correction has a real, non-zero effect on individual windows' log-likelihood (5.94 → 6.06 for one window) — so this null result isn't "the test couldn't see it," it's "the effect is real but too small/inconsistent across 401 matches to show up in pooled RPS."

**Practical takeaway, final**: on this dataset, prefer `use_xG=False, use_dixon_coles=False` by default. Both recommendations are now fully validated — not just "the bug made it look this way," but confirmed under a version of the code capable of detecting either effect. This performs identically to the fully-featured config, is simpler, trains faster, and removes two moving parts that this ablation couldn't justify keeping. That's not evidence either feature could never help anywhere — a bigger dataset, or matches where low-scoring correlation genuinely matters more (cup ties, relegation six-pointers), might reveal something a season-spanning average washes out — just that neither is pulling weight here.

### Calibration

Not re-checked per arm — WP001 already covers pooled calibration for the `both` configuration in detail (see its README, now also reflecting bug #3's fix). Given all 4 arms are statistically indistinguishable on both MAE and RPS here, there's no reason to expect a materially different calibration story for `neither`/`xg_only`/`dc_only`, but that's an inference from the equivalence above, not a separately verified result.

## Reproducing / resuming

Open `wp002_ablation.ipynb` and run top to bottom. `cv_checkpoint_<arm>.pkl` for all 4 arms in this folder already hold the completed, fully-corrected run above — re-running the ablation cell will detect them and report all 4 arms already done. `cv_checkpoint_both.pkl` is a copy of WP001's checkpoint — if WP001 ever gets re-run again (e.g. its live-fetched data shifts, or a future fix), re-copy it here rather than letting it drift out of sync.
