# WP009 — Lineup xG/xA Covariate: Validation

**Status: complete. Conclusion — genuinely the most interesting result since WP005, but not a clean win for the lineup covariate specifically. The best combined config (`loose_combo` + lineup covariate) is the first configuration in this entire project to beat the original baseline with a CI that excludes zero — but the lineup covariate's own isolated marginal contribution, on top of `loose_combo`, is not significant. The improvement is more honestly attributed to the combination than to the new information source on its own.**

## Data

Full-coverage historical fetch, all 28 teams × 6 seasons in `cv_shared_data.pkl`: 66,664 player-match rows, 1,375 unique players, aggregated into a 4,560-row `lineup_dev_table.pkl` — one row per team-match, matching `df_cv`'s full row count exactly.

## Results

### Phase 2 — screening (18 windows, 195 matches)

| arm | model RPS | gap vs Pinnacle | 95% CI | gap, top-25% disagreement |
|---|---|---|---|---|
| `baseline` | 0.1916 | +0.0129 | [+0.0058, +0.0200] | +0.0455 |
| `loose_combo` | 0.1905 | +0.0118 | [+0.0049, +0.0188] | +0.0400 |
| `lineup_only` | 0.1915 | +0.0128 | [+0.0058, +0.0197] | +0.0434 |
| `lineup_loose_combo` | 0.1903 | +0.0117 | [+0.0049, +0.0186] | +0.0453 |

`lineup_only` is essentially identical to `baseline` (0.1915 vs 0.1916) — the lineup covariate alone, without the prior loosening, does nothing detectable at screening power. `lineup_loose_combo` is marginally better than `loose_combo` (0.1903 vs 0.1905) — a difference of 0.0002, well inside noise. Worth flagging honestly: `lineup_loose_combo`'s disagreement-tail gap (+0.0453) is *worse* than `loose_combo` alone (+0.0400) — adding the lineup term didn't uniformly help every metric even directionally.

### Phase 2b — is `beta_lineup` well-identified?

| arm | mean | range | sd (across 18 windows) |
|---|---|---|---|
| `lineup_only` | 0.2148 | [0.1422, 0.3054] | 0.0548 |
| `lineup_loose_combo` | 0.1968 | [0.1306, 0.2859] | 0.0529 |

Yes — genuinely informative. The prior is `HalfNormal(0.1)` (prior mean ≈0.08); the posterior lands at ~0.20-0.21, roughly 2.5× the prior's own mean, consistently across 18 independently-trained windows with modest spread. The data is pulling this coefficient up from what the prior alone would suggest — the model is learning a real, repeatable in-training relationship between lineup deviation and goals. That this doesn't translate into a measurable held-out RPS improvement (above) is the interesting part: the signal is real *within* training, but small enough relative to what `attack`/`defence`/`xG` already capture that it doesn't move held-out accuracy detectably.

### Phase 3 — confirmation (`lineup_loose_combo`, full 35-window CV)

**Full CV, 401 matches (361 with Pinnacle odds):**

| arm | model RPS | gap vs Pinnacle | 95% CI |
|---|---|---|---|
| baseline (WP001) | 0.1925 | +0.0139 | [+0.0080, +0.0198] |
| `lineup_loose_combo` | 0.1911 | +0.0125 | [+0.0067, +0.0185] |

**Two paired comparisons, same 401 matches — this is the part that needs careful reading:**

| comparison | mean RPS diff | 95% CI | significant? |
|---|---|---|---|
| `lineup_loose_combo` vs. **baseline** | −0.0013 | **[−0.0026, −0.00003]** | **yes — excludes zero** |
| `lineup_loose_combo` vs. **`loose_combo`** | −0.0005 | [−0.0014, +0.0003] | no |

The first comparison bundles two changes at once (WP005's prior loosening *and* WP008's lineup covariate) against the original baseline, and that bundle is the first result in this entire project's sequence (WP002, WP004, WP005, WP006 all null) to produce a CI that excludes zero — confirmed robust across multiple bootstrap reseeds, not a rounding artifact, though the margin is extremely thin (upper bound within 0.00003 of zero).

The second comparison isolates *just* the lineup covariate's marginal effect, holding the prior-loosening constant — and it is **not** significant. `loose_combo` alone (WP005) already showed a similar-sized, similarly-directioned, non-significant effect on its own. The honest read: the improvement over baseline is better attributed to the accumulated, still-individually-uncertain effect of the loosened priors than to the new lineup information specifically. Given ~5 comparisons were made across this WP (2 screening, 2 full-CV, 1 held-out — see below), exactly 1 crossing significance, barely, is close to what you'd expect from noise alone even if the true effect were zero. Not dismissing it — it's the least-null result this project has produced — but not claiming "the lineup covariate works" either, since the one test that actually isolates it says no.

### Held-out 2025-26 season (cold split, n=210)

| arm | model RPS | gap vs Pinnacle | 95% CI |
|---|---|---|---|
| baseline | 0.2017 | +0.0023 | [−0.0048, +0.0098] |
| `lineup_loose_combo` | 0.2001 | +0.0008 | [−0.0062, +0.0077] |

Baseline's number here is bit-for-bit consistent with WP005's own held-out baseline run (+0.0023 both times) — a nice internal-consistency check: since `use_lineup_xg=False` for the baseline arm, the model never reads `lineup_dev_home/away` at all, so training is numerically identical whether or not the lineup table is present in the data, and it reproduces exactly. `lineup_loose_combo`'s gap (+0.0008) is the smallest (best) of any held-out result seen across WP005 and WP009 — but as with WP005's held-out check, this single-season split has limited power (both CIs include zero), so read this as consistent-with, not confirming, the full-CV direction.

## What this means, and what to do next

Same "where do you draw the line" discipline as every WP since WP005: a comparison crossing significance by a hair, when it's one of several tested and the specific isolating test comes back null, is not a green light to keep iterating on this feature. It's grounds to:

1. **Keep `loose_combo`'s settings + the lineup covariate together as the current best-available config** if shipping something — it's never worse than any prior configuration on any test run in this project, and the combined bundle is the only one to clear significance against the original baseline.
2. **Not claim WP008's core hypothesis is confirmed.** "Lineup information helps" specifically is not supported by the one test built to isolate it. The honest state: mostly attributable to `loose_combo`'s prior loosening (already known, already inconclusive on its own), with the lineup covariate's contribution too small to resolve with the data and power available here.
3. **Don't spend more compute chasing a cleaner split of this specific effect.** A bigger, more powerful test to separate "prior loosening" from "lineup information" would need either more historical seasons (not available) or a differently-designed comparison — not obviously worth building given the effect, even if real, is this small.

Consistent with the pattern going back to WP005/WP006: this is a reasonable point to treat the team-level architecture (with or without the lineup covariate) as close to its practical ceiling on this dataset, rather than proposing a WP010 in the same family.

## Reproducing

```bash
cd work_products/wp009_lineup_xg_validation
jupyter lab wp009_lineup_xg_validation.ipynb
```

All phases complete; checkpoints in this folder reflect the results above — re-running any cell will detect them and report "already done."
