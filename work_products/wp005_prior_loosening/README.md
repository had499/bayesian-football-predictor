# WP005 — Prior Loosening (structural resolution)

**Status: complete. Conclusion — no single shrinkage-prior knob moves the needle. The combined arm (`loose_combo`) shows a small, consistent, but not-quite-significant improvement across three separate tests — worth keeping in mind, not worth shipping on this evidence, not worth more compute chasing it further. Structural work should move to WP006 (per-team sigma).**

## Why

WP004 proved the model's ~8% RPS gap to Pinnacle is a **resolution** problem — its match-by-match probability *ordering* is genuinely worse than the market's, and no post-hoc recalibration (temperature, Platt, isotonic — all tried) closes more than 5% of it. WP003 diagnosed the mechanism: the model over-shrinks, pulling team strengths toward the league mean so standout teams can't separate.

That over-shrinkage is baked into the priors. This work product loosens them, **one knob at a time**, and tests whether that buys resolution back — measured on WP003's 401-match comparison, with a guard against overfitting that yardstick.

**Out of scope:** per-team `sigma` / partial pooling. It's a bigger model change (lets the *data* decide each team's volatility rather than a global setting) and gets its own work product, **WP006**.

## The knobs

The current `ModelConfig` shrinkage settings, and why each is a suspect:

| Field | Current | What it controls | Over-shrink suspicion |
|---|---|---|---|
| `init_scale` | 0.20 | SD of the t=0 team-strength prior `att_0`/`def_0` — the dominant cross-team spread term | least suspicious after a first look (see preview below) — its implied spread is roughly right |
| `home_adv_sd` | 0.02 | per-team home-advantage SD around `home_mu=0.13` | tiny → the prior forces every team's home edge to within ~0.012 of the league mean |
| `sigma_att` / `sigma_def` | 0.008 | AR(1) innovation SD — how far strength moves round-to-round | with `rho≈0.97` the stationary SD is only ~0.03; strengths barely drift within a season |
| `rho_att_alpha` / `rho_def_alpha` | 29 | Beta(29,1) → AR(1) persistence ≈ 0.967 | near a pure random walk with almost no innovation — strengths are near-frozen; combined with tiny `sigma` this is the rigidity that stops in-season form separating teams |

### Phase 1 preview (quick smoke test, 3 arms, 40 draws — the notebook runs all 6 properly)

| | baseline | `init_scale`→0.30 | `home_adv_sd`→0.06 | real EPL |
|---|---|---|---|---|
| attack SD (cross-team) | 0.21 | ~0.30 | 0.21 | — |
| `home_adv` SD (cross-team) | **0.012** | 0.012 | 0.036 | teams' home edges do vary, though the per-team estimate is noisy |
| implied top/bottom scoring ratio | 2.4× | ~4× (at 0.40 it was ~5×) | 2.4× | ~1.7× (p95/p5) to ~2.6× (max/min), per team-season |
| implied P(H)/P(D)/P(A) | 0.39 / 0.28 / 0.31 | — | 0.39 / 0.28 / 0.31 | 0.43 / 0.24 / 0.33 |
| implied league λ (home) | 1.14 | — | 1.13 | 1.56 |

Two things this already suggests, to confirm with the full triage:
1. **`init_scale` is probably not the culprit** — its baseline implied spread (2.4×) is already in the right ballpark, and pushing it to 0.40 overshoots to ~5×. `loose_init` starts at **0.30** for that reason.
2. **`home_adv_sd` is clearly pinning per-team home advantage** (implied SD 0.012), and **the AR(1) rigidity** (`sigma`≈0.008 + `rho`≈0.97) is the other prime suspect — the prior barely lets a team's strength move over a season, so a team in hot form can't pull away. `loose_home`, `loose_sigma`, and `loose_rho` are the arms to watch.
3. The prior also implies a low base scoring rate (λ≈1.14 vs real 1.56) and slightly too many draws — worth noting but a separate issue from resolution.

## Method — three phases

### Phase 1 — Prior-predictive triage (in-notebook, minutes)

For the baseline config and five one-knob-loosened candidates (`init_scale` 0.20→0.30, `home_adv_sd` 0.02→0.06, `sigma_att/def` 0.008→0.020, `rho_*_alpha` 29→12 ≈ Beta(12,1)≈0.92, and a combo), draw from `pm.sample_prior_predictive` and summarise what the prior *implies*:

- cross-team SD of `attack` / `defence`
- cross-team SD of `home_adv`
- implied top-vs-bottom team home-goals ratio
- implied league goal rate
- implied P(home / draw / away)

against real EPL reference values computed from `df_cv` itself. **An arm whose prior implies an absurd spread or scoreline is disqualified before any CV.** (The preview above already did a first pass — `home_adv_sd` and the AR(1) rigidity look like the live suspects, `init_scale` less so.)

### Phase 2 — Screening CV (heavy; run yourself)

Every 2nd window (18 of 35) via the WP001/WP002 subprocess-per-window harness — `scripts/run_cv_window.py` now takes `--config-json '{"init_scale": 0.40}'` for arbitrary `ModelConfig` overrides (unknown keys are rejected, not silently dropped; the overrides are recorded in every result row). `baseline` reuses WP001's finished checkpoint filtered to the screening windows, so only the 5 loosened arms actually run (~90 fits).

Per arm, using WP003's odds join + scoring:
- pooled RPS on the screening matches
- RPS gap to Pinnacle (paired bootstrap)
- **the WP003 disagreement-decile pattern** — if a looser prior makes the gap grow *faster* with disagreement, that's the model overfitting to noise, not gaining resolution. Disqualify.

### Phase 3 — Confirmation (heavy; run yourself, finalists only)

For the 1–2 arms that screened best:
- full 35-window CV
- a **cold held-out-season check** the tuning never touched: train through 2024-25, score every 2025-26 match in one shot. The screening metric (gap to Pinnacle on the 401 walk-forward matches) is what was optimised, so a finalist has to also hold up on a split it wasn't tuned on.

Ship a change only if it improves resolution on the full CV **and** doesn't inflate the disagreement pattern **and** holds up on the held-out season.

## Results

### Phase 1 — prior-predictive triage

Full 6-arm run (see the table in "The knobs" preview above for the numbers — the full triage matched the smoke-test preview closely: `init_scale`'s baseline-implied spread was already close to realistic, `home_adv_sd` and the AR(1) rigidity were the standout suspects, and `loose_combo`/`loose_init`@0.40-equivalent pushed the top/bottom scoring ratio to a implausible ~3.4–3.7×, above the real ~2.5–2.6× — a caution carried into how the Phase 2/3 results below should be read, not a disqualification, since `loose_init` in the actual sweep used the gentler 0.30 value).

### Phase 2 — screening (all 6 arms, clean)

18 screening windows, 195 test matches, joined to Pinnacle closing odds (same harness as WP003):

| arm | n | model RPS | gap vs Pinnacle | 95% CI | gap, top-25% disagreement |
|---|---|---|---|---|---|
| `baseline` | 195 | 0.1916 | +0.0129 | [+0.0058, +0.0200] | +0.0455 |
| `loose_init` (`init_scale` 0.20→0.30) | 195 | 0.1909 | +0.0123 | [+0.0051, +0.0194] | +0.0438 |
| `loose_home` (`home_adv_sd` 0.02→0.06) | 195 | 0.1914 | +0.0128 | [+0.0056, +0.0199] | +0.0396 |
| `loose_sigma` (`sigma_att/def` 0.008→0.020) | 195 | 0.1912 | +0.0126 | [+0.0056, +0.0195] | +0.0393 |
| `loose_rho` (`rho_*_alpha` 29→12) | 195 | 0.1919 | +0.0132 | [+0.0059, +0.0202] | +0.0466 |
| `loose_combo` (init+home+sigma) | 195 | 0.1905 | +0.0118 | [+0.0049, +0.0188] | +0.0400 |

All six arms are within noise of each other — the full spread across all of them (+0.0118 to +0.0132) is smaller than any single arm's CI width (~0.014). No individual knob shows an effect. The only consistent pattern: 4 of 5 loosened arms nudge the top-25%-disagreement gap down from baseline's +0.0455 — `loose_rho` doesn't. `loose_combo` has the best point estimate on the primary metric, `loose_sigma` on the secondary one — differences this small aren't a basis for picking a winner on their own, but `loose_combo` was carried into Phase 3 as the single "give loosening its best shot" candidate (combines three knobs rather than promoting one that individually showed nothing).

### Phase 3 — confirmation (`loose_combo`, full 35-window CV + held-out 2025-26 season)

**Full CV, 401 matches (361 with Pinnacle odds) — properly powered, not a screen:**

| arm | n | model RPS | gap vs Pinnacle | 95% CI |
|---|---|---|---|---|
| baseline (= WP001/WP003) | 361 | 0.1925 | +0.0139 | [+0.0080, +0.0198] |
| `loose_combo` | 361 | 0.1916 | +0.0130 | [+0.0072, +0.0190] |

**Paired, same 401 matches:** RPS(`loose_combo`) − RPS(baseline) = **−0.0008**, 95% CI **[−0.0018, +0.0002]**. Negative means `loose_combo` is slightly better — but the CI just barely fails to exclude zero (upper bound +0.0002). Not significant, but close, and in the favourable direction.

**Held-out 2025-26 season (single train/test split, train ≤ round 172, predict all 36 rounds cold), n=210:**

| arm | model RPS | gap vs Pinnacle | 95% CI |
|---|---|---|---|
| baseline | 0.2017 | +0.0023 | [−0.0048, +0.0098] |
| `loose_combo` | 0.2013 | +0.0020 | [−0.0052, +0.0095] |

Two things stand out here. First, `loose_combo` is again marginally better than baseline — same direction as the screen and the full CV, third time in a row. Second, and more strikingly: **on this held-out season, even baseline's gap to Pinnacle is not statistically distinguishable from zero** — a much smaller gap than WP003's robust, CI-excluding-zero +0.0139 on the pooled 401-match walk-forward comparison. Read that as season-specific variance from a single 210-match split (this is a much less powerful test than the 401-match rolling comparison, and 2025-26 may simply be an easier-to-predict season), **not** as evidence the model has closed the gap to the market generally — WP003's result, built from 6 seasons of rolling windows, is the one to trust for that claim.

**Overall verdict:** three separate tests (screen, full CV, held-out season) all point the same small direction for `loose_combo` — never significant on its own, but consistently not-negative across every cut of the data, which is worth more than a single null result would be. It doesn't clear the bar to ship (the primary full-CV test's CI misses excluding zero by a hair), and it doesn't change WP005's headline conclusion (no single shrinkage-prior knob has a real effect on resolution). But `loose_combo`'s settings (`init_scale=0.30, home_adv_sd=0.06, sigma_att/def=0.020`) are worth carrying forward as the new default candidate to combine with WP006's per-team `sigma` work, rather than reverting to the original tight priors — there's a small, repeatedly-directionally-positive signal here that a bigger test (more seasons, or WP006's added flexibility) might resolve one way or the other.

## Reproducing

```bash
cd work_products/wp005_prior_loosening
jupyter lab wp005_prior_loosening.ipynb
```

All checkpoints in this folder are the completed, clean runs behind the Results above (`cv_checkpoint_<arm>.pkl` = Phase 2 screen, `cv_checkpoint_full_loose_combo.pkl` = Phase 3 full CV, `cv_checkpoint_holdout_*.pkl` = Phase 3 held-out season) — re-running any cell will detect them and skip straight to "already done." Delete a specific checkpoint to force that piece to redo. Requires `../wp003_bookmaker_benchmark/odds_raw.pkl` (from running WP003 once) for all the analysis cells.
