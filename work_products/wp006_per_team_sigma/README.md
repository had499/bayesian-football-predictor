# WP006 — Per-Team Sigma (partial pooling)

**Status: complete. Conclusion — partial pooling of team volatility doesn't help either. This is the fifth structural/feature attempt in a row (xG, Dixon-Coles, global prior loosening, now per-team sigma) with no real effect on resolution. Phase 3 deliberately skipped — Phase 2 produced no finalist worth confirming. Recommendation below: stop iterating on team-level architecture.**

## Why

WP005 found that loosening the *global* shrinkage priors — one shared `init_scale`, `home_adv_sd`, `sigma_att/def` for every team — doesn't recover resolution, individually or combined, though the combined arm (`loose_combo`) showed a small, consistent-but-not-significant improvement across three separate tests.

This tests a structurally different mechanism: **let each team's own AR(1) innovation SD be partially pooled toward a shared population value, instead of forcing every team through one global sigma.** A promoted side or a team mid-managerial-change plausibly needs more round-to-round volatility than a stable top-six squad — a global knob can't represent that; a global bump (WP005) helps and hurts different teams simultaneously, which is a coherent explanation for why it read flat there.

## What changed (real model code, not a config override)

- **`src/football_model/model/priors.py`** — new `ar1_hierarchical_sigma(name, n_teams, pop_scale)`: `sigma_pop ~ HalfNormal(pop_scale)`, then each team's own sigma is `raw * sigma_pop` where `raw ~ HalfNormal(1)` is fixed and doesn't depend on `sigma_pop` — non-centered, the same funnel-avoidance pattern as `home_advantage_prior`'s `home_adv_raw`, just via multiplicative scaling (HalfNormal is a scale family) instead of an additive shift. Also `rho_prior`, splitting the rho half out of `ar1_hyperpriors` so it can be reused when sigma comes from the hierarchical path instead.
- **`src/football_model/model/components.py`** — `ar1_team_process`'s `sigma_t` line now broadcasts against `season_arr[:, None]` instead of `season_arr`, so it works whether `sigma` is the original scalar or a new `(n_teams,)` vector. Backward-compatible by construction (a regression test cross-checks the scalar path against a manual numpy recursion) — every existing config's behaviour is unchanged.
- **`src/football_model/types/model_data.py`** — new `ModelConfig.use_per_team_sigma: bool = False`. When `True`, `sigma_att`/`sigma_def` are reinterpreted as the population HalfNormal's scale rather than one global value.
- **`src/football_model/model/model.py`** — the standard AR(1) branch now checks the flag and calls `ar1_hierarchical_sigma` instead of a plain `pm.HalfNormal` when set.
- **`scripts/run_cv_window.py`** — no changes needed to *run* this (WP005's `--config-json` plumbing already validates and passes through arbitrary `ModelConfig` fields, `use_per_team_sigma` included — proven end-to-end before any of this was written up). One addition: each window's fitted `sigma_att_team`/`sigma_def_team` (posterior mean per team) is now recorded in the result row when `use_per_team_sigma=True`, so Phase 2 can check whether the fitted values land on the teams you'd expect.

**Tests** (all new, all passing before any CV was run): `tests/test_priors.py` (non-centered property, `sigma_team = raw * sigma_pop` exactly, population scale actually shifts the distribution), `tests/test_components.py` (scalar-sigma regression guard via manual recursion, per-team-sigma manual recursion, and a variance check proving a low-sigma team really does move less than a high-sigma one), `tests/test_model.py` (`build_model` registers the right variable names with the flag on/off, builds and samples cleanly). Full suite: 66/66 passing.

## Arms

Reusing WP001/WP005 checkpoints where possible — no compute wasted re-running configs already fit:

| arm | `use_per_team_sigma` | `init_scale` | `home_adv_sd` | `sigma_att`/`def` (or pop scale) | source |
|---|---|---|---|---|---|
| `baseline` | ❌ | 0.20 | 0.02 | 0.008 | WP001's checkpoint, reused |
| `loose_combo` | ❌ | 0.30 | 0.06 | 0.020 | WP005's checkpoint, reused |
| `per_team_sigma` | ✅ | 0.20 | 0.02 | 0.008 (pop scale) | new — isolates partial pooling alone |
| `per_team_sigma_loose` | ✅ | 0.30 | 0.06 | 0.020 (pop scale) | new — pooling + WP005's other loosening combined |

Same yardstick as WP003/WP004/WP005: pooled RPS and paired-bootstrap gap to Pinnacle closing odds on the 401-match walk-forward comparison, plus the disagreement-decile overfitting check.

## Results

### Phase 1 — prior-predictive triage (run)

| | baseline | loose_combo | per_team_sigma | per_team_sigma_loose |
|---|---|---|---|---|
| attack SD (cross-team) | 0.213 | 0.323 | 0.212 | 0.322 |
| home_adv SD (cross-team) | 0.011 | 0.034 | 0.012 | 0.037 |
| top/bottom home-goals ratio | 2.35 | 3.69 | 2.39 | 3.89 |
| `sigma_att_team` within-draw max/min ratio | — | — | 71 [20, 1158] | 71 [20, 1158] |
| `sigma_att_team` pooled mean | — | — | 0.004 [0.000, 0.014] | 0.009 [0.001, 0.035] |

`per_team_sigma`'s attack spread/ratio essentially matches `baseline` (as it should — only the sigma mechanism differs, not the other knobs), and `per_team_sigma_loose` matches `loose_combo` the same way. That part behaves exactly as designed.

**Two things worth watching, not blocking:**

1. **The within-draw team-to-team sigma ratio is very heavy-tailed** — median 71×, up to 1158× at the 95th percentile. That's `raw ~ HalfNormal(1)` doing what a HalfNormal does: most draws cluster low, but the right tail is long, and across 28 independent teams the max/min ratio of a heavy-tailed distribution can blow up even though no individual team's value is unreasonable. In practice the likelihood constrains each team's actual fitted `raw` from real round-to-round data, so this is a prior-only property, not a guarantee the posterior looks like this — but it's a real risk to watch in Phase 2: a team with a short or noisy history could get an inflated sigma the data doesn't really support. Worth checking directly (Phase 2's per-team table) rather than assuming it away.
2. **The pooled mean sigma is smaller than the equivalent global value**, not the same. `per_team_sigma`'s `pop_scale=0.008` (same number `baseline` uses directly) produces a *typical* team sigma of ~0.004 — about half — because `sigma_team = raw × sigma_pop` compounds two sub-1-median HalfNormal draws instead of being one direct value. If Phase 2 shows `per_team_sigma` underperforming, this compounding (not the pooling mechanism itself) is the first thing to check — it may need a somewhat larger `pop_scale` than the direct global-sigma equivalent to land on the same typical volatility.

### Phase 2 — screening CV (run)

18 screening windows, 195 test matches, joined to Pinnacle closing odds (same harness as WP003/WP005):

| arm | n | model RPS | gap vs Pinnacle | 95% CI | gap top-25% disagree |
|---|---|---|---|---|---|
| `baseline` | 195 | 0.1916 | +0.0129 | [+0.0058, +0.0200] | +0.0455 |
| `loose_combo` | 195 | 0.1905 | +0.0118 | [+0.0049, +0.0188] | +0.0400 |
| `per_team_sigma` | 195 | 0.1923 | +0.0136 | [+0.0063, +0.0209] | +0.0450 |
| `per_team_sigma_loose` | 195 | 0.1908 | +0.0121 | [+0.0050, +0.0194] | +0.0423 |

The two clean, controlled comparisons — everything else held equal, only the sigma mechanism differing:

- `per_team_sigma` vs `baseline`: **+0.0136 vs +0.0129 — per-team pooling is very slightly *worse*, not better.** Direction, not just magnitude, is wrong.
- `per_team_sigma_loose` vs `loose_combo`: **+0.0121 vs +0.0118 — same story, adding pooling on top of the already-loosened priors doesn't help there either.**

Both differences are far smaller than the ~0.014 CI width — not distinguishable from noise — but the consistent direction (never better, marginally worse both times) is the opposite of what the mechanism was supposed to buy. `loose_combo` alone (WP005's finalist, no per-team pooling) remains the best-scoring arm across all six configs tested between WP005 and WP006 combined.

This is consistent with the Phase 1 caveat flagged before any CV ran: `per_team_sigma`'s pooled mean sigma (0.004) came out about half of the direct global value (0.008) due to the two-HalfNormal compounding — meaning most teams likely ended up *more* frozen under partial pooling, not less, with only a few outlier teams gaining real flexibility. The prediction that this compounding would be "the first thing to check if `per_team_sigma` underperforms" turned out to be exactly what happened.

### Phase 2b — does fitted sigma land on the right teams? (run)

Mean fitted `sigma_att_team`, highest first (`per_team_sigma`): **Brighton, Chelsea, Arsenal, Newcastle, Liverpool** top the list — all established clubs, not promoted sides. Promoted teams landed in the top half of fitted volatility only **5/14** times (`per_team_sigma`) and **6/14** (`per_team_sigma_loose`) — at or below the ~7/14 you'd expect from pure chance. The hypothesis ("promoted/volatile squads should show up with higher fitted sigma") is **not supported**.

Worth separating two conclusions here, because they're different: the hierarchical mechanism itself is doing something real — Brighton and Chelsea topping the list across both arms is a genuinely plausible finding (both clubs were notably volatile/managerially chaotic across these seasons), so the model is picking up real heterogeneity, not noise. It's just heterogeneity that isn't well-predicted by "was this team promoted" — more likely tangled up with managerial change, squad trading turnover, or just noisier historical form, none of which promotion status proxies for. Interesting, but it doesn't rescue the RPS result: capturing this heterogeneity doesn't improve prediction, whatever's driving it.

### Phase 3 — confirmation

**Deliberately skipped.** Phase 2 produced no finalist that beat its matched baseline — there's nothing to confirm, and Phase 3's much larger compute (full 35-window CV + held-out season) isn't justified chasing a screen that already came back flat. Right call.

## What this means, and what to do next

Count the full run of structural/feature attempts across this project: xG (null, WP002), Dixon-Coles (null, WP002/WP004), post-hoc recalibration (null, WP004), every individual global shrinkage prior (null, WP005), combined global loosening (suggestive, never significant, WP005), and now per-team sigma partial pooling, alone or combined (null — and if anything, marginally negative, WP006). **That's five or six independent structural changes in the same family — team-level Poisson-GLM, however reparameterized — producing nothing real.**

The disciplined move at this point isn't a seventh variant. It's treating that string of nulls as the actual finding: the ceiling of this model family, on this data, has very likely been found. Two honest paths from here, not a sixth pooling scheme:

1. **The lineup-quality covariate** (discussed separately) — a genuinely different *kind* of change, adding information the model currently lacks rather than restructuring how it uses the information it already has. This is the one avenue not yet tried that has a real mechanistic reason to expect a different outcome (WP003 traced part of the market's edge specifically to team-news the model never sees).
2. **Write up the ceiling finding as the project's actual conclusion.** "This model beats naive, doesn't beat the market, and here is the fairly exhaustive list of standard extensions that don't close the gap" is a complete, legitimate answer — not an unfinished one.

## Reproducing

```bash
cd work_products/wp006_per_team_sigma
jupyter lab wp006_per_team_sigma.ipynb
```

Phase 1 and Phase 2 (+2b) are complete and reflected above — checkpoints in this folder (`cv_checkpoint_<arm>.pkl`) will show as already done on re-run. Phase 3's cells are unrun by design (`FINALIST` left `None`) — see "Phase 3 — confirmation" above for why.
