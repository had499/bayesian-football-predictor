# WP008 — Starting-XI xG/xA Lineup Covariate

**Status: complete. Implemented, tested (83/83), and verified end-to-end on real fetched data. No CV comparison against baselines here by design — that's WP009.**

## What this is

Adds a lineup-quality covariate to `theta` — how much stronger/weaker today's *confirmed starting XI* is than that team's own recent normal, built from Understat per-player xG/xA (confirmed sufficient, no new data source needed — WP007). Attacking-only; the defensive on/off-differential idea from planning is a separate, later work product.

## Design (recap)

- **Additive term, not a replacement for `attack`.** `attack[team,t]` stays exactly as-is. The new term is a small correction layered on top, the same pattern `beta_xG * log(xG)` already uses.
- **A deviation from the team's own normal, not an absolute score** — `log(today_rate / team's_recent_normal_rate)` — so it can't just be redescribing "this is a good team," which `attack` already captures.
- **Starting XI only.** Substitutes' actual in-match minutes are a *consequence* of the match being predicted and would leak; excluded from the pre-match feature entirely (see `get_player_data.py`'s module docstring).
- One shared `beta_lineup ~ HalfNormal(0.1)`, same convention as `beta_xG`.

## What was built

**Prerequisite — model cleanup** (done first, on `main`): removed `use_form_decomposition` (never used in any validated WP run) and five dead `ModelConfig` fields, plus their now-orphaned code (`ar1_hyperpriors`, `match_effect_prior`). Net −134 lines. 65/65 tests passing before any WP008 code was written — the right, simpler foundation to build this on.

**`src/football_model/data/get_player_data.py`** — Understat acquisition. `discover_squads` (team+season → player IDs), `fetch_player_match_logs` (one call per *unique* player, returns their whole history — confirmed in WP007), `resolve_player_team` (attaches which side of `h_team`/`a_team` a player was actually on, resolved per `(player, season)` — not collapsed across a player's whole career, which would mis-resolve a transfer followed by a fixture against their old club). No rate limiting in `understatapi` itself, so this adds its own delay between requests.

**`src/football_model/features/lineup_features.py`** — the leakage-free aggregation, three expanding-window stages (same convention as the existing team-level rolling xG in `prepare_model_data.py`):
1. `add_rolling_player_rate` — each player's own per-90 (xG+xA), using only strictly-earlier matches, blended toward the population mean when history is thin (trust weight ramps 0→1 over 900 minutes / 10 full matches).
2. `compute_starting_xi_rate` — trust-weighted average over confirmed starters only (`position != 'Sub'` — verified empirically in WP007/WP008: starters average 85 min, subs average 25).
3. `add_lineup_deviation` — each team's own expanding average of #2, and the log-ratio of today vs. that normal (zero for a team's first tracked match — no history to compare against).

10 tests, including the critical one: a synthetic player with a deliberately huge performance spike in their *last* match, proving an earlier match's `rolling_rate` isn't influenced by it, with the exact leak-free value hand-computed and checked, not just a loose bound.

**Model wiring**: `ModelConfig.use_lineup_xg`, `ModelData.lineup_dev_home/away` (default zero — always safe to read even when the feature isn't in use), `model.py`'s `theta_home/away` gain `beta_lineup * lineup_dev`. `prepare_model_data` takes an optional `lineup_dev_table` (from `build_lineup_deviation_table`), joined by `(team_long, date)` — defaults to all-zeros when omitted, so every existing call site is unaffected.

**`predict.py` mirror — written and cross-checked before any CV wiring**, not after (the exact discipline this project learned the hard way from the missing-xG and un-evaluated-Dixon-Coles bugs): `compute_theta`/`predict_match_lambdas`/`predict_rows` all extended, with two tests proving the numpy path matches `build_model`'s actual PyTensor `lambda_home`/`lambda_away` bit-for-bit (one via `predict_match_lambdas` directly, one via `predict_rows`'s batched `ModelData`-reading path).

**`run_cv_window.py`**: extracts `beta_lineup` from the trace and applies it via `predict_rows`, same pattern as `beta_xG`/`rho_dc`. Takes an optional `lineup_dev_table` key in the shared-data pickle (`shared.get("lineup_dev_table")` — `None` for every existing WP001–006 shared-data file, which disables the covariate exactly as if it were never passed). `use_lineup_xg` itself needed **no script changes at all** — WP005's `--config-json` plumbing already validates and passes through any `ModelConfig` field.

## Verification

- 83/83 tests passing (11 new: 3 model-wiring, 2 predict.py cross-checks, 3 prepare_model_data join tests, 10 in `test_lineup_features.py` minus one overlap — see test files for the exact count per module).
- **Live end-to-end smoke test on real fetched data**, not just synthetic unit tests: pulled real per-player match logs for 4 teams / 2 seasons (4,158 rows, 129 unique players), built a real `lineup_dev_table` (304 team-match rows — first tracked match for each team correctly shows `lineup_dev=0.0`, subsequent matches show small plausible deviations as the XI varies), ran one full real CV window (window 8, `use_lineup_xg=True`) through `run_cv_window.py` end-to-end. Completed cleanly: `MAE=0.931`, `beta_lineup=0.086` (a small, sane value under its `HalfNormal(0.1)` prior — not pinned at zero, not blown up), consistent with the small effect expected from only 4/28 teams having real (non-default-zero) lineup data in this smoke test.

## What WP008 deliberately does not include

A full-coverage historical fetch (all 28 teams × 6 seasons) and any comparison against previous baselines — that's WP009, kept separate on purpose (see that README): this project's track record is a long string of nulls on structural changes, and the thing that built a mechanism shouldn't also be the one that judges it in the same sitting.

## Reproducing

```python
from football_model.data.get_player_data import get_player_match_data
from football_model.features.lineup_features import build_lineup_deviation_table

pmd = get_player_match_data(teams=[...], seasons=[...])  # Understat team-name format, underscores for spaces
table = build_lineup_deviation_table(pmd)
# attach as shared['lineup_dev_table'] = table alongside df_cv/windows, then
# scripts/run_cv_window.py --config-json '{"use_lineup_xg": true}' picks it up automatically
```

WP009 will do the full-coverage fetch (all teams/seasons already in WP001's `cv_shared_data.pkl`) and the actual comparison.
