# WP011 — Multi-League Hierarchical Pooling

**Status: implemented, unit-tested (25 new tests, 108/108 total passing), and verified end-to-end on real fetched EPL + Bundesliga + La_Liga data — including a real concurrent 3-window run that caught and fixed one genuine bug (see Compute below). Full 5-league walk-forward CV (the actual WP009-style validation) not run — that's the next step, and per this project's standing division of labor, the real 18-window screen and 35-window confirmation are yours to run via `wp011_multileague_hierarchy.ipynb`, same as every prior WP.**

## Why

WP005 (loosen priors), WP006 (per-team sigma), WP009 (new lineup information), WP010 (locate the gap) all converged on the same conclusion: no structural change *within* the current EPL-only architecture recovers the resolution gap to Pinnacle (WP003/WP004). This WP is deliberately different in kind, not another retuning: it adds more *independent teams and seasons* to the hierarchy — pooling across Europe's top leagues — so the model's shared hyperparameters (how much team strength typically varies, how big home advantage typically is) are estimated from far more data than ~20 EPL teams over 6 seasons can support. See the design discussion in this WP's conversation history for the full reasoning; recapped below.

**Scope discipline**: this WP only touches the base hierarchical model (`attack`/`defence`/`home_adv`/AR1). It does **not** extend the lineup-xG covariate (WP008/009) to other leagues — that's a separate, later WP if this one pays off.

## Architecture

Three different treatments for three different parameters — not a uniform "add a league dimension everywhere":

- **`home_adv` → extend the existing per-team pooling by one level.** Today: team → global (`home_adv[team] = mu + sd * raw[team]`, one shared `mu`/`sd` for every team — see `priors.py::home_advantage_prior`). New: team → league → global — `mu_league[l] ~ Normal(mu_global, sd_between_league)`, each team's `home_adv` pools toward its own league's `mu_league`/`sd_league` instead of one shared value. Best-motivated single change: home advantage varying by country is well-established in the literature, and directly targets WP010's B5 finding (the model's home-goal expectation running low).
- **`sigma_att`/`sigma_def` → stay ONE shared global scalar, not split by league.** The goal is a better-estimated single number, not more granularity — WP006 already showed extra granularity (per-team) didn't pay off, and splitting this by league risks the same failure mode for a similar reason. Same global scalar, fed by a much larger hierarchy of teams.
- **`rho`** (AR1 persistence) — stays global. No evidence it needs to vary by league; lowest priority to touch.
- **`beta_xG`, `rho_dc`** — stay global, small/low-risk parameters.
- Each team's own `attack[team,t]`/`defence[team,t]` trajectory is still informed only by that team's own matches, exactly as today — cross-league pooling sharpens the shared assumptions everyone gets shrunk toward, it doesn't give any team new information about itself.
- No cross-league matches are modeled (no Champions/Europa League fixtures) — each of the 5 leagues is a fully separate domestic competition; teams never play across leagues in this dataset.

**Implementation note — disjoint-but-separate time axes, not a shared padded one.** Each league keeps its own `n_time`/`n_teams`-shaped AR(1) recursion (`ar1_team_process` called once per league, looped in Python, each call producing its own `attack_<league>`/`defence_<league>` Deterministic), sharing only the scalar `rho_att`/`rho_def`/`sigma_att`/`sigma_def` priors and the league-pooled `home_adv_<league>` across calls. Considered and rejected: concatenating all leagues onto one shared/disjoint global time index (so every team has a slot at every league's time step, frozen via `active_mask` while "playing" in another league). That would work but wastes compute quadratically — `n_time × n_teams` for the shared z-noise matrix and the scan grows with the *product* of total teams and total time steps, not the sum, even though a team is only ever active during its own league's own stretch. Per-league independent calls are linear in the actual amount of data (`Σ_l n_time_l × n_teams_l`) and match the physical reality (leagues never share a match) exactly — implemented in `build_multileague_model` (`src/football_model/model/model.py`).

Match-level training data is joined into one Poisson likelihood per league (`goals_home_<league>`/`goals_away_<league>`), with `use_dixon_coles`'s `rho_dc` (if enabled) applied as ONE shared correction over every league's matches concatenated together — a single learned correlation parameter, not one per league, matching how `beta_xG` also stays shared.

## Data

**`get_understat_data` (`src/football_model/data/get_data.py`) already supports this** — its `leagues` parameter defaults to `['EPL', 'RFPL', 'Bundesliga', 'La_Liga', 'Serie_A', 'Ligue_1']`. Per your instruction: **top 5**, dropping `RFPL` (not top-5, and Russian clubs have been excluded from UEFA competition since 2022). Same tooling as WP001/008/009, no new library.

**"Matching against bets" turned out to be scoped to zero new work.** The odds crosswalk is only ever needed for evaluation, and evaluation stays EPL-only — WP003's `CODE_TO_FD` crosswalk is reused untouched. The other 4 leagues are pure training data: team identity only needs to be consistent *within Understat's own data*, a single source, never cross-joined to anything for those leagues in this WP.

**Encoding — checked empirically, not assumed.** Fetched real 2023-season data for all 4 non-EPL top-5 leagues and inspected every team name directly: Understat's own naming is already fully anglicized and plain ASCII across the board (`"Bayern Munich"` not `"Bayern München"`, `"Alaves"` not `"Alavés"`, `"FC Cologne"` not `"1. FC Köln"`) — the Unicode-normalization risk flagged in planning doesn't materialize in practice. Added a defensive NFC-normalization pass in `get_data.py` anyway (nearly free, and Understat's naming could change) — `_normalize_team_names`, covered by `tests/test_get_data.py` with a synthetic NFC/NFD-mismatch case proving it actually resolves the failure mode, not just a no-op.

**Round/calendar computation needed no changes at all — turned out simpler than planned.** The original concern was that `prepare_model_data` assumes one continuous single-league calendar. In practice, each league is prepared **completely independently** — its own `prepare_model_data(league_df, max_round=...)` call, own local team-index space, own time axis starting at 0 — never concatenated into one shared dataframe. No `league` column needed anywhere; `add_rounds_to_data`/`add_match_ids`/`prepare_model_data` are all called once per league, unmodified. See Architecture below for why this is also the computationally right choice, not just the simplest one.

## League inclusion — my call, exercised on real data, not presumed upfront

Start with all 5. A league gets **dropped** (falling back to 4, 3, or 2 leagues rather than blocking the whole WP) only if, after reasonable effort, it fails a concrete bar:
- Understat data doesn't fetch cleanly for that league/season (matches WP007/WP008's existing "check squad discovery and match logs actually return sane data" standard).
- Team-name encoding/consistency issues that don't resolve with a small, explicit crosswalk (same spirit as WP003's "only 6 names differ from the obvious" — a handful of manual overrides is fine, systematic unresolvable corruption is not).
- Suspiciously incomplete season coverage (missing rounds, big date gaps) that would distort that league's contribution to the shared hyperparameters more than it helps.

This is a data-quality bar, not a subjective one — reported plainly in the results (which leagues made it in, why any didn't) rather than quietly worked around.

## What was built

- **`src/football_model/model/priors.py`** — `league_home_advantage_prior(league_names, league_n_teams, mu_center, mu_scale, between_league_scale, sd_scale)`: the new team → league → global hierarchy, non-centered throughout (same funnel-avoidance pattern as `ar1_hierarchical_sigma`/`home_advantage_prior`). `sd` (within-league team spread) stays one shared scalar across every league, deliberately — only `mu` gets a league level, matching the evidence-backed scope from planning.
- **`src/football_model/model/model.py`** — `build_multileague_model(leagues: dict[str, ModelData], config)`: the joint model. Kept **completely separate** from `build_model`, not refactored to share code with it — `build_model` is what every past WP's real CV numbers were produced by, and the existing-behaviour regression guarantee is strongest when that function's source is simply untouched, not parameterized to also handle a new case. Raises `NotImplementedError` for `use_lineup_xg`/`use_per_team_sigma`/`use_opponent_adjusted_xG` (WP011 scope: base hierarchical model only).
- **`src/football_model/types/model_data.py`** — one new `ModelConfig` field, `home_mu_league_sd` (between-league spread of the home-advantage mean); unused by the existing single-league path.
- **`src/football_model/data/prepare_model_data.py`** — `max_round_for_cutoff_date(df, cutoff_date)` and `prepare_multileague_data(engineered_dfs, cutoff_date, ...)`: resolve a shared real-world cutoff *date* to each league's own `max_round`, so leagues with different season lengths/calendars stay leakage-safe against the same point in time, then call the **unmodified** `prepare_model_data` once per league.
- **`src/football_model/data/get_data.py`** — `_normalize_team_names` (Unicode defence, see Data above).
- **`scripts/run_cv_window.py`** — `run_window_multileague(dfs_by_league, eval_league, window, ...)`: same shape and same result-dict schema as the existing `run_window`, kept fully separate from it for the same regression-safety reason `build_multileague_model` is kept separate from `build_model`. `main()` dispatches to it automatically when the shared-data pickle carries a `dfs_by_league` key (absent in every WP001–010 pickle, so their behaviour is provably unaffected — `.get()` returns `None` and the original single-league path runs exactly as before). Also: `run_windows_concurrent` (see "Compute" below).
- **`prepare_multileague_data`'s graceful league-skipping** — a real bug this WP's own live testing caught (see Compute/Verification below): a non-eval league fetched with fewer seasons than the eval league can have zero matches before an early window's training cutoff; `prepare_multileague_data` now skips that league for that window (printing why) instead of crashing.

## Compute — concurrency, trimming, screening

Built in response to a direct question about compute cost, not bolted on after the fact:

- **`run_windows_concurrent`** (`scripts/run_cv_window.py`) replaces the sequential `for w in windows: subprocess.run(...)` loop every prior WP's notebook used. Windows are already isolated subprocesses by design (nothing shared across them) — the one real hazard is `main()`'s checkpoint read-modify-write racing if two subprocesses point at the same checkpoint file at once (a classic lost-update: both read the same "before," both write their own "after," one silently overwrites the other). Solved by giving each concurrent window its own throwaway checkpoint file and merging into the real one only from the main thread as each finishes (`concurrent.futures.as_completed`), never inside a worker. 8 dedicated tests in `tests/test_run_cv_window.py` against a fast fake worker script (`tests/fake_cv_worker.py`, no PyMC) prove: results are complete and correct, it's actually faster than sequential (timed, not assumed), already-done windows are skipped without re-dispatching, 12 windows at 6-way concurrency produce zero lost updates, and a failing window doesn't block the others or crash the batch.
- **Trimming**: no new code needed — it's a fetch-time choice (fewer `years` passed to `get_understat_data` for the 4 non-EPL leagues than EPL's own 6 seasons). The statistical rationale: pooling's benefit is mainly "more independent teams," not "more seasons per team," so trimming keeps most of the benefit for a fraction of the compute.
- **Screening**: the notebook (`wp011_multileague_hierarchy.ipynb`) follows the exact same 18-window-then-35 pattern as WP005/WP006/WP009, via `SCREEN_WINDOWS = list(range(1, len(windows) + 1, 2))`.

**A real bug found by combining all three**, not by unit tests alone: ran 3 real windows (5, 10, 15) concurrently against real fetched EPL + Bundesliga + La_Liga data, Bundesliga/La_Liga trimmed to 2023–2024 only. Windows 5 and 10 (early in EPL's 2020-21 season) crashed — their training cutoff date fell *before* Bundesliga/La_Liga's trimmed data even starts, and `prepare_multileague_data` didn't handle "this league has zero eligible matches yet" gracefully (an empty dataframe produced a `NaN` time-axis length, not a clean skip). Fixed by having `prepare_multileague_data` skip a league for a given window when it has no matches before that window's cutoff — the league-level analogue of `active_mask` already handling a not-yet-promoted team within one league. Covered by a regression test (`test_prepare_multileague_data_skips_league_with_no_matches_before_cutoff`) before moving on. **The practical implication for real runs: trimming non-EPL history means early screening/CV windows will train on fewer leagues than later ones, not fail — worth knowing going in, not a bug to work around.**

Re-ran the same 3 windows after the fix: **all 3 completed successfully.**

| window | leagues actually used | MAE | ll_improvement |
|---|---|---|---|
| 5  (train rounds 1–56)  | EPL only — Bundesliga/La_Liga correctly skipped, cutoff 2022-02-20 predates their trimmed data | 1.058 | +4.34 |
| 10 (train rounds 1–81)  | EPL only — same reason, cutoff 2022-11-13 | 0.927 | −0.40 |
| 15 (train rounds 1–106) | all 3 — Bundesliga/La_Liga's 2023 data has started by this cutoff | 0.737 | +2.64 |

Windows 5 and 10 print the skip message and fall back to an EPL-only fit automatically — no crash, no manual intervention. Window 10's negative `ll_improvement` isn't a red flag on its own (individual windows are noisy; WP001's own baseline has non-monotonic per-window results too) — it's flagged here for completeness, not because it's unusual.

**Concurrency, measured on this real run**: 3 windows at `max_workers=3` completed in **160.9s** wall time. Summing each window's own individual completion time from the log (windows 5/10 ≈85-90s each once running solo-ish, window 15 ≈123s) gives a rough sequential estimate of ~300s — call it **~1.9x faster** than one-at-a-time, on a 12-core machine running 3 full NUTS jobs at once. Short of a perfect 3x, as expected — 3 concurrent JAX/NUTS processes contend for the same cores — but a real, measured win, not a hoped-for one.

## Compute — a second issue: partial-coverage windows dilute the aggregate result

Trimming means non-EPL leagues only have data from a certain point onward — so *early* windows in the 35-window walk-forward (whichever ones have a training cutoff before the trimmed leagues' history starts) run EPL-only, correctly, but contribute **zero** multi-league pooling benefit. With `TRIM_YEARS_NON_EPL = ['2023', '2024', '2025']`, that's roughly the first 40% of the 35 windows (window 15 was the first of the three tested to have full coverage; windows 5 and 10 did not). Left unhandled, the pooled gap-to-Pinnacle across all windows blends "no possible effect" windows with "full joint-model" windows, diluting any real effect toward the baseline — a real statistical cost of trimming, not just a mechanical one.

Fixed two ways:
1. **`run_window_multileague`'s result now records `leagues_used` (what was actually trained on, after `prepare_multileague_data`'s skip logic) separately from `leagues_available` (what was passed in)** — previously it only recorded the latter, which would have made this impossible to detect after the fact. Verified correct on real data: window 5 → `leagues_available=['Bundesliga','EPL'], leagues_used=['EPL']`; window 15 → both lists equal.
2. **The notebook's Phase 2 analysis gained a "Phase 2b" cell** (`leagues_coverage_split`/`filter_ckpt_to_windows`) that splits windows into full-coverage vs. partial-coverage and reports the gap-to-Pinnacle separately for each — plus the same split applied to `baseline`/`lineup_loose_combo`'s results on the *same* window numbers, so the full-coverage comparison is apples-to-apples against arms trimming never affected. This is the real test of the hypothesis; the all-windows pooled number (kept too, for comparability with every prior WP's methodology) is the conservative, diluted one.

## Verification

- **108/108 tests passing** (25 new: 4 `league_home_advantage_prior` tests, 6 `build_multileague_model` tests including a real tiny NUTS run, 1 `predict_rows`-vs-multileague-model cross-check, 5 `prepare_multileague_data`/`max_round_for_cutoff_date` tests — including the graceful-league-skip regression test added after the real concurrent run found the bug — 3 `get_data` encoding tests, 8 `run_windows_concurrent` tests against a fast fake worker script (no PyMC), plus a second synthetic league fixture added to `tests/conftest.py` with a deliberately different team count and round count from the existing one, so no test could accidentally pass just because both leagues happened to be the same shape).
- **The key cross-check**: `test_predict_rows_matches_epl_slice_of_multileague_model` proves evaluation needs **zero new `predict.py` code** — a league's own `attack_<name>`/`defence_<name>`/`home_adv_<name>` posterior slice, read via the ModelData `prepare_model_data` already returns for that league, produces bit-for-bit identical `lambda_home`/`lambda_away` through the exact same `predict_rows` this project has used since WP001. A multi-league-trained EPL slice is, at prediction time, indistinguishable from a model that was only ever trained on EPL.
- **Leakage/independence proven structurally, not statistically**: `test_build_multileague_model_league_independence_graph_structure` walks the actual PyTensor computation graph (`pytensor.graph.traversal.ancestors`) and asserts a league's `attack`/`defence` Deterministic has zero ancestor dependency on any other league's nodes — a deterministic proof of the core architectural claim, not one that could pass by sampling luck.
- **Regression safety**: full existing suite (83 tests before this WP) untouched and still green; `build_model`/`run_window` source code is byte-for-byte unchanged.
- **Live end-to-end smoke test on real fetched data** (not just synthetic fixtures): real EPL + Bundesliga 2023-season data (760 + 612 match-observation rows), one real window (`train` rounds 1–25, `test` rounds 26–27, 465 combined training matches, full `2000+2000`-draw × 3-chain NUTS via `run_window_multileague` exactly as the real CV harness will call it) — completed in 38.8s with no divergence warnings, `ll_improvement=+4.41` over naive (model beats naive, same sanity bar WP001 used), `rho_dc=0.0104` (small, sane), `lambda_home`/`lambda_away` both in plausible ranges (`[0.76, 2.48]`, `[0.87, 2.55]`) across all 20 test matches. Confirms fetch → independent per-league prep → joint training → EPL-slice extraction → existing unmodified `predict_rows` works correctly end-to-end on real data, not just synthetic fixtures.

## Baseline & benchmark

Unchanged from every WP since WP001 — non-negotiable for comparability:
- **WP001 baseline** (tight priors, EPL-only) — the anchor.
- **WP009 `lineup_loose_combo`** (current best-available) — the actual bar to beat.
- **New arm**: multi-league pooled architecture, EPL-only priors otherwise left at WP001 defaults (not `loose_combo`'s manually-loosened values — the point of this WP is to let the data set `sigma_att`/`sigma_def`/`home_adv`'s spread, not to guess a looser number by hand; `loose_combo` on top is a natural follow-up arm if the base pooled version shows promise).

All arms evaluated **only** on the same 401 EPL matches / 361 Pinnacle-covered subset, same 35 walk-forward windows, same RPS + paired bootstrap — training data volume changes, the yardstick does not.

## What we'd expect to see

Recapped from the design discussion: hierarchical scale-parameters estimated from a small number of groups (~20-25 active EPL teams) are known to be imprecisely/conservatively estimated. Pooling ~100 teams across 5 leagues gives a much larger effective sample for `sigma_att`/`sigma_def` and `home_adv`'s league-level `mu`, which could reduce the over-shrinkage WP003/WP004/WP010 diagnosed — for a specific, data-driven reason, not a guessed adjustment like `loose_combo`.

Calibrated expectation: moderate chance of a small, real, mechanistically explainable improvement in calibration; low chance of closing most of the remaining gap to Pinnacle, since WP010 already showed that gap is diffuse and plausibly dominated by information (team news, referee, market money) no historical dataset carries regardless of how many leagues it spans. Success here looks like "closes some more of the gap, measurably, for an explainable reason" — not "becomes competitive with the market."

## Practical cost

The smoke test's 2-league, single-season window (465 training matches, ~38 teams total) sampled in 38.8s — a genuinely useful reference point, not just a guess: a full 5-league, 6-season run has roughly 5x the teams and a similar multiple of total matches, so expect each of the 35 walk-forward windows to take meaningfully longer than WP001–010's EPL-only windows (which this same hardware/sampler handled in comparable per-window time to what was just seen for 2 leagues). Data acquisition remains low-friction — same tooling, same discipline as WP007/WP008.

## What's left — not part of this implementation pass

- **The actual WP009-style validation**: full 5-league, 6-season fetch, a `run_window_multileague`-driven 35-window walk-forward CV, and the same paired-bootstrap-vs-Pinnacle comparison against WP001's baseline and WP009's `lineup_loose_combo` this WP was scoped to produce. This is the real compute — yours to run, per this project's standing division of labor (I do data acquisition and all testing/plumbing; you run the walk-forward training).
- League-inclusion decisions (see "League inclusion" above) get made once real 5-league data is actually fetched and looked at, not presumed here.

## Reproducing

```python
from football_model.data.get_data import get_understat_data
from football_model.features.add_metadata import add_rounds_to_data, add_match_ids, add_home_away_goals_xg
from football_model.data.prepare_model_data import prepare_multileague_data
from football_model.model.model import build_multileague_model
from football_model.types.model_data import ModelConfig
import pymc as pm

dfs_by_league = {}
for league in ["EPL", "Bundesliga", "La_Liga", "Serie_A", "Ligue_1"]:
    raw = get_understat_data(years=["2023"], leagues=[league])
    dfs_by_league[league] = add_home_away_goals_xg(add_match_ids(add_rounds_to_data(raw)))

leagues = prepare_multileague_data(dfs_by_league, cutoff_date="2024-03-01")
model = build_multileague_model(leagues, ModelConfig(clip_theta=5.0, center_team_strength=False, use_xG=True))
with model:
    trace = pm.sample(2000, tune=2000, chains=3, nuts_sampler="numpyro")
# trace.posterior["attack_EPL"], ["home_adv_EPL"], etc. -- read exactly like
# the single-league model's "attack"/"home_adv", per league.
```

For the actual walk-forward CV: build a shared-data pickle with `{"dfs_by_league": {...}, "eval_league": "EPL", "windows": [...]}` (same `windows` shape WP001 already uses) and call `scripts/run_cv_window.py` exactly as every prior WP has — `main()` auto-detects the `dfs_by_league` key and routes to `run_window_multileague`.
