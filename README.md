# Bayesian Football Predictor

A Bayesian hierarchical model for predicting English Premier League football match outcomes using PyMC and FastAPI.

## Overview

This project implements a time-varying Bayesian model that learns team strengths from historical match data and generates probabilistic predictions for upcoming fixtures. The model accounts for:
- Dynamic team attack and defense strengths that evolve over time
- Team-specific home advantage
- Expected goals (xG) as a baseline prior
- Full uncertainty quantification through posterior distributions

## Understanding the Metrics

This project uses several statistics to judge whether the model is any good. Numbers referencing these show up throughout this README — here's what each one actually means, without the maths.

**MAE (Mean Absolute Error)** — "On average, how many goals off was the prediction?" If the model says a team will score 1.3 goals and they score 2, that's an error of 0.7; average that across every match. Lower is better. Simple, but it only judges goal counts — it says nothing about whether the model correctly called who wins.

**Log-Likelihood (LL)** — "How probable did the model consider what actually happened?" Every match, the model assigns a probability to the real result; log-likelihood adds the logs of those probabilities across all matches.
- It's *always* negative — that's normal, not a bad sign, since log() of any probability (always ≤1) is ≤0.
- Closer to zero = better (`-30` beats `-50`).
- The raw number means almost nothing alone — only *comparisons* matter (model's LL vs. a baseline's LL, on the *same* matches).
- Rule of thumb for "how much better": `exp(difference)` gives a "Bayes Factor." A value of ~1-3 isn't meaningfully different from the baseline — you want 20+ before treating an improvement as real. Small totals on small samples can also just be measurement artifacts (see Design Choice #6's history below) — treat any single LL comparison cautiously.

**RPS (Ranked Probability Score)** — the metric this project trusts most for judging match predictions. It scores how good the model's win/draw/loss *probabilities* were, and — unlike LL on goal counts — it understands outcomes are *ordered*: predicting "draw" when the away team actually won is a smaller mistake than predicting "home win" when the away team won. Lower is better; 0 is a perfect prediction, ~0.2 is roughly what a solid football forecasting model achieves.

**Calibration ("is 70% actually 70%?")** — separate from accuracy. Take every match where the model said "70% chance of a home win" — in a well-calibrated model, about 70 of every 100 such matches should actually end in a home win. If it's really 90, the model is under-confident; if it's 50, it's over-confident and its probabilities can't be trusted at face value. This matters a lot for anything betting-related: comparing your probability to a bookmaker's only makes sense if your probability is honest.

**Bootstrap confidence interval — "is this improvement real, or just luck?"** — any single test on a handful of matches can look good or bad purely by chance. A bootstrap confidence interval resamples the results many times to see how much the "improvement" number actually wanders around. If the resulting range **excludes zero**, the improvement is unlikely to be a fluke at this sample size. If the range includes zero, the model isn't yet distinguishable from doing nothing extra.

**Putting it together**: the project's current validated result is an RPS improvement of 0.036 over a naive baseline, 95% confidence interval [0.026, 0.046] — a real, if modest, edge, measured across 401 out-of-sample matches spanning five Premier League seasons (see "Out-of-Sample Testing" below). That is *not* the same as "beats bookmakers" — it's the honest answer to "is this better than just guessing the league average," measured properly for the first time in this project's history.

## Architecture

### Core Components

```
├── src/football_model/          # Core modeling package
│   ├── data/                    # Data fetching and preparation
│   ├── features/                # Feature engineering
│   ├── model/                   # PyMC model definition
│   └── types/                   # Data structures
├── services/predictor/          # FastAPI web service
│   ├── predictor.py            # API endpoints
│   ├── Dockerfile              # Container definition
│   └── docker-compose.yml      # Orchestration
└── notebooks/                   # Exploratory analysis
```

### Data Flow

1. **Data Ingestion**: Fetch match data from Understat API (goals, xG, dates, teams)
2. **Feature Engineering**: Add rounds, match IDs, home/away metadata
3. **Model Preparation**: Convert to team indices, temporal structure
4. **Training**: Sample posterior using NUTS (No U-Turn Sampler)
5. **Prediction**: Generate probability distributions for upcoming matches
6. **API Serving**: FastAPI endpoints expose predictions as JSON

## Model Design

### Hierarchical Structure

The model uses a **time-varying hierarchical Bayesian approach**:

```python
# Team strengths evolve over time with AR(1) process
attack[t, team] ~ Normal(ρ_att * attack[t-1, team], σ_att)
defense[t, team] ~ Normal(ρ_def * defense[t-1, team], σ_def)

# Expected goals (Poisson likelihood)
λ_home = exp(attack[t, home] - defense[t, away] + home_adv[home])
λ_away = exp(attack[t, away] - defense[t, home])

goals_home ~ Poisson(λ_home)
goals_away ~ Poisson(λ_away)
```

### Key Design Choices

#### 1. **Time-Varying Parameters**
- **Why**: Team strength changes throughout the season (injuries, form, transfers)
- **How**: Autoregressive process with persistence parameter ρ (typically ~0.95)
- **Benefit**: Model adapts to recent performance while maintaining stability
- **Validation**: Rolling-window cross-validation (35 windows, 2020-21 through 2025-26 seasons) gives **MAE: 0.913 ± 0.153**, and a pooled **RPS improvement of 0.036 over naive (95% CI [0.026, 0.046], excludes zero)** across 401 out-of-sample matches — see "Out-of-Sample Testing" below for the full picture, and "Understanding the Metrics" above if these terms are new.

#### 2. **Team-Specific Home Advantage**
- **Why**: Home advantage varies by team (stadium, fans, travel)
- **How**: Learned parameter `home_adv[team]` added to home team's log-rate
- **Benefit**: Captures differential home effects (e.g., intense atmospheres)

#### 3. **Poisson Likelihood**
- **Why**: Goals are discrete, non-negative, and relatively rare
- **How**: `goals ~ Poisson(λ)` where λ is the expected goals rate
- **Alternative Considered**: Negative Binomial (allows overdispersion)
- **Experimental Result**: Tested both in `modelling.ipynb`. Poisson matched NB performance with faster convergence. The additional dispersion parameter α didn't improve out-of-sample log-likelihood significantly.

#### 4. **Hierarchical Priors**
- **Why**: Partial pooling shares information across teams
- **How**: Hyperpriors on σ_att, σ_def, ρ_att, ρ_def
- **Benefit**: Stabilises estimates, especially for newly promoted teams

#### 5. **Multi-Season Handling (relegation/promotion)**
- **Why**: Training on more than one season means the team roster changes every year — relegated teams stop playing, promoted teams appear mid-timeline. A plain per-team AR(1) over the full `(n_time, n_teams)` grid would let a relegated team's latent strength keep randomly walking on pure innovation noise with no data to anchor it, and would let that noise contaminate the centering constraint used to keep team strengths identifiable.
- **How**:
  - `active_mask` (`n_time × n_teams`) marks which teams are actually in the league at each round, built in `prepare_model_data`. The AR(1) step (`ar1_team_process` in `model/components.py`) is gated by this mask: a team's attack/defence is **frozen at its last active value** while it's out of the league, rather than drifting. Centering (`centered_over_teams` / the soft-centering `Potential`s) also uses this mask so relegated/not-yet-promoted teams never dilute the "average current team = 0" reference point.
  - `season_start_mask` flags each season's opening `season_start_window` rounds (default 5). Innovation variance (σ_att/σ_def) is inflated by `season_start_sigma_mult` (default 3.0) during that window for **every** team, not just returners — squads turn over in the transfer window regardless of whether a team was relegated, so this captures faster-than-usual strength movement at the start of every season, and it naturally covers a returning team's first matches back since promotion/relegation only happens between seasons.
- **Benefit**: A relegated team's strength holds at "last known form" instead of reverting to noise, and reconnects cleanly to the league-average reference point the moment they're promoted back — with no special-casing needed for teams that bounce between divisions repeatedly.

#### 6. **One Observation Per Match, Not Two**
- **Why**: Understat data gives one row per team per match (a home-perspective row and an away-perspective row). `goals_home`/`goals_away` are row-relative (this row's team vs. opponent), and `theta_home`/`theta_away` in `model.py` mirror that — `home_adv` is only added to `theta_home`, gated by that row's own `is_home` flag. Training on both rows fed every match's outcome into the Poisson likelihood twice: once correctly (home team's goals with their `home_adv` bonus) and once through the away-perspective row's `theta_away`, which computes the *same* home team's goals with no `home_adv` term at all — a duplicate under a mismatched mean function, biasing `home_adv` toward zero and overstating confidence everywhere else (pseudo-replication).
- **How**: `prepare_model_data` now keeps only the home-perspective row (`is_home == 1`) when building the final observation arrays, so each match contributes exactly one `(goals_home, goals_away)` pair. Team-level rolling stats that need full home+away history (the xG baseline, the active-team mask) are still computed from both perspectives *before* that filter is applied, so a team's away form still counts toward its own history — only the final Poisson likelihood is deduplicated.
- **Benefit**: `n_obs` halves to exactly `n_matches`, `home_adv` estimates should shift upward (no longer attenuated), and posterior uncertainty should widen to reflect the real amount of independent evidence.

#### 7. **Dixon-Coles Low-Score Correlation Correction (optional, off by default)**
- **Why**: Modelling home and away goals as independent Poissons misprices low scorelines — real matches show a small negative correlation (e.g. a leading team sitting back late on), so independent Poisson under-predicts 0-0/1-1 draws relative to what actually happens.
- **How**: A single learned parameter `rho_dc` nudges just the four low-scoring cells (0-0, 1-0, 0-1, 1-1) via a correction factor τ(x,y), added as a `pm.Potential` — mathematically equivalent to multiplying the joint likelihood by τ, without touching the existing independent Poisson likelihoods. Enable with `config.use_dixon_coles=True`.
- **Status**: Implemented and tested (`use_dixon_coles: bool = False` in `ModelConfig`), not yet wired into the deployed `/predict` endpoint — the serving-side Monte Carlo scoreline sampling in `predictor.py` still draws home/away goals independently, so even with `rho_dc` trained, live predictions won't reflect the correlation until that sampling step is updated too.


### Model Parameters

| Parameter | Description | Prior/Constraint | Learned Value |
|-----------|-------------|------------------|---------------|
| `attack[t, team]` | Team's attacking strength at time t | AR(1), σ ~ HalfNormal(0.1) | Varies by team (±0.5 range) |
| `defense[t, team]` | Team's defensive weakness at time t | AR(1), σ ~ HalfNormal(0.1) | Varies by team (±0.5 range) |
| `home_adv[team]` | Team-specific home advantage | Normal(0, 0.5) | Typically 0.1-0.3 (10-35% boost) |
| `ρ_att`, `ρ_def` | Persistence of team strength | Beta(9, 1) → ~0.9 | Learned: ~0.92-0.95 (high persistence) |
| `σ_att`, `σ_def` | Innovation in team strength | HalfNormal(0.1) | Learned: ~0.05-0.08 (gradual change) |
| `season_start_window` | Rounds at each season's start treated as high-uncertainty | Config, default 5 | — |
| `season_start_sigma_mult` | σ_att/σ_def multiplier during that window | Config, default 3.0 | — |

### Prediction Algorithm

When `/predict` is called:

1. **Load posterior samples** (20,000 draws from MCMC)
2. **Subsample for efficiency** (1,000 draws to reduce CPU usage)
3. **For each match and each sample:**
   - Calculate `λ_home = exp(attack_home - defense_away + home_adv)`
   - Calculate `λ_away = exp(attack_away - defense_home)`
   - Sample goals: `goals ~ Poisson(λ)`
4. **Aggregate across samples:**
   - Outcome probabilities: `P(home win) = mean(goals_home > goals_away)`
   - Scoreline distribution: Count frequency of each (goals_home, goals_away)
   - Expected goals: Mean of λ samples

**Why Monte Carlo?** Analytically computing outcome probabilities from independent Poisson distributions is complex for all scorelines. Sampling naturally handles the full joint distribution.


### Model Assumptions
1. **Goals are Poisson**: Validated by comparing variance to mean (approximately equal)
2. **Independence**: Assumes goals don't strongly affect each other — a reasonable approximation, though not exact (see the Dixon-Coles correction, Design Choice #7, which exists specifically to patch the low-score cells where this assumption is weakest)
3. **Stationary within-season**: Team strength changes smoothly, no sudden jumps

### Out-of-Sample Testing

Rolling-window cross-validation: train on rounds 1-N, predict round N+1, repeat for many different N. The current run spans **35 windows across the 2020-21 through 2025-26 EPL seasons** (5 seasons, 5 relegation/promotion boundaries), run with `use_xG=True` and `use_dixon_coles=True`:

| Metric | Value |
|---|---|
| MAE | 0.913 ± 0.153 goals/match |
| Pooled RPS (401 test matches) | Model: 0.198 vs. Naive: 0.234 |
| RPS improvement over naive | **0.036, 95% CI [0.026, 0.046]** — excludes zero |
| Windows individually beating naive on LL | 29/35 (83%) |

The RPS result is the first properly statistics-backed validation result in this project — earlier headline numbers (see Key Findings below) were measured on samples too small to distinguish from noise, or under a likelihood bug that has since been fixed.

**Caveat**: this run has `use_xG` and `use_dixon_coles` enabled *together* — the improvement hasn't been decomposed to say how much each feature contributes individually. An ablation (xG-only, Dixon-Coles-only, neither) would answer that; not yet done.

**Also found**: pooled calibration checking (see `modelling.ipynb`) shows the model is well-calibrated for close-to-even predictions but *under-confident* specifically when it already favors a home win — e.g. a "60-80% predicted" bucket saw an 86% actual home-win rate. Not yet root-caused; a leading hypothesis is over-shrinkage of `home_adv` or attack/defence for stand-out teams.


## Experimental Extensions

Several model variants were tested in `modelling.ipynb`. Some (xG, Dixon-Coles) are now part of the validated notebook config; none of them are wired into the live `predictor.py` service yet — its `TrainRequest` doesn't expose `use_xG`/`use_dixon_coles` as options at all, so the deployed API always trains the plain config regardless of what the notebook has validated.

### xG as Feature — Re-tested After a Likelihood Fix, Now Enabled
The model has built-in support for using expected goals (xG) as a feature:
```python
λ = exp(β_xG * log(xG) + attack - defense + home_adv)
```

**Status**: `config.use_xG=True` — **enabled**, part of the current validated CV config (see "Out-of-Sample Testing" above).

**History**: an earlier round of testing found "marginal improvement, not worth the complexity" and shipped with it disabled. That finding was measured under a likelihood bug that fed every match's outcome into training twice (Design Choice #6) — a bug that specifically corrupts before/after comparisons like this one. After fixing it, xG was re-enabled; it's now part of the config that produced the statistically significant RPS result above.

**Open question**: the current result has xG enabled *together* with Dixon-Coles — it isn't yet known how much of the improvement is attributable to xG specifically. Needs an ablation.

**Alternative, still untested post-fix**: Opponent-adjusted xG (trust xG more vs strong defenses) — previously found minimal improvement, but under the same bug, so that finding shouldn't be trusted either.

### Form Decomposition Model 
Separates team strength into:
- **Ability**: Long-term stable strength (ρ ≈ 0.98, σ ≈ 0.003)
- **Form**: Short-term fluctuations (ρ ≈ 0.89, σ ≈ 0.015)

```python
attack[t] = ability_attack[t] + form_attack[t]
defense[t] = ability_defense[t] + form_defense[t]
```

**Result**: Minimal improvement over single time-varying parameter (~0.02 MAE reduction). Added complexity not justified. May revisit if form signals (injuries, managerial changes) become available.


## Testing

A pytest suite under `tests/` covers feature engineering (`add_rounds_to_data`, `add_match_ids`, `add_home_away_goals_xg`), multi-season data prep (`active_mask`/`season_start_mask` construction, no-leakage on `max_round`), the AR(1) gating/centering logic (relegation freeze, season-start variance inflation, masked centering), `build_model` smoke tests (prior-predictive and a tiny real NUTS run, both branches of `use_form_decomposition`), and the FastAPI service (`/status`, `/predict`, `/gameweeks`, rate limiting) against a temp `DATA_DIR` so it never touches real trained artifacts.

```bash
pip install -e .
pip install -r requirements-dev.txt
pytest
```

## Future Enhancements

### Model Improvements
- [x] **xG feature**: re-tested after fixing a likelihood bug that had corrupted the original "marginal improvement" finding — now enabled (`use_xG=True`), part of the validated config. Still needs an ablation to isolate its individual contribution from Dixon-Coles.
- [ ] **Form decomposition**: separate ability (long-term) from form (short-term) — tested, needs stronger signal, and currently conflicts with the season-start variance inflation (Design Choice #5 inflates *both* components uniformly, which undercuts the point of splitting them)
- [ ] **Player-level data**: incorporate lineups, injuries, suspensions
- [ ] **Multiple leagues**: train joint model across competitions
- [x] **Correlation between home/away goals**: implemented via the Dixon-Coles low-score correction (Design Choice #7), optional (`use_dixon_coles=True`) — not yet wired into the live `/predict` endpoint's scoreline sampling, which still draws home/away goals independently
- [ ] **Match context**: minutes played, red cards, weather conditions
- [ ] **Real benchmark**: compare against actual bookmaker odds (RPS vs. de-vigged implied probabilities) — the only way to know if this is genuinely competitive, not just "beats a flat average". No odds data exists in this repo yet.
- [ ] **Fix the home-win-confidence calibration gap** noted in "Out-of-Sample Testing" above


**Key Findings:**
1. **Poisson is sufficient**: NB overdispersion didn't improve fit
2. **Time-varying beats static**: AR(1) dynamics capture form changes
3. **Home advantage varies**: team-specific effects matter
4. **Model generalises, with real statistical backing**: 35-window rolling CV across 2020-2025 (5 seasons, 5 relegation/promotion boundaries) gives a pooled RPS improvement over naive of 0.036 (95% CI [0.026, 0.046], excludes zero) — the first properly-powered validation result in this project's history. Individual windows aren't universally positive (29/35, 83%), consistent with a genuine but moderate edge rather than dramatic outperformance.
5. **Earlier "Bayes Factor" claims were an artifact, not a finding**: an earlier version of this README claimed a 10^5–10^8 Bayes Factor over naive from a single-season analysis. That number was computed under a likelihood bug that fed every match's outcome into training twice (Design Choice #6) — the bug inflated apparent confidence throughout, and the number should not be trusted. The RPS result in #4 is the corrected, honest replacement, and is a far more modest (but real) claim.
6. **xG re-enabled**: now part of the validated config (see "xG as Feature" above) — the original "marginal, not worth it" verdict was reached under the same likelihood bug and shouldn't be trusted either.
