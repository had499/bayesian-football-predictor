# Bayesian Football Predictor

A Bayesian hierarchical model for predicting English Premier League football match outcomes using PyMC and FastAPI.

## Overview

This project implements a time-varying Bayesian model that learns team strengths from historical match data and generates probabilistic predictions for upcoming fixtures. The model accounts for:
- Dynamic team attack and defense strengths that evolve over time
- Team-specific home advantage
- Expected goals (xG) as a baseline prior
- Full uncertainty quantification through posterior distributions

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
- **Validation**: Rolling window cross-validation shows consistent **MAE: 0.765 ± 0.137** across 13 test windows, beating naive baseline by ~15-30 LL points per window. Model performance stable from round 25 onwards.

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


### Model Parameters

| Parameter | Description | Prior/Constraint | Learned Value |
|-----------|-------------|------------------|---------------|
| `attack[t, team]` | Team's attacking strength at time t | AR(1), σ ~ HalfNormal(0.1) | Varies by team (±0.5 range) |
| `defense[t, team]` | Team's defensive weakness at time t | AR(1), σ ~ HalfNormal(0.1) | Varies by team (±0.5 range) |
| `home_adv[team]` | Team-specific home advantage | Normal(0, 0.5) | Typically 0.1-0.3 (10-35% boost) |
| `ρ_att`, `ρ_def` | Persistence of team strength | Beta(9, 1) → ~0.9 | Learned: ~0.92-0.95 (high persistence) |
| `σ_att`, `σ_def` | Innovation in team strength | HalfNormal(0.1) | Learned: ~0.05-0.08 (gradual change) |

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
2. **Independence**: Assumes goals don't affect each other (reasonable approximation)
3. **Stationary within-season**: Team strength changes smoothly, no sudden jumps

### Out-of-Sample Testing
Notebooks include rolling window cross-validation:
- Train on rounds 1-N
- Predict round N+1
- Repeat for all N
- Measure MAE and log-likelihood vs baselines


## Experimental Extensions

Several model variants were tested in `modelling.ipynb` but not deployed:

### xG as Feature (Tested in Notebook)
The model has built-in support for using expected goals (xG) as a feature:
```python
λ = exp(β_xG * log(xG) + attack - defense + home_adv)
```

**Status**: Code exists (`config.use_xG=True`) but **disabled**.

**Findings from notebook**:
- β_xG learned value: ~0.85-0.95 (model down-weights xG, doesn't fully trust it)
- Performance: Marginal improvement over pure team strength model
- Trade-off: Adds dependency on xG data quality and availability
- Decision: Keep model simple and data-independent for now

**Alternative tested**: Opponent-adjusted xG (trust xG more vs strong defenses) - minimal improvement.

### Form Decomposition Model 
Separates team strength into:
- **Ability**: Long-term stable strength (ρ ≈ 0.98, σ ≈ 0.003)
- **Form**: Short-term fluctuations (ρ ≈ 0.89, σ ≈ 0.015)

```python
attack[t] = ability_attack[t] + form_attack[t]
defense[t] = ability_defense[t] + form_defense[t]
```

**Result**: Minimal improvement over single time-varying parameter (~0.02 MAE reduction). Added complexity not justified. May revisit if form signals (injuries, managerial changes) become available.


## Future Enhancements

### Model Improvements
- [ ] **Enable xG feature**: Tested in notebook with good results - deploy if xG data integration added
- [ ] **Form decomposition**: Separate ability (long-term) from form (short-term) - tested, needs stronger signal
- [ ] **Player-level data**: Incorporate lineups, injuries, suspensions
- [ ] **Multiple leagues**: Train joint model across competitions
- [ ] **Correlation**: Model correlation between home/away goals (current assumption: independent)
- [ ] **Match context**: Minutes played, red cards, weather conditions


**Key Findings:**
1. **Poisson is sufficient**: NB overdispersion didn't improve fit
2. **Time-varying beats static**: AR(1) dynamics capture form changes  
3. **Home advantage varies**: Team-specific effects matter (0.1-0.3 boost)
4. **Model generalises**: Consistent performance across 13 CV windows
5. **Beats baselines significantly**: 10^5-10^8 Bayes Factor over naive
6. **xG optional**: Tested xG feature (β_xG ~ 0.9) but kept model simple - not needed for good performance
