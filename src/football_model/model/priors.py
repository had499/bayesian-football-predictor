import pymc as pm

def ar1_hyperpriors(
    name: str,
    sigma_scale: float,
    rho_alpha: float,
    rho_beta: float,
):
    sigma = pm.HalfNormal(f"sigma_{name}", sigma_scale)
    rho   = pm.Beta(f"rho_{name}", rho_alpha, rho_beta)
    return sigma, rho


def rho_prior(name: str, rho_alpha: float, rho_beta: float):
    """Just the AR(1) persistence half of ar1_hyperpriors — split out so a
    per-team sigma (ar1_hierarchical_sigma below) can be swapped in without
    also needing a second, unused scalar sigma from ar1_hyperpriors."""
    return pm.Beta(f"rho_{name}", rho_alpha, rho_beta)


def ar1_hierarchical_sigma(name: str, n_teams: int, pop_scale: float):
    """Per-team AR(1) innovation SD, partially pooled toward a shared
    population value instead of every team sharing one global sigma (WP005
    found loosening that single global value, alone or combined with other
    priors, doesn't recover resolution — WP006 tests whether letting
    volatility itself vary by team does, e.g. a promoted side or a team
    mid-managerial-change plausibly needs a bigger sigma than a stable
    top-six squad, which one shared value can't represent).

    Non-centered: HalfNormal is a scale family (HalfNormal(sigma) is exactly
    sigma * HalfNormal(1) in distribution), so `raw ~ HalfNormal(1)` fixed
    and `sigma_team = raw * sigma_pop` reparameterizes the same hierarchical
    model without sigma_team's own sampling distribution depending on
    sigma_pop — the same funnel-avoidance trick as home_advantage_prior's
    `home_adv_raw`, just via multiplicative scaling instead of an additive
    shift (a HalfNormal can't be shifted into negative territory the way a
    Normal can, so multiply by the scale here rather than add).

    Returns (sigma_team, sigma_pop): sigma_team has shape (n_teams,) and is
    what ar1_team_process should receive as its `sigma` argument; sigma_pop
    is the shared scalar teams are pooled toward (exposed mainly for
    diagnostics — how much are teams actually differing from each other).
    """
    sigma_pop = pm.HalfNormal(f"sigma_{name}_pop", pop_scale)
    raw = pm.HalfNormal(f"sigma_{name}_team_raw", 1.0, shape=n_teams)
    sigma_team = pm.Deterministic(f"sigma_{name}_team", raw * sigma_pop)
    return sigma_team, sigma_pop


def team_strength_prior(name, n_teams, scale=0.2):
    return pm.Normal(name, mu=0.0, sigma=scale, shape=n_teams)


def home_advantage_prior(n_teams, mu_center=0.13, mu_scale=0.03, sd_scale=0.02):
    mu = pm.Normal("home_mu", mu_center, mu_scale)
    sd = pm.HalfNormal("home_sd", sd_scale)
    # Non-centered: home_sd is itself a small, weakly-informed random variable,
    # so a centered Normal(mu, sd) here is exactly the classic NUTS "funnel"
    # geometry — same prior, better sampling geometry, no change in meaning.
    home_adv_raw = pm.Normal("home_adv_raw", 0.0, 1.0, shape=n_teams)
    return pm.Deterministic("home_adv", mu + sd * home_adv_raw)


def match_effect_prior(n_matches, sd_scale=0.1):
    sigma = pm.HalfNormal("sigma_match", sd_scale)
    return pm.Normal("match_effect", 0.0, sigma, shape=n_matches)
