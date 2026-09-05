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
