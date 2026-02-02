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
    return pm.Normal("home_adv", mu=mu, sigma=sd, shape=n_teams)


def match_effect_prior(n_matches, sd_scale=0.1):
    sigma = pm.HalfNormal("sigma_match", sd_scale)
    return pm.Normal("match_effect", 0.0, sigma, shape=n_matches)
