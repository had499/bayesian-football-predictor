import pymc as pm

def rho_prior(name: str, rho_alpha: float, rho_beta: float):
    """AR(1) persistence prior, shared by every team. Kept separate from
    whichever sigma prior is in play (a plain global HalfNormal, or the
    per-team ar1_hierarchical_sigma below) so build_model can mix and match
    without a sigma it doesn't need tagging along."""
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


def league_home_advantage_prior(
    league_names, league_n_teams, mu_center=0.13, mu_scale=0.03,
    between_league_scale=0.03, sd_scale=0.02,
):
    """Multi-league extension of home_advantage_prior (WP011): adds ONE more
    level to the existing team -> global pooling, making it
    team -> league -> global. Each league's own average home advantage
    (`mu_league`) is itself partially pooled toward a shared `mu_global`,
    instead of every team in every league being pooled toward one identical
    mean the way the single-league version does — home advantage is
    well-documented to vary systematically by country/competition, unlike
    (as far as WP011's diagnostics found any reason to believe) the
    within-league team-to-team spread, which is why only mu gets a league
    level here: `sd` (how much teams within a league differ from their own
    league's mean) stays a single shared scalar across every league, the
    same role config.home_adv_sd already played pre-WP011 — WP011 only
    extends the part of the hierarchy with a specific, evidenced reason to
    vary by league; adding more per-league flexibility than that isn't
    evidence-backed (see WP006's per-team sigma: more granularity than the
    evidence calls for tends to just pool noise, not signal).

    Non-centered throughout, same funnel-avoidance pattern as
    home_advantage_prior and ar1_hierarchical_sigma.

    league_names/league_n_teams: parallel lists, one entry per league, used
    only to build readable/unique PyMC variable names (`home_adv_<league>`)
    so multiple leagues' worth of per-team home_adv can coexist in one
    pm.Model without name collisions.

    Returns (mu_global, mu_league, sd, home_adv_by_league): home_adv_by_league
    is a list of (n_teams_l,) tensors, one per league, in the same order as
    the inputs.
    """
    assert len(league_names) == len(league_n_teams)
    n_leagues = len(league_names)

    mu_global = pm.Normal("home_mu_global", mu_center, mu_scale)
    between_league_sd = pm.HalfNormal("home_mu_between_league_sd", between_league_scale)
    mu_league_raw = pm.Normal("home_mu_league_raw", 0.0, 1.0, shape=n_leagues)
    mu_league = pm.Deterministic("home_mu_league", mu_global + between_league_sd * mu_league_raw)

    sd = pm.HalfNormal("home_sd", sd_scale)  # shared within-league team spread

    home_adv_by_league = []
    for l, (league_name, n_teams_l) in enumerate(zip(league_names, league_n_teams)):
        raw = pm.Normal(f"home_adv_raw_{league_name}", 0.0, 1.0, shape=n_teams_l)
        home_adv_l = pm.Deterministic(f"home_adv_{league_name}", mu_league[l] + sd * raw)
        home_adv_by_league.append(home_adv_l)

    return mu_global, mu_league, sd, home_adv_by_league
