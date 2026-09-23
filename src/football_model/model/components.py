import pymc as pm
import pytensor.tensor as pt
from pytensor import scan

def ar1_team_process(
    name,
    n_time,
    n_teams,
    sigma,
    rho,
    active_mask=None,
    season_start_mask=None,
    season_start_sigma_mult=1.0,
    init_scale=0.2,
    entry_shift=0.0,
):
    """Non-centered AR(1) random walk per team.

    entry_shift: scalar added to a team's latent value in the one round where
        it flips from inactive to active (a promoted team's first round —
        WP007). Real promoted sides start well below the league average
        (~-0.37 attack / ~-0.26 defence in log-goal terms, 33 team-seasons),
        but a prior of N(0, init_scale) says "average team". The shift is
        applied once, at entry, and then persists through the walk — rather
        than being a first-season-only offset that would have to vanish
        abruptly in season two. 0.0 (default) leaves every earlier config's
        recursion exactly unchanged. Pass a pm random variable to learn it.

    active_mask: optional (n_time, n_teams) 0/1 array. Where a team is
        inactive (e.g. relegated), the walk is frozen at its last active
        value instead of continuing to accumulate innovation noise — a
        team's latent strength should hold steady while they're out of the
        league, not drift toward the population mean on noise alone.
    season_start_mask: optional (n_time,) 0/1 array flagging each season's
        opening rounds. Innovation variance is inflated there (scaled by
        season_start_sigma_mult) to reflect summer squad turnover, which
        affects every team, not just ones returning from relegation.
    sigma: either a scalar (one innovation SD shared by every team — the
        default, everything before WP006) or a (n_teams,) vector (one SD per
        team, e.g. from ar1_hierarchical_sigma in priors.py — WP006). Both
        go through the same `sigma * (...)` line below; broadcasting handles
        which one it is, so nothing else in this function needs to know.
    """
    # Non-centered parameterization for better sampling when sigma is small
    z = pm.Normal(f"{name}_std", 0, 1, shape=(n_time, n_teams))

    if active_mask is None:
        active_arr = pt.ones((n_time, n_teams))
    else:
        active_arr = pt.as_tensor_variable(active_mask)

    if season_start_mask is None:
        season_arr = pt.zeros((n_time,))
    else:
        season_arr = pt.as_tensor_variable(season_start_mask)

    # Per-timestep (and, if sigma is per-team, per-team) sigma: inflated
    # during each season's opening window. season_arr[:, None] makes this
    # (n_time, 1) so it broadcasts against a scalar sigma -> (n_time, 1)
    # (unchanged behaviour from before WP006: each scan step still gets a
    # size-1 value that broadcasts fine against z_t's (n_teams,)) or against
    # a (n_teams,) sigma -> (n_time, n_teams), one column per team.
    sigma_t = sigma * (1.0 + (season_start_sigma_mult - 1.0) * season_arr[:, None])

    # 1.0 exactly where a team is active at t but was not at t-1.
    entry_arr = active_arr[1:] * (1.0 - active_arr[:-1])

    # Use scan instead of Python loop for efficiency with many time steps
    def step(z_t, active_t, sigma_t_val, entry_t, x_prev, rho, shift):
        x_new = rho * x_prev + sigma_t_val * z_t + entry_t * shift
        # Freeze at last value while the team is inactive (out of the league)
        return pt.switch(active_t > 0, x_new, x_prev)

    # Initialize with first time step
    x_init = init_scale * z[0]

    # Scan over remaining time steps (t=1 to n_time-1)
    x_rest = scan(
        fn=step,
        sequences=[z[1:], active_arr[1:], sigma_t[1:], entry_arr],
        outputs_info=[x_init],
        non_sequences=[rho, entry_shift],
        n_steps=n_time - 1,
        strict=True,
        return_updates=False,
    )

    # Concatenate initial and scanned results
    result = pt.concatenate([x_init[None, :], x_rest], axis=0)

    return pm.Deterministic(name, result)


def centered_over_teams(x, name, active_mask=None):
    """Center x across teams at each time point.

    If active_mask is given, only currently-active teams (n_time, n_teams)
    contribute to the per-round mean — otherwise relegated/not-yet-promoted
    teams' frozen or unconstrained values would distort the "average current
    team = 0" identifiability constraint.
    """
    if active_mask is None:
        return pm.Deterministic(
            name,
            x - x.mean(axis=1, keepdims=True)
        )

    active_arr = pt.as_tensor_variable(active_mask)
    n_active = pt.maximum(active_arr.sum(axis=1, keepdims=True), 1.0)
    masked_mean = (x * active_arr).sum(axis=1, keepdims=True) / n_active
    return pm.Deterministic(
        name,
        x - masked_mean
    )


def masked_mean_over_teams(x, active_mask=None):
    """Per-timestep mean of x across teams, counting only active teams."""
    if active_mask is None:
        return x.mean(axis=1)

    active_arr = pt.as_tensor_variable(active_mask)
    n_active = pt.maximum(active_arr.sum(axis=1), 1.0)
    return (x * active_arr).sum(axis=1) / n_active


def soft_clip(x, limit=2.0):
    return limit * pm.math.tanh(x / limit)


def dixon_coles_tau(lambda_home, lambda_away, goals_home, goals_away, rho):
    """Elementwise Dixon-Coles (1997) correction factor tau(x, y; lambda, mu, rho).

    Equals 1 everywhere except the four low-scoring cells (0-0, 1-0, 0-1,
    1-1), where it nudges the joint probability to correct for the
    correlation an independent home/away Poisson likelihood can't capture
    (e.g. a leading team sitting back late on).

    Dixon & Coles (1997), eq. 4, with x = home goals ~ Poisson(lambda) and
    y = away goals ~ Poisson(mu):
        tau(0,0) = 1 - lambda*mu*rho     tau(0,1) = 1 + lambda*rho
        tau(1,0) = 1 + mu*rho            tau(1,1) = 1 - rho
    i.e. a cell's correction uses the rate of the side that scored 0 — a
    1-0 home win is the AWAY side's zero, so it scales with `mu`. (Before
    WP007 this had the two single-goal cells swapped, which makes the joint
    sum to 1 + rho*(lambda-mu)^2*exp(-lambda-mu) instead of exactly 1: an
    improper likelihood whenever the two rates differ, i.e. every match.)
    """
    goals_home = pt.as_tensor_variable(goals_home)
    goals_away = pt.as_tensor_variable(goals_away)

    is_00 = pt.eq(goals_home, 0) & pt.eq(goals_away, 0)
    is_10 = pt.eq(goals_home, 1) & pt.eq(goals_away, 0)
    is_01 = pt.eq(goals_home, 0) & pt.eq(goals_away, 1)
    is_11 = pt.eq(goals_home, 1) & pt.eq(goals_away, 1)

    return pt.switch(
        is_00, 1 - lambda_home * lambda_away * rho,
        pt.switch(
            is_10, 1 + lambda_away * rho,
            pt.switch(
                is_01, 1 + lambda_home * rho,
                pt.switch(is_11, 1 - rho, 1.0),
            ),
        ),
    )


def dixon_coles_adjustment(lambda_home, lambda_away, goals_home, goals_away, sd=0.1):
    """Add the Dixon-Coles low-score correlation correction as a pm.Potential.

    Learns a single correlation parameter `rho_dc` and nudges just the four
    low-scoring cells (0-0, 1-0, 0-1, 1-1) via tau(x, y). Adding
    log(tau).sum() as a Potential is mathematically equivalent to
    multiplying the joint likelihood by tau — no change needed to the
    existing independent Poisson likelihoods themselves.
    """
    rho = pm.Normal("rho_dc", 0.0, sd)
    tau = dixon_coles_tau(lambda_home, lambda_away, goals_home, goals_away, rho)
    # Guard against tau <= 0 (possible for extreme rho/lambda combos,
    # especially early in tuning) so log() stays finite. Only a lower bound
    # is needed — large tau is harmless.
    tau = pt.maximum(tau, 1e-6)
    return pm.Potential("dixon_coles", pt.log(tau).sum())


def xg_measurement_channel(
    lambda_home, lambda_away, xg_home, xg_away,
    weight=0.5, shape_alpha=2.0, shape_beta=0.5,
):
    """Treat each side's realised match xG as a second, noisy measurement of
    the SAME latent scoring rate that generates its goals (WP007).

    Why not `goals ~ Poisson(xG)` or an xG covariate: within a team-season
    Var(goals)/Var(xG) is ~2.2x, and xG's CV^2 is ~0.32 — xG is a *smoother*
    signal of the rate, not a rate whose realisations are Poisson. So it gets
    its own Gamma observation model with mean `kappa * lambda` and a learned
    shape `xg_shape` (Var = mean^2 / shape), and it sharpens attack/defence
    without changing the goals likelihood or the goals predictive. Empirically
    first-half xG predicts second-half goals better than first-half goals do
    (r=0.73 vs 0.68 attack, 0.62 vs 0.55 defence), which is exactly what a
    second measurement channel exploits and the old rolling-mean covariate
    (beta_xG ~ HalfNormal(0.1), i.e. ~zero weight) could not.

    `weight` tempers the channel's log-density: goals and xG share the same
    shot-level information, so treating them as fully independent evidence
    would double-count. 0 disables it; 1 is full trust.

    xg_home/xg_away are the match's OWN realised xG — fine for training rows
    (an observation), never a predictor for the match being forecast.
    """
    xg_home = pt.as_tensor_variable(xg_home)
    xg_away = pt.as_tensor_variable(xg_away)
    shape = pm.Gamma("xg_shape", alpha=shape_alpha, beta=shape_beta)
    kappa = pm.LogNormal("xg_kappa", mu=0.0, sigma=0.1)
    logp = 0.0
    for lam, obs in ((lambda_home, xg_home), (lambda_away, xg_away)):
        gamma = pm.Gamma.dist(alpha=shape, beta=shape / (kappa * lam))
        logp = logp + pm.logp(gamma, obs).sum()
    return pm.Potential("xg_channel", weight * logp)
