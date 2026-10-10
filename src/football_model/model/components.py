import numpy as np
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
    entry_scale=None,
    revert_to=None,
    z=None,
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
    # Non-centered parameterization for better sampling when sigma is small.
    #
    # WP037: `z` may be supplied by the caller instead, so the attack and defence
    # walks can share CORRELATED innovations. Note x_init = init_scale * z[0] below,
    # so injecting a correlated z correlates the starting LEVELS and the per-round
    # innovations in one stroke -- which is what we want, since the measured
    # cross-team corr(attack, defence) of +0.79 is a statement about levels.
    if z is None:
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

    # WP025: optional per-(time, team) weight on the entry event, so `entry_shift`
    # becomes a COEFFICIENT rather than a fixed offset. With standardised external
    # Elo as the weight, a promoted side's starting level scales with how strong it
    # actually is, instead of every promoted side getting the same shift. WP007's
    # global offset fitted -0.06 against a real gap of -0.37, and measured pre-debut
    # Elo spans 1475 (Huddersfield 2017) to 1726 (Burnley 2023) -- a spread one
    # scalar cannot represent. None leaves the recursion exactly as before.
    if entry_scale is not None:
        entry_arr = entry_arr * pt.as_tensor_variable(entry_scale)[1:]

    # Use scan instead of Python loop for efficiency with many time steps
    # WP025: optional REVERSION TARGET. Without it the walk reverts toward 0 --
    # "the average team". With an external strength estimate as the target, the walk
    # reverts toward where that estimate says the team belongs, while match results
    # still drive short-term movement. At the fitted rho ~ 0.993 the pull is 0.7% per
    # round, ~21% of the gap over a season: an anchor, not an override.
    #
    # This is the GENERAL form of the entry_scale idea above. entry_scale only acts
    # once, when a team joins the league; revert_to acts on every round for every
    # team, so it also carries information that arrives mid-season (a deep European
    # run lifts a side's Elo, and the EPL-only pipeline cannot see those matches).
    if revert_to is None:
        def step(z_t, active_t, sigma_t_val, entry_t, x_prev, rho, shift):
            x_new = rho * x_prev + sigma_t_val * z_t + entry_t * shift
            # Freeze at last value while the team is inactive (out of the league)
            return pt.switch(active_t > 0, x_new, x_prev)

        x_init = init_scale * z[0]
        seqs = [z[1:], active_arr[1:], sigma_t[1:], entry_arr]
    else:
        target = pt.as_tensor_variable(revert_to)

        def step(z_t, active_t, sigma_t_val, entry_t, mu_t, x_prev, rho, shift):
            x_new = mu_t + rho * (x_prev - mu_t) + sigma_t_val * z_t + entry_t * shift
            return pt.switch(active_t > 0, x_new, x_prev)

        # start AT the target rather than at zero, else the walk spends its first
        # season drifting toward an anchor it should have begun on
        x_init = target[0] + init_scale * z[0]
        seqs = [z[1:], active_arr[1:], sigma_t[1:], entry_arr, target[1:]]

    # Scan over remaining time steps (t=1 to n_time-1)
    x_rest = scan(
        fn=step,
        sequences=seqs,
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
    season_idx=None, n_seasons=0, kappa_tau_sd=0.1, obs_weight=None,
    name="xg", shape_fixed=None,
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

    `name` prefixes every random variable this function creates, so the same Gamma
    measurement machinery can carry a SECOND, non-xG observation of the latent rate.
    WP039 uses name="mkt" for market-implied rates. The default "xg" keeps every
    earlier result's variable names byte-identical.
    """
    xg_home = pt.as_tensor_variable(xg_home)
    xg_away = pt.as_tensor_variable(xg_away)
    # WP039b: `shape_fixed` pins the Gamma shape instead of learning it. The shape IS
    # the channel's trust: Var = mean^2/shape, so CV = 1/sqrt(shape). WP039's market
    # channel learned shape 58-120 (CV 9%) against a Gamma(2,0.5) prior with mean 4,
    # i.e. it chose to believe the market's implied rate almost exactly -- which made
    # the likelihood extremely stiff and drove max rhat to 1.412. Pinning it turns
    # "how much do we trust this measurement" into a knob we set rather than one the
    # training rows get to maximise.
    if shape_fixed is not None:
        shape = pt.as_tensor_variable(float(shape_fixed))
    else:
        shape = pm.Gamma(f"{name}_shape", alpha=shape_alpha, beta=shape_beta)
    if season_idx is None:
        kappa = pm.LogNormal(f"{name}_kappa", mu=0.0, sigma=0.1)
    else:
        # Season-varying calibration (WP014). Non-centered so the per-season
        # deviations don't sit in a funnel with their own scale.
        mu_k = pm.Normal(f"{name}_kappa_mu", 0.0, 0.1)
        tau_k = pm.HalfNormal(f"{name}_kappa_tau", kappa_tau_sd)
        z_k = pm.Normal(f"{name}_kappa_z", 0.0, 1.0, shape=n_seasons)
        kappa_s = pm.Deterministic(f"{name}_kappa_season", pt.exp(mu_k + tau_k * z_k))
        kappa = kappa_s[season_idx]
    logp = 0.0
    for lam, obs in ((lambda_home, xg_home), (lambda_away, xg_away)):
        gamma = pm.Gamma.dist(alpha=shape, beta=shape / (kappa * lam))
        ll = pm.logp(gamma, obs)
        logp = logp + (ll if obs_weight is None else obs_weight * ll).sum()
    return pm.Potential(f"{name}_channel", weight * logp)


def global_walk(name, n_time, sigma_scale, init_mu=0.0, init_sd=0.2):
    """A slow random walk on a single global scalar over rounds (WP015).

    Used for league-wide home advantage and the global scoring level — both
    measured to drift substantially while the model held them constant. Simpler
    than ar1_team_process: one series, no per-team axis, no active mask (a
    league-level quantity is always "active"), rho fixed at 1 so the level can
    genuinely wander rather than reverting to a fixed mean.

    Returns a (n_time,) Deterministic; index it with the observation's t.
    """
    n_time = int(n_time)   # ModelData.n_time is numpy.int64; pm.shape rejects it
    sigma = pm.HalfNormal(f"sigma_{name}", sigma_scale)
    z = pm.Normal(f"{name}_z", 0.0, 1.0, shape=n_time)
    x0 = pm.Normal(f"{name}_0", init_mu, init_sd)
    # cumsum is exact for rho=1 and avoids a scan entirely
    return pm.Deterministic(name, x0 + pt.concatenate([pt.zeros(1), pt.cumsum(sigma * z[1:])]))


def outcome_likelihood(lambda_home, lambda_away, goals_home, goals_away,
                       weight=1.0, max_goals=10):
    """Score the observed HOME/DRAW/AWAY outcome directly (WP017).

    The model otherwise maximises GOAL-COUNT likelihood and is evaluated on
    outcome probabilities (RPS). Those are not the same objective: the H/D/A
    split falls out of convolving two independent Poissons, and getting both
    marginal goal rates right does not guarantee the convolution allocates
    probability correctly between the three outcomes. Measured on 1,314 CV
    matches, the model over-predicts home wins by ~1.4pp and under-predicts
    away wins by ~1.5pp while its fitted `home_mu` (0.211) matches the data's
    own home:away log-ratio (0.204) almost exactly.

    This adds `weight * sum(log P(observed outcome | lambda_home, lambda_away))`
    so training can trade a little goal-count fit for a correct outcome split —
    the latents shift slightly off goal-optimal rather than a final rescaling
    being applied after the fact (which is what WP004's post-hoc Platt did).

    IMPORTANT caveat: the outcome is a deterministic function of the goals, so
    this term DOUBLE-COUNTS the same observation at a coarser granularity. That
    is deliberate — it is an objective re-weighting (a weighted composite
    likelihood), not a claim of independent information. The consequence is that
    the posterior is no longer the posterior of a well-specified generative
    model, so its credible intervals lose their usual interpretation. Point
    predictions remain usable, which is what RPS scores. `weight=0` recovers the
    plain model exactly.
    """
    K = int(max_goals)
    g = pt.arange(K + 1)
    lgam = pt.gammaln(g + 1.0)
    # log Poisson pmf on a grid: (n_obs, K+1) for each side
    log_ph = g[None, :] * pt.log(lambda_home)[:, None] - lambda_home[:, None] - lgam[None, :]
    log_pa = g[None, :] * pt.log(lambda_away)[:, None] - lambda_away[:, None] - lgam[None, :]
    joint = pt.exp(log_ph[:, :, None] + log_pa[:, None, :])          # (n_obs, K+1, K+1)

    gh_grid, ga_grid = np.meshgrid(np.arange(K + 1), np.arange(K + 1), indexing="ij")
    masks = pt.as_tensor_variable(np.stack([
        (gh_grid > ga_grid).astype("float64"),      # home win
        (gh_grid == ga_grid).astype("float64"),     # draw
        (gh_grid < ga_grid).astype("float64"),      # away win
    ]))                                             # (3, K+1, K+1)
    p3 = pt.tensordot(joint, masks, axes=[[1, 2], [1, 2]])           # (n_obs, 3)
    p3 = p3 / p3.sum(axis=1, keepdims=True)         # renormalise the truncated grid

    # which outcome actually happened is DATA, so index it directly
    gh = np.asarray(goals_home).astype(int)
    ga = np.asarray(goals_away).astype(int)
    obs_idx = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))
    p_obs = p3[pt.arange(len(obs_idx)), pt.as_tensor_variable(obs_idx)]
    return pm.Potential("outcome_likelihood", weight * pt.log(pt.maximum(p_obs, 1e-12)).sum())



def count_measurement_channel(
    name, lambda_home, lambda_away, obs_home, obs_away,
    weight=0.5, kappa_mu=0.0, kappa_sd=0.2, alpha_beta=0.1, obs_weight=None,
):
    """A tempered NegBinomial measurement channel on the shared latent rate.

    Generic form of the xG channel's idea for COUNT observables: some per-match
    count is treated as a second, noisy measurement of the same `lambda` that
    generates goals, with its own scale (`{name}_kappa`, free, so the channel
    supplies information through the PATTERN of counts and not their level) and
    its own dispersion (`{name}_alpha`).

    NegBinomial rather than Poisson because these counts are overdispersed
    (shots var/mean 2.35, deep completions 3.09). `weight` tempers the
    log-density: goals, xG and any shot- or possession-derived count all derive
    from overlapping match events, so untempered they would be counted as
    independent evidence several times over.

    Adds TWO scalars and ZERO latents — deliberately the xG-channel shape, not
    WP016's latent split, which added ~14,600 parameters and made the model worse.
    """
    obs_home = pt.as_tensor_variable(obs_home)
    obs_away = pt.as_tensor_variable(obs_away)
    kappa = pm.LogNormal(f"{name}_kappa", mu=kappa_mu, sigma=kappa_sd)
    alpha = pm.HalfNormal(f"{name}_alpha", 1.0 / alpha_beta)
    logp = 0.0
    for lam, obs in ((lambda_home, obs_home), (lambda_away, obs_away)):
        nb = pm.NegativeBinomial.dist(mu=kappa * lam, alpha=alpha)
        ll = pm.logp(nb, obs)
        logp = logp + (ll if obs_weight is None else obs_weight * ll).sum()
    return pm.Potential(f"{name}_channel", weight * logp)


def deep_completion_channel(
    lambda_home, lambda_away, deep_home, deep_away,
    weight=0.5, kappa_mu=1.62, kappa_sd=0.2, obs_weight=None,
):
    """Observe DEEP COMPLETIONS as a measurement of the same latent rate (WP020).

    Understat's `deep`: passes completed within ~20m of the opposing goal,
    excluding crosses. The reason this is worth a channel when WP019's shot count
    was not: it is **not a function of the shot list**. Every decomposition of the
    shot feed tried in WP019 correlated >=0.93 with shot count or total xG, because
    they are all sums over the same 116k shots — `(count, total)` nearly spans that
    feed. Deep completions measure possession worked into dangerous areas that did
    NOT produce a shot, which is structurally invisible to a model whose only
    observables are goals and shot-derived xG.

    Sizing, from the per-match premise check that correctly predicted WP019's null
    (n=7,680, leave-one-season-out, team-clustered): on top of goals+xG, deep adds
    OOS R^2 +0.0535 [+0.0107, +0.1094] against +0.0457 for xG over goals — ~117%
    of the effect of the one channel that works. Default kappa log-mean 1.62 is
    log(6.87/1.36), the data's deep-per-unit-rate ratio.

    CAVEAT recorded before running: the framework this prediction comes from is
    pooled across teams, so part of deep's measured value may be team
    IDENTIFICATION that the attack/defence latents already carry. A null here
    would indicate exactly that, and would discredit the premise-check approach.
    """
    return count_measurement_channel(
        "deep", lambda_home, lambda_away, deep_home, deep_away,
        weight=weight, kappa_mu=kappa_mu, kappa_sd=kappa_sd, obs_weight=obs_weight,
    )


def shot_count_channel(
    lambda_home, lambda_away, shots_home, shots_away,
    weight=0.5, kappa_mu=2.175, kappa_sd=0.2, alpha_beta=0.1, obs_weight=None,
):
    """Observe each side's SHOT COUNT as a third measurement of the same latent
    scoring rate (WP019).

    Why this is not the WP016 mistake: WP016 split the defence latent in two and
    made the model worse, because a split adds ~14,600 parameters whose noise
    flows into the combined prediction. This adds ONE latent-free observation
    channel and two scalars (`shot_kappa`, `shot_alpha`) on top of the existing
    attack/defence walks — structurally the xG channel, which is the only thing
    in this project that has ever worked.

    Why shot count carries information total xG does not: xG is shots x quality,
    so a channel observing xG alone constrains only their product. Regressing a
    team-season's second-half goals/match on its first-half log-volume and
    log-quality separately gives coefficients +1.028 and +0.554 (difference
    +0.474, cluster-bootstrap CI [+0.132, +0.771]); total xG forces them EQUAL.
    Volume is also the more repeatable skill (first->second-half r = +0.770 vs
    +0.567), so it is the lower-noise indicator of the two.

    NegBinomial rather than Poisson: shots/match are overdispersed, var/mean
    2.35 on this sample. `shot_kappa` is the shots-per-unit-rate scale (~8.8
    here, hence the default log-mean 2.175) and is free, so the channel supplies
    information through the *pattern* of shot counts, not their level.

    `weight` tempers the log-density exactly as in `xg_measurement_channel`:
    goals, xG and shots all derive from the same shot-level events, so untempered
    they would be triple-counted as independent evidence.
    """
    return count_measurement_channel(
        "shot", lambda_home, lambda_away, shots_home, shots_away,
        weight=weight, kappa_mu=kappa_mu, kappa_sd=kappa_sd, alpha_beta=alpha_beta,
        obs_weight=obs_weight,
    )


def time_decay_weights(match_day, half_life_days, normalise=True):
    """Dixon & Coles (1997) time decay: weight each match by exp(-xi * age).

    This is the OTHER half of that paper, and the half this project never
    implemented — WP002/WP007 tested only the tau correction. It is a genuinely
    different mechanism from the AR(1) walk: a walk lets team strengths DRIFT,
    whereas decay changes how much each observation COUNTS. Dixon & Coles needed
    decay because their strengths were static; ours are not, so whether it adds
    anything is an open question rather than a foregone one.

    Premise check (9,120 team-matches, predicting each team's next match from a
    decay-weighted mean of its past xG): the optimal half-life is ~270 days, worth
    -1.01% MSE against no decay, with the curve flat between 180 and 545 days and
    SHORT half-lives clearly worse (30 days: +2.19%). That was measured on a static
    weighted mean with no walk at all, so it is an upper bound on what the model
    can gain, not a prediction.

    `normalise=True` rescales the weights to average 1.0. This isolates the thing
    being tested — RECENCY WEIGHTING — from a confound: raw exp(-xi*age) weights
    shrink the total likelihood mass, which widens the posterior simply because the
    model is acting on less effective data. With a 270-day half-life over 10 years
    of history the raw effective sample size collapses, so an unnormalised arm would
    conflate "recency helps" with "less data hurts". Set False for the literal
    Dixon-Coles form.
    """
    day = pt.as_tensor_variable(np.asarray(match_day, dtype="float64"))
    xi = np.log(2.0) / float(half_life_days)
    w = pt.exp(-xi * (day.max() - day))       # age measured back from the most recent match
    if normalise:
        w = w / w.mean()
    return w


def avail_offset_terms(out_h, out_a, k, log):
    """WP061: the availability Poisson offset, for BOTH the home and away side.

        offset_home = log(1 - k*out_home) + log(1 + k*out_away)
        offset_away = log(1 - k*out_away) + log(1 + k*out_home)

    A team's own unavailable minutes depress its scoring rate and lift its opponent's.
    `k` is FIXED, never estimated -- that is the whole point. Every fitted covariate tried
    in this project has been shrunk to nothing by the ~29,000 latents (additive terms land
    at ~2% of prior SD), so there must be no parameter here to shrink.

    `log` is injected (`pm.math.log` from build_model, `np.log` from predict.py) so the
    model and the predictor share ONE implementation. model.py's theta is otherwise synced
    to predict.py BY HAND, and two real bugs have already come from that gap -- a missing
    xG term and a swapped team pair. A shared function removes the gap structurally rather
    than relying on a test to notice it drifting.
    """
    return (log(1.0 - k * out_h) + log(1.0 + k * out_a),
            log(1.0 - k * out_a) + log(1.0 + k * out_h))
