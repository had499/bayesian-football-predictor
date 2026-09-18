"""Shared prediction formula, used by both the CV harness
(scripts/run_cv_window.py) and the live service
(services/predictor/predictor.py). Exists so both consumers derive
predictions from the SAME formula as
football_model.model.model.build_model's theta_home/theta_away — instead of
each hand-reimplementing it separately, which is exactly how a discrepancy
(a missing xG term in run_cv_window.py) went unnoticed for a while.

Plain numpy, not PyTensor: designed to run on posterior MEANS (scalars) or
full posterior SAMPLE arrays (vectorized) after training/sampling is done,
outside any pm.Model context. There's no way to literally share code across
the PyTensor (training-time, symbolic) / numpy (prediction-time, numeric)
boundary, so the two are kept in sync by hand — if build_model's theta
formula changes, this module needs updating to match, and vice versa.

`predict_rows` below is the OTHER thing that must live here and nowhere
else: which team/time index and which xG baseline a prediction reads for a
given match. That bookkeeping used to get hand-rebuilt inside
run_cv_window.py (a fresh pd.unique()-based team mapping, a separate
(t, team, opp) -> xG dict) — and because ModelData.team_mapping depends on
row order, that reconstruction silently disagreed with prepare_model_data's
own mapping for two teams, corrupting predictions for them without ever
raising an error. There must be exactly one place that turns "a match" into
"a (t, team_idx, opp_idx, xG) tuple" — a ModelData built by
prepare_model_data — and every consumer reads that struct directly instead
of re-deriving it.

`dixon_coles_tau`/`dixon_coles_log_correction`/`dc_outcome_probs` mirror
football_model.model.components.dixon_coles_tau, for the same reason as
everything above: `dixon_coles_adjustment` only ever runs *inside*
build_model, as a training-time pm.Potential shaping the posterior. Nothing
outside of it applied that correction when scoring predictions — CV's
log-likelihood and RPS/calibration were computed as if Dixon-Coles were
always off, silently, regardless of `use_dixon_coles`. These three
functions are what evaluation code (scripts/run_cv_window.py, the WP001/
WP002 notebooks) must route through instead of re-deriving tau by hand, so
this exact gap can't reopen in a second place the way the xG one did.
"""
import numpy as np
from scipy.stats import poisson


def soft_clip(x, limit=2.0):
    """Numpy mirror of football_model.model.components.soft_clip."""
    return limit * np.tanh(x / limit)


def compute_theta(
    attack_for,
    defense_against,
    home_adv_for,
    is_home,
    *,
    beta_xG=None,
    xG_for=None,
    use_opponent_adjusted_xG=False,
    xG_adjustment_strength=0.3,
    beta_lineup=None,
    lineup_dev_for=None,
):
    """theta for one side of a match — the team whose goal rate this is.

    Mirrors build_model's theta_home/theta_away: call this once per side.
    For the home team: (attack[home], defense[away], home_adv[home],
    is_home=1). For the away team: (attack[away], defense[home],
    home_adv[away] — irrelevant, since is_home=0 zeroes it out — is_home=0).
    home_adv only ever applies when is_home=1, exactly like `* data.home`
    in model.py.

    `defense_against` doubles as the opponent-adjusted-xG adjustment input
    in both directions — for the home side it's the away team's defense
    (model.py's `defense[opp_idx]`), for the away side it's the home team's
    defense (`defense[team_idx]`), which is exactly what
    `use_opponent_adjusted_xG` adjusts by in build_model.

    `lineup_dev_for` (WP008) is already a log-ratio (today's starting-XI
    quality vs. that team's own recent normal — see
    football_model.features.lineup_features), so unlike xG it's added
    directly, no extra log() here — matching `beta_lineup * lineup_dev_home`
    in build_model exactly.
    """
    theta = attack_for - defense_against + home_adv_for * is_home

    if beta_xG is not None and xG_for is not None:
        if use_opponent_adjusted_xG:
            adj = np.clip(1.0 + xG_adjustment_strength * defense_against, 0.5, 1.5)
            beta_xG = beta_xG * adj
        theta = theta + beta_xG * np.log(xG_for + 0.01)

    if beta_lineup is not None and lineup_dev_for is not None:
        theta = theta + beta_lineup * lineup_dev_for

    return theta


def predict_match_lambdas(
    attack_team,
    defense_team,
    attack_opp,
    defense_opp,
    home_adv_team,
    clip_theta,
    *,
    beta_xG=None,
    xG_team=None,
    xG_opp=None,
    use_opponent_adjusted_xG=False,
    xG_adjustment_strength=0.3,
    beta_lineup=None,
    lineup_dev_team=None,
    lineup_dev_opp=None,
):
    """Predicted (lambda_team, lambda_opp) goal rates for one match, where
    `team` is always the home side (matches how training data is built —
    prepare_model_data keeps home-perspective rows only). Every argument
    accepts either a scalar (e.g. posterior means, one prediction) or a
    numpy array (e.g. full posterior samples, vectorized over draws) —
    plain elementwise numpy ops, so both broadcast the same way.

    Mirrors build_model's theta_home/theta_away + soft_clip + exp exactly —
    same clip_theta value the model was actually trained with must be
    passed in, not assumed, to avoid silently drifting from it.
    """
    theta_team = compute_theta(
        attack_team, defense_opp, home_adv_team, is_home=1.0,
        beta_xG=beta_xG, xG_for=xG_team,
        use_opponent_adjusted_xG=use_opponent_adjusted_xG,
        xG_adjustment_strength=xG_adjustment_strength,
        beta_lineup=beta_lineup, lineup_dev_for=lineup_dev_team,
    )
    theta_opp = compute_theta(
        attack_opp, defense_team, home_adv_for=0.0, is_home=0.0,
        beta_xG=beta_xG, xG_for=xG_opp,
        use_opponent_adjusted_xG=use_opponent_adjusted_xG,
        xG_adjustment_strength=xG_adjustment_strength,
        beta_lineup=beta_lineup, lineup_dev_for=lineup_dev_opp,
    )
    theta_team = soft_clip(theta_team, clip_theta)
    theta_opp = soft_clip(theta_opp, clip_theta)
    return np.exp(theta_team), np.exp(theta_opp)


def predict_rows(
    model_data,
    row_indices,
    attack,
    defense,
    home_adv,
    clip_theta,
    *,
    beta_xG=None,
    use_opponent_adjusted_xG=False,
    xG_adjustment_strength=0.3,
    beta_lineup=None,
    max_t=None,
):
    """Predicted (lambda_home, lambda_away) for a batch of matches, reading
    every index and xG baseline straight off `model_data` — never rebuild a
    team->id mapping or a (t, team, opp) lookup by hand for this. That's
    exactly how two separate bugs made it into the CV harness: a missing xG
    term, then a pair of teams silently swapped between two independently
    computed mappings.

    `model_data` must be a ModelData from prepare_model_data covering (at
    least) the rows being predicted, built from the SAME underlying data
    `attack`/`defense`/`home_adv` were fit on — so `team_idx`/`opp_idx` here
    index the same team axis those posterior arrays do. Pass the ModelData
    from a `prepare_model_data(df, max_round=<last round you need xG/goals
    for>)` call; its team_mapping is computed from the full team/opp_team
    columns before any max_round filtering, so it's identical regardless of
    which max_round you pass — safe to reuse for both training and
    later/test rounds of the same underlying df.

    `row_indices`: positions into model_data's arrays (e.g.
    `np.where(model_data.t_idx <= some_round)[0]`), not row labels.

    `attack`/`defense`: (n_time, n_teams) posterior means, or (n_samples,
    n_time, n_teams) — pass `attack[:, t, :]`-sliced arrays per row yourself
    if you need full posterior samples per match; this function's fancy
    indexing is written for the 2D (means) case used by CV, matching how
    run_cv_window.py used to hand-loop this row by row.

    `max_t`: caps every selected row's t_idx at this value before indexing
    into attack/defense — use the last trained time index when predicting
    rounds beyond the trained window (latent state hasn't moved past
    training's end), e.g. CV test rounds. Leave None to use each row's own
    t_idx (e.g. predicting within the trained range).

    Returns (lambda_home, lambda_away, row_indices) — the row_indices are
    handed back so callers can index goals_home/goals_away/etc. off the same
    model_data with the same positions, instead of re-deriving them.
    """
    idx = np.asarray(row_indices)
    t = model_data.t_idx[idx].astype(int)
    if max_t is not None:
        t = np.minimum(t, max_t)
    team = model_data.team_idx[idx].astype(int)
    opp = model_data.opp_idx[idx].astype(int)

    lambda_home, lambda_away = predict_match_lambdas(
        attack_team=attack[t, team],
        defense_team=defense[t, team],
        attack_opp=attack[t, opp],
        defense_opp=defense[t, opp],
        home_adv_team=home_adv[team],
        clip_theta=clip_theta,
        beta_xG=beta_xG,
        xG_team=model_data.xG_home[idx] if beta_xG is not None else None,
        xG_opp=model_data.xG_away[idx] if beta_xG is not None else None,
        use_opponent_adjusted_xG=use_opponent_adjusted_xG,
        xG_adjustment_strength=xG_adjustment_strength,
        beta_lineup=beta_lineup,
        lineup_dev_team=model_data.lineup_dev_home[idx] if beta_lineup is not None else None,
        lineup_dev_opp=model_data.lineup_dev_away[idx] if beta_lineup is not None else None,
    )
    return lambda_home, lambda_away, idx


def dixon_coles_tau(lambda_home, lambda_away, goals_home, goals_away, rho):
    """Numpy mirror of football_model.model.components.dixon_coles_tau.

    Equals 1 everywhere except the four low-scoring cells (0-0, 1-0, 0-1,
    1-1), where it nudges the joint probability to correct for home/away
    goal correlation an independent Poisson likelihood can't represent.
    Broadcasts elementwise — pass scalars for one match, or same-shaped
    arrays (e.g. a whole test window, or a scoreline grid's row/col indices)
    for many at once.
    """
    lambda_home, lambda_away, goals_home, goals_away = np.broadcast_arrays(
        np.asarray(lambda_home, dtype=float), np.asarray(lambda_away, dtype=float),
        np.asarray(goals_home), np.asarray(goals_away),
    )
    is_00 = (goals_home == 0) & (goals_away == 0)
    is_10 = (goals_home == 1) & (goals_away == 0)
    is_01 = (goals_home == 0) & (goals_away == 1)
    is_11 = (goals_home == 1) & (goals_away == 1)

    tau = np.ones_like(lambda_home, dtype=float)
    tau = np.where(is_00, 1 - lambda_home * lambda_away * rho, tau)
    tau = np.where(is_10, 1 + lambda_home * rho, tau)
    tau = np.where(is_01, 1 + lambda_away * rho, tau)
    tau = np.where(is_11, 1 - rho, tau)
    return tau


def dixon_coles_log_correction(lambda_home, lambda_away, goals_home, goals_away, rho):
    """log(tau), clamped exactly like training's pm.Potential
    (`pt.maximum(tau, 1e-6)` in dixon_coles_adjustment) so log() stays
    finite. This is the term that must be ADDED to independent-Poisson
    log-likelihood for a match to match what build_model actually scores
    when use_dixon_coles=True — 0 (log(1)) outside the four low-score
    cells, so it's always safe to add even for a match that isn't one of
    them.
    """
    tau = dixon_coles_tau(lambda_home, lambda_away, goals_home, goals_away, rho)
    return np.log(np.maximum(tau, 1e-6))


def dc_outcome_probs(lambda_home, lambda_away, rho=None, max_goals=10):
    """Home/draw/away win probabilities from Poisson goal-rate parameters,
    via exact convolution over the scoreline grid — optionally with the
    Dixon-Coles correction applied to the four cells it affects before
    renormalizing.

    Pass rho=None (the default) for the plain independent-Poisson
    probabilities, exactly matching a model trained with
    use_dixon_coles=False. Pass the trained model's rho_dc (a posterior
    mean, same pattern as attack/defense/home_adv elsewhere in this module)
    to get the probabilities the model actually implies when
    use_dixon_coles=True — using plain independent-Poisson probabilities
    here regardless of rho is exactly the evaluation gap this function
    exists to close.
    """
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals, lambda_home)
    pa = poisson.pmf(goals, lambda_away)
    grid = np.outer(ph, pa)  # grid[i, j] = P(home scores i, away scores j)

    if rho is not None:
        goals_h_grid, goals_a_grid = np.meshgrid(goals, goals, indexing="ij")
        tau_grid = dixon_coles_tau(lambda_home, lambda_away, goals_h_grid, goals_a_grid, rho)
        grid = grid * np.maximum(tau_grid, 0.0)  # keep grid entries non-negative

    p_home = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, 1).sum()
    total = p_home + p_draw + p_away  # <1 due to truncation at max_goals; renormalize
    return p_home / total, p_draw / total, p_away / total
