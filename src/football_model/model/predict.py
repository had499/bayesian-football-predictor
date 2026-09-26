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
    intercept=0.0,
    absence_term=0.0,
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
    """
    theta = intercept + absence_term + attack_for - defense_against + home_adv_for * is_home

    if beta_xG is not None and xG_for is not None:
        if use_opponent_adjusted_xG:
            adj = np.clip(1.0 + xG_adjustment_strength * defense_against, 0.5, 1.5)
            beta_xG = beta_xG * adj
        theta = theta + beta_xG * np.log(xG_for + 0.01)

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
    intercept=0.0,
    absence_team=0.0,
    absence_opp=0.0,
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
        intercept=intercept, absence_term=absence_team,
    )
    theta_opp = compute_theta(
        attack_opp, defense_team, home_adv_for=0.0, is_home=0.0,
        beta_xG=beta_xG, xG_for=xG_opp,
        use_opponent_adjusted_xG=use_opponent_adjusted_xG,
        xG_adjustment_strength=xG_adjustment_strength,
        intercept=intercept, absence_term=absence_opp,
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
    max_t=None,
    intercept=0.0,
    beta_absence=None,
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
        intercept=intercept,
        absence_team=(beta_absence * model_data.absent_home[idx]) if beta_absence is not None else 0.0,
        absence_opp=(beta_absence * model_data.absent_away[idx]) if beta_absence is not None else 0.0,
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
    # Dixon & Coles (1997) eq. 4: the correction for a 1-0 uses the AWAY rate
    # (the side that scored 0), and for 0-1 the HOME rate — see
    # components.dixon_coles_tau's docstring. Swapped before WP007.
    tau = np.where(is_10, 1 + lambda_away * rho, tau)
    tau = np.where(is_01, 1 + lambda_home * rho, tau)
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


def scoreline_grid(lambda_home, lambda_away, rho=None, max_goals=10):
    """Joint scoreline probabilities P(home=i, away=j) on a (max_goals+1)^2
    grid, optionally Dixon-Coles corrected, renormalised over the grid.

    lambda_home / lambda_away / rho broadcast against each other, so this
    covers a single match (scalars -> (G, G)), a batch of matches
    ((M,) -> (M, G, G)) or full posterior draws ((D, M) -> (D, M, G, G)).
    `rho` may be None (plain independent Poisson), a scalar, or an array
    that broadcasts with the lambdas (e.g. (D, 1) for one rho per draw).
    """
    goals = np.arange(max_goals + 1)
    lh = np.asarray(lambda_home, dtype=float)[..., None, None]
    la = np.asarray(lambda_away, dtype=float)[..., None, None]
    ph = poisson.pmf(goals[:, None], lh)     # (..., G, 1) — home goals i
    pa = poisson.pmf(goals[None, :], la)     # (..., 1, G) — away goals j
    grid = ph * pa

    if rho is not None:
        rho_b = np.asarray(rho, dtype=float)
        # give rho the same trailing (1, 1) grid axes as the lambdas
        rho_b = rho_b.reshape(rho_b.shape + (1, 1)) if rho_b.ndim else rho_b
        tau = dixon_coles_tau(lh, la, goals[:, None], goals[None, :], rho_b)
        grid = grid * np.maximum(tau, 0.0)  # keep grid entries non-negative

    return grid / grid.sum(axis=(-2, -1), keepdims=True)  # <1 mass from truncation / DC; renormalise


def grid_outcome_probs(grid):
    """(..., G, G) scoreline grid -> (..., 3) home/draw/away probabilities."""
    p_home = np.tril(grid, -1).sum(axis=(-2, -1))
    p_draw = np.trace(grid, axis1=-2, axis2=-1)
    p_away = np.triu(grid, 1).sum(axis=(-2, -1))
    return np.stack([p_home, p_draw, p_away], axis=-1)


def dc_outcome_probs(lambda_home, lambda_away, rho=None, max_goals=10):
    """Home/draw/away win probabilities from Poisson goal-rate parameters,
    via exact convolution over the scoreline grid — optionally with the
    Dixon-Coles correction applied to the four cells it affects before
    renormalizing.

    Pass rho=None (the default) for the plain independent-Poisson
    probabilities, exactly matching a model trained with
    use_dixon_coles=False. Pass the trained model's rho_dc (a posterior
    mean, same pattern as attack/defence/home_adv elsewhere in this module)
    to get the probabilities the model actually implies when
    use_dixon_coles=True — using plain independent-Poisson probabilities
    here regardless of rho is exactly the evaluation gap this function
    exists to close.
    """
    p = grid_outcome_probs(scoreline_grid(lambda_home, lambda_away, rho, max_goals))
    return p[..., 0], p[..., 1], p[..., 2]


# ---------------------------------------------------------------------------
# Posterior-averaged forecasting (WP007)
#
# scripts/run_cv_window.py used to score every test match from posterior
# MEANS (one lambda per side, exp() of an averaged theta) at the frozen last
# trained latent state. That throws away two things a Bayesian model has:
#   1. parameter uncertainty — E[exp(theta)] != exp(E[theta]), and the
#      outcome probability worth scoring is the posterior AVERAGE of the
#      per-draw probabilities, not the probability at the average draw;
#   2. forecast-horizon uncertainty — a round beyond the trained window is
#      one more step of the latent process (variance sigma^2, times the
#      season-start multiplier at a season opener, plus a promoted side's
#      entry shift), which "freeze at the last trained value" ignores.
# The functions below do both, reading the same posterior variable names
# build_model registers, so they can't drift from it independently.
# ---------------------------------------------------------------------------
def forecast_latent(
    posterior,
    *,
    last_t,
    target_ts,
    active_mask,
    season_start_mask,
    season_start_sigma_mult,
    rng,
    active_last=None,
):
    """Roll the trained latent attack/defence state forward from `last_t`
    to each round in `target_ts`, sampling the process's own innovations.

    posterior: dict of numpy arrays, leading axis = draw. Reads "attack" and
        "defence" (or, if build_model hard-centred them, "attack_raw" /
        "defence_raw", which it also registers), plus whichever of sigma_att /
        sigma_def (or sigma_*_team), rho_att / rho_def, promo_att / promo_def,
        att_0 / def_0 exist — absent means the model didn't have it
        (rho absent -> random walk, rho=1; promo absent -> no entry shift).
    active_mask / season_start_mask: FULL-horizon arrays (index up to
        max(target_ts)), e.g. from prepare_model_data(max_round=test_end).
    active_last: (n_teams,) activity at `last_t` as the TRAINED model saw it
        (train ModelData.active_mask[last_t]); defaults to active_mask[last_t].
        A team is "entering" at step k when it's active at k but wasn't at k-1.

    Returns {t: (attack (D, K), defence (D, K))} — strengths at round t,
    re-centred over that round's active set iff the model was hard-centred.
    """
    centred = "attack_raw" in posterior
    out = {}
    D = next(iter(posterior.values())).shape[0]
    targets = sorted(set(int(t) for t in target_ts))
    max_t = max(targets) if targets else last_t

    state, params = {}, {}
    for side, short in (("attack", "att"), ("defence", "def")):
        raw = posterior[f"{side}_raw"] if centred else posterior[side]
        base = posterior.get(f"{short}_0")               # (D, K) static level, or None
        latent = raw[:, last_t, :]
        # ar1_static's AR(1) part reverts toward 0 around the static att_0, so
        # roll the deviation forward and add the static level back afterwards.
        state[short] = latent - base if base is not None else latent
        sig = posterior.get(f"sigma_{short}_team", posterior.get(f"sigma_{short}"))
        sig = np.asarray(sig)
        params[short] = dict(
            base=base,
            sigma=sig if sig.ndim == 2 else sig[:, None],                 # (D, K) or (D, 1)
            rho=np.asarray(posterior[f"rho_{short}"])[:, None] if f"rho_{short}" in posterior else 1.0,
            shift=np.asarray(posterior[f"promo_{short}"])[:, None] if f"promo_{short}" in posterior else 0.0,
        )

    prev_active = np.asarray(active_mask[last_t] if active_last is None else active_last, dtype=float)
    for k in range(last_t + 1, max_t + 1):
        act_k = np.asarray(active_mask[k], dtype=float)                   # (K,)
        entry_k = act_k * (1.0 - prev_active)
        mult = 1.0 + (season_start_sigma_mult - 1.0) * float(season_start_mask[k])
        for short in ("att", "def"):
            pr = params[short]
            z = rng.standard_normal(state[short].shape)
            new = pr["rho"] * state[short] + pr["sigma"] * mult * z + entry_k[None, :] * pr["shift"]
            state[short] = np.where(act_k[None, :] > 0, new, state[short])
        prev_active = act_k
        if k in targets:
            att = state["att"] + (params["att"]["base"] if params["att"]["base"] is not None else 0.0)
            dfn = state["def"] + (params["def"]["base"] if params["def"]["base"] is not None else 0.0)
            if centred:
                n_act = max(act_k.sum(), 1.0)
                att = att - (att * act_k[None, :]).sum(1, keepdims=True) / n_act
                dfn = dfn - (dfn * act_k[None, :]).sum(1, keepdims=True) / n_act
            out[k] = (att, dfn)
    return out


def posterior_lambdas(
    model_data,
    row_indices,
    posterior,
    clip_theta,
    *,
    last_t,
    season_start_sigma_mult,
    n_draws=400,
    seed=0,
    propagate=True,
    active_last=None,
):
    """Per-draw (lambda_home, lambda_away), shape (D, M), for `row_indices`
    of `model_data` (built with max_round covering them).

    Rows at t <= last_t read the trained latent state directly. Rows beyond
    it are forecast with `forecast_latent` (propagate=True, the honest
    version) or held at the last trained value (propagate=False — the old
    harness behaviour, kept so a comparison can isolate its effect).
    Reads intercept / beta_xG / home_adv from `posterior` if present, exactly
    as build_model's theta does.
    """
    idx = np.asarray(row_indices)
    D_all = next(iter(posterior.values())).shape[0]
    keep = np.linspace(0, D_all - 1, min(n_draws, D_all)).astype(int)
    post = {k: np.asarray(v)[keep] for k, v in posterior.items()}
    D = len(keep)

    t = model_data.t_idx[idx].astype(int)
    h = model_data.team_idx[idx].astype(int)
    a = model_data.opp_idx[idx].astype(int)
    rng = np.random.default_rng(seed)

    beyond = t > last_t
    fc = {}
    if propagate and beyond.any():
        fc = forecast_latent(
            post, last_t=last_t, target_ts=t[beyond],
            active_mask=model_data.active_mask,
            season_start_mask=model_data.season_start_mask,
            season_start_sigma_mult=season_start_sigma_mult,
            rng=rng, active_last=active_last,
        )

    M = len(idx)
    att_h = np.empty((D, M)); att_a = np.empty((D, M)); def_h = np.empty((D, M)); def_a = np.empty((D, M))
    for j in range(M):
        if t[j] <= last_t:
            A, Df = post["attack"][:, t[j], :], post["defence"][:, t[j], :]
        elif t[j] in fc:
            A, Df = fc[t[j]]
        else:  # propagate=False: hold at last trained state
            A, Df = post["attack"][:, last_t, :], post["defence"][:, last_t, :]
        att_h[:, j], att_a[:, j] = A[:, h[j]], A[:, a[j]]
        def_h[:, j], def_a[:, j] = Df[:, h[j]], Df[:, a[j]]

    b0 = post["intercept"][:, None] if "intercept" in post else 0.0
    if "beta_absence" in post:
        ba = post["beta_absence"][:, None]
        abs_h = ba * model_data.absent_home[idx][None, :]
        abs_a = ba * model_data.absent_away[idx][None, :]
    else:
        abs_h = abs_a = 0.0
    home_adv = post["home_adv"][:, h]                                       # (D, M)
    beta = post["beta_xG"][:, None] if "beta_xG" in post else None
    lam_h, lam_a = predict_match_lambdas(
        attack_team=att_h, defense_team=def_h, attack_opp=att_a, defense_opp=def_a,
        home_adv_team=home_adv, clip_theta=clip_theta, intercept=b0,
        absence_team=abs_h, absence_opp=abs_a,
        beta_xG=beta,
        xG_team=model_data.xG_home[idx][None, :] if beta is not None else None,
        xG_opp=model_data.xG_away[idx][None, :] if beta is not None else None,
    )
    return lam_h, lam_a


def posterior_outcome_probs(lam_h, lam_a, rho_dc=None, max_goals=12):
    """(D, M) per-draw lambdas -> (M, 3) home/draw/away probabilities, the
    posterior AVERAGE of the per-draw (Dixon-Coles-corrected) probabilities.
    rho_dc: None, or (D,) per-draw rho samples."""
    rho = None if rho_dc is None else np.asarray(rho_dc, dtype=float)[:, None]
    per_draw = grid_outcome_probs(scoreline_grid(lam_h, lam_a, rho, max_goals))  # (D, M, 3)
    p = per_draw.mean(axis=0)
    return p / p.sum(axis=1, keepdims=True)


def posterior_goals_loglik(lam_h, lam_a, goals_home, goals_away, rho_dc=None, max_goals=12):
    """Per-match log posterior-predictive probability of the observed
    scoreline: log of the draw-average of P(gh, ga | draw), with the
    Dixon-Coles cell factor included when rho_dc is given. (M,) array."""
    rho = None if rho_dc is None else np.asarray(rho_dc, dtype=float)[:, None]
    grid = scoreline_grid(lam_h, lam_a, rho, max_goals)                      # (D, M, G, G)
    gh = np.minimum(np.asarray(goals_home, int), max_goals)
    ga = np.minimum(np.asarray(goals_away, int), max_goals)
    cell = grid[:, np.arange(len(gh)), gh, ga]                              # (D, M)
    return np.log(np.maximum(cell.mean(axis=0), 1e-12))


def ranked_probability_score(probs, outcome):
    """RPS for (M, 3) home/draw/away probabilities and outcome codes
    0=home, 1=draw, 2=away. Lower is better; (M,) array."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), np.asarray(outcome, int)] = 1.0
    return 0.5 * ((np.cumsum(probs, 1)[:, :2] - np.cumsum(onehot, 1)[:, :2]) ** 2).sum(1)
