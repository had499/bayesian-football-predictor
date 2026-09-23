import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from football_model.model.components import dixon_coles_tau as dixon_coles_tau_pt
from football_model.model.model import build_model
from football_model.model.predict import (
    compute_theta,
    dc_outcome_probs,
    dixon_coles_log_correction,
    dixon_coles_tau,
    forecast_latent,
    grid_outcome_probs,
    posterior_goals_loglik,
    posterior_lambdas,
    posterior_outcome_probs,
    predict_match_lambdas,
    predict_rows,
    ranked_probability_score,
    scoreline_grid,
    soft_clip,
)
from football_model.types.model_data import ModelConfig


def test_compute_theta_basic_arithmetic():
    theta_home = compute_theta(attack_for=0.5, defense_against=0.2, home_adv_for=0.1, is_home=1.0)
    assert np.isclose(theta_home, 0.5 - 0.2 + 0.1)

    theta_away = compute_theta(attack_for=0.5, defense_against=0.2, home_adv_for=0.1, is_home=0.0)
    assert np.isclose(theta_away, 0.5 - 0.2)  # home_adv zeroed out when is_home=0


def test_compute_theta_with_xg():
    theta = compute_theta(
        attack_for=0.0, defense_against=0.0, home_adv_for=0.0, is_home=0.0,
        beta_xG=0.9, xG_for=1.5,
    )
    assert np.isclose(theta, 0.9 * np.log(1.5 + 0.01))


def test_compute_theta_opponent_adjusted_xg_clips():
    # adjustment factor must stay within [0.5, 1.5] even for extreme defense values
    theta_extreme_pos = compute_theta(
        0.0, 100.0, 0.0, 0.0, beta_xG=1.0, xG_for=1.0,
        use_opponent_adjusted_xG=True, xG_adjustment_strength=0.3,
    )
    theta_extreme_neg = compute_theta(
        0.0, -100.0, 0.0, 0.0, beta_xG=1.0, xG_for=1.0,
        use_opponent_adjusted_xG=True, xG_adjustment_strength=0.3,
    )
    # both should match the clipped-adjustment (1.5x and 0.5x beta_xG) result,
    # not blow up from the unclipped 1 + 0.3*(+-100) adjustment factor
    assert np.isclose(theta_extreme_pos - (-100.0), 1.5 * 1.0 * np.log(1.01))
    assert np.isclose(theta_extreme_neg - 100.0, 0.5 * 1.0 * np.log(1.01))


def test_soft_clip_matches_components_version():
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    y = soft_clip(x, limit=2.0)
    assert np.all(np.abs(y) <= 2.0)
    assert y[2] == 0.0


def test_predict_match_lambdas_matches_build_model_no_xg(two_season_model_data):
    """The real cross-check: values fed into predict_match_lambdas come from
    an actual prior-predictive draw of build_model, and the output must
    match that same draw's own lambda_home/lambda_away Deterministics
    exactly — proving the numpy mirror really does compute the same formula
    as the PyTensor model, not just something plausible-looking."""
    md = two_season_model_data
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True)
    model = build_model(md, config)

    with model:
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    attack = idata.prior["attack"].values[0, 0]
    defence = idata.prior["defence"].values[0, 0]
    home_adv = idata.prior["home_adv"].values[0, 0]
    lambda_home_model = idata.prior["lambda_home"].values[0, 0]
    lambda_away_model = idata.prior["lambda_away"].values[0, 0]

    for row in [0, 5, min(20, len(md.t_idx) - 1)]:
        t, team, opp = int(md.t_idx[row]), int(md.team_idx[row]), int(md.opp_idx[row])

        lam_team, lam_opp = predict_match_lambdas(
            attack_team=attack[t, team],
            defense_team=defence[t, team],
            attack_opp=attack[t, opp],
            defense_opp=defence[t, opp],
            home_adv_team=home_adv[team],
            clip_theta=config.clip_theta,
        )
        assert np.isclose(lam_team, lambda_home_model[row], rtol=1e-5)
        assert np.isclose(lam_opp, lambda_away_model[row], rtol=1e-5)


def test_predict_rows_matches_build_model_batched(two_season_model_data):
    """predict_rows is what run_cv_window.py now calls instead of hand-
    looping team/time lookups — must match build_model's own lambda_home/
    lambda_away exactly, for a whole batch of rows at once, reading indices
    straight off ModelData (no separately-rebuilt team mapping)."""
    md = two_season_model_data
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True)
    model = build_model(md, config)

    with model:
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    attack = idata.prior["attack"].values[0, 0]
    defence = idata.prior["defence"].values[0, 0]
    home_adv = idata.prior["home_adv"].values[0, 0]
    lambda_home_model = idata.prior["lambda_home"].values[0, 0]
    lambda_away_model = idata.prior["lambda_away"].values[0, 0]

    rows = np.array([0, 5, min(20, len(md.t_idx) - 1)])
    lam_home, lam_away, returned_rows = predict_rows(
        md, rows, attack=attack, defense=defence, home_adv=home_adv,
        clip_theta=config.clip_theta,
    )
    assert np.array_equal(returned_rows, rows)
    assert np.allclose(lam_home, lambda_home_model[rows], rtol=1e-5)
    assert np.allclose(lam_away, lambda_away_model[rows], rtol=1e-5)


def test_predict_rows_max_t_freezes_latent_state(two_season_model_data):
    """max_t must cap every row's t_idx before indexing attack/defense —
    used to predict rounds beyond the trained window (e.g. CV test rounds)
    by holding latent state at its last trained value."""
    md = two_season_model_data
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True)
    model = build_model(md, config)

    with model:
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    attack = idata.prior["attack"].values[0, 0]
    defence = idata.prior["defence"].values[0, 0]
    home_adv = idata.prior["home_adv"].values[0, 0]

    row = np.array([int(np.argmax(md.t_idx))])  # a row at the latest t_idx
    frozen_t = 0
    lam_home_frozen, lam_away_frozen, _ = predict_rows(
        md, row, attack=attack, defense=defence, home_adv=home_adv,
        clip_theta=config.clip_theta, max_t=frozen_t,
    )

    team, opp = int(md.team_idx[row[0]]), int(md.opp_idx[row[0]])
    lam_home_expected, lam_away_expected = predict_match_lambdas(
        attack_team=attack[frozen_t, team], defense_team=defence[frozen_t, team],
        attack_opp=attack[frozen_t, opp], defense_opp=defence[frozen_t, opp],
        home_adv_team=home_adv[team], clip_theta=config.clip_theta,
    )
    assert np.isclose(lam_home_frozen[0], lam_home_expected)
    assert np.isclose(lam_away_frozen[0], lam_away_expected)


def test_predict_match_lambdas_matches_build_model_with_xg(two_season_model_data):
    md = two_season_model_data
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True, use_xG=True)
    model = build_model(md, config)

    with model:
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    attack = idata.prior["attack"].values[0, 0]
    defence = idata.prior["defence"].values[0, 0]
    home_adv = idata.prior["home_adv"].values[0, 0]
    beta_xG = float(idata.prior["beta_xG"].values[0, 0])
    lambda_home_model = idata.prior["lambda_home"].values[0, 0]
    lambda_away_model = idata.prior["lambda_away"].values[0, 0]

    for row in [0, 5, min(20, len(md.t_idx) - 1)]:
        t, team, opp = int(md.t_idx[row]), int(md.team_idx[row]), int(md.opp_idx[row])

        lam_team, lam_opp = predict_match_lambdas(
            attack_team=attack[t, team],
            defense_team=defence[t, team],
            attack_opp=attack[t, opp],
            defense_opp=defence[t, opp],
            home_adv_team=home_adv[team],
            clip_theta=config.clip_theta,
            beta_xG=beta_xG,
            xG_team=md.xG_home[row],
            xG_opp=md.xG_away[row],
        )
        assert np.isclose(lam_team, lambda_home_model[row], rtol=1e-5)
        assert np.isclose(lam_opp, lambda_away_model[row], rtol=1e-5)


def test_dixon_coles_tau_matches_pytensor_version():
    """The numpy mirror must match football_model.model.components'
    actual PyTensor tau exactly — same cross-check pattern as
    predict_match_lambdas vs. build_model's real lambda_home/lambda_away."""
    lh, la = pt.dscalar("lh"), pt.dscalar("la")
    gh, ga = pt.iscalar("gh"), pt.iscalar("ga")
    rho = pt.dscalar("rho")
    tau_sym = dixon_coles_tau_pt(lh, la, gh, ga, rho)
    f = pytensor.function([lh, la, gh, ga, rho], tau_sym)

    cases = [
        (1.2, 0.8, 0, 0, 0.1),
        (1.2, 0.8, 1, 0, 0.1),
        (1.2, 0.8, 0, 1, -0.05),
        (1.2, 0.8, 1, 1, 0.2),
        (1.2, 0.8, 2, 2, 0.1),   # outside the 4 cells -> tau == 1
        (0.5, 3.0, 0, 0, -0.15),
    ]
    for lh_v, la_v, gh_v, ga_v, rho_v in cases:
        expected = f(lh_v, la_v, gh_v, ga_v, rho_v)
        actual = dixon_coles_tau(lh_v, la_v, gh_v, ga_v, rho_v)
        assert np.isclose(actual, expected), (lh_v, la_v, gh_v, ga_v, rho_v)


def test_dixon_coles_tau_broadcasts_over_arrays():
    lambda_home = np.array([1.2, 1.2, 0.5])
    lambda_away = np.array([0.8, 0.8, 3.0])
    goals_home = np.array([0, 2, 0])
    goals_away = np.array([0, 2, 0])
    tau = dixon_coles_tau(lambda_home, lambda_away, goals_home, goals_away, rho=0.1)
    assert tau.shape == (3,)
    assert np.isclose(tau[0], 1 - 1.2 * 0.8 * 0.1)   # (0,0) cell
    assert tau[1] == 1.0                              # outside the 4 cells
    assert np.isclose(tau[2], 1 - 0.5 * 3.0 * 0.1)    # (0,0) cell again


def test_dixon_coles_log_correction_is_log_of_clamped_tau():
    # A (0,0) match with an extreme rho drives tau negative; the correction
    # must clamp before taking log, same as training's pm.Potential does.
    lam_h, lam_a, rho = 5.0, 5.0, 10.0  # tau = 1 - 25*10, deeply negative
    corr = dixon_coles_log_correction(lam_h, lam_a, 0, 0, rho)
    assert np.isfinite(corr)
    assert np.isclose(corr, np.log(1e-6))

    # Outside the 4 cells, tau == 1 so the correction is exactly 0.
    assert dixon_coles_log_correction(1.2, 0.8, 3, 2, rho=0.5) == 0.0


def test_dc_outcome_probs_rho_none_matches_rho_zero():
    # rho=0 makes tau==1 everywhere, i.e. identical to the plain
    # independent-Poisson probabilities rho=None already gives.
    p_none = dc_outcome_probs(1.4, 1.1, rho=None)
    p_zero = dc_outcome_probs(1.4, 1.1, rho=0.0)
    assert np.allclose(p_none, p_zero)
    assert np.isclose(sum(p_none), 1.0)


def test_dc_outcome_probs_rho_shifts_draw_probability():
    # Positive rho at these lambdas should change P(draw) relative to the
    # uncorrected baseline — proving rho actually reaches the returned
    # probabilities, not just the (0,0)/(1,1) grid cells in isolation.
    p_home_0, p_draw_0, p_away_0 = dc_outcome_probs(1.4, 1.1, rho=0.0)
    p_home_r, p_draw_r, p_away_r = dc_outcome_probs(1.4, 1.1, rho=0.15)
    assert not np.isclose(p_draw_0, p_draw_r)
    for probs in [(p_home_0, p_draw_0, p_away_0), (p_home_r, p_draw_r, p_away_r)]:
        assert np.isclose(sum(probs), 1.0)


def test_compute_theta_intercept_shifts_both_sides():
    base = compute_theta(0.3, 0.1, 0.2, is_home=1.0)
    assert np.isclose(compute_theta(0.3, 0.1, 0.2, is_home=1.0, intercept=0.25), base + 0.25)
    lam_h0, lam_a0 = predict_match_lambdas(0.3, 0.1, 0.2, 0.0, 0.15, clip_theta=1e6)  # wide clip: tanh ~ identity
    lam_h1, lam_a1 = predict_match_lambdas(0.3, 0.1, 0.2, 0.0, 0.15, clip_theta=1e6, intercept=0.25)
    assert np.isclose(lam_h1 / lam_h0, np.exp(0.25)) and np.isclose(lam_a1 / lam_a0, np.exp(0.25))


def test_numpy_tau_keeps_joint_normalised():
    from scipy.stats import poisson

    g = np.arange(0, 40)
    for lam, mu, rho in [(2.4, 0.7, 0.05), (1.0, 1.8, -0.07)]:
        gh, ga = np.meshgrid(g, g, indexing="ij")
        joint = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu)) * dixon_coles_tau(lam, mu, gh, ga, rho)
        assert np.isclose(joint.sum(), 1.0, atol=1e-10)


def test_scoreline_grid_shapes_and_broadcasting_match_scalar_path():
    lh, la = np.array([[1.4, 0.9], [1.6, 1.1]]), np.array([[1.1, 1.3], [0.8, 1.0]])   # (D=2, M=2)
    rho = np.array([0.05, -0.05])
    grid = scoreline_grid(lh, la, rho[:, None], max_goals=10)
    assert grid.shape == (2, 2, 11, 11)
    assert np.allclose(grid.sum(axis=(-2, -1)), 1.0)
    for d in range(2):
        for m in range(2):
            ph, pd_, pa = dc_outcome_probs(lh[d, m], la[d, m], rho=rho[d], max_goals=10)
            assert np.allclose(grid_outcome_probs(grid[d, m]), [ph, pd_, pa])


def _fake_posterior(D=6, T=4, K=3, seed=0, with_promo=False, rho=None):
    rng = np.random.default_rng(seed)
    post = dict(
        attack=rng.normal(0, 0.2, (D, T, K)), defence=rng.normal(0, 0.2, (D, T, K)),
        home_adv=rng.normal(0.2, 0.05, (D, K)),
        sigma_att=np.full(D, 0.05), sigma_def=np.full(D, 0.05),
    )
    if with_promo:
        post["promo_att"], post["promo_def"] = np.full(D, -0.4), np.full(D, -0.3)
    if rho is not None:
        post["rho_att"] = post["rho_def"] = np.full(D, rho)
    return post


def test_forecast_latent_freezes_inactive_teams_and_is_deterministic_with_zero_sigma():
    post = _fake_posterior(with_promo=False)
    post["sigma_att"] = post["sigma_def"] = np.zeros(6)
    T_full = 7
    active = np.ones((T_full, 3)); active[4:, 2] = 0.0        # team 2 leaves after t=3
    out = forecast_latent(post, last_t=3, target_ts=[5, 6], active_mask=active,
                          season_start_mask=np.zeros(T_full), season_start_sigma_mult=1.0,
                          rng=np.random.default_rng(0))
    for t in (5, 6):
        att, dfn = out[t]
        assert np.allclose(att, post["attack"][:, 3, :])      # sigma=0, rho=1 -> unchanged
        assert np.allclose(dfn, post["defence"][:, 3, :])


def test_forecast_latent_variance_grows_with_horizon_and_season_start_multiplier():
    D = 4000
    post = dict(attack=np.zeros((D, 3, 2)), defence=np.zeros((D, 3, 2)),
                sigma_att=np.full(D, 0.1), sigma_def=np.full(D, 0.1))
    T_full = 8
    ss = np.zeros(T_full); ss[6] = 1.0                        # t=6 is a season opener
    out = forecast_latent(post, last_t=2, target_ts=[3, 5, 6], active_mask=np.ones((T_full, 2)),
                          season_start_mask=ss, season_start_sigma_mult=3.0,
                          rng=np.random.default_rng(1))
    sd = {t: out[t][0].std() for t in (3, 5, 6)}
    assert np.isclose(sd[3], 0.1, rtol=0.1)                            # 1 step
    assert np.isclose(sd[5], 0.1 * np.sqrt(3), rtol=0.1)               # 3 steps
    assert np.isclose(sd[6], 0.1 * np.sqrt(3 + 9 - 1), rtol=0.1)       # + a 3x-sigma step at t=6


def test_forecast_latent_applies_promotion_shift_at_entry_only():
    D = 500
    post = dict(attack=np.zeros((D, 3, 2)), defence=np.zeros((D, 3, 2)),
                sigma_att=np.zeros(D), sigma_def=np.zeros(D),
                promo_att=np.full(D, -0.4), promo_def=np.full(D, -0.3))
    T_full = 6
    active = np.ones((T_full, 2)); active[:4, 1] = 0.0        # team 1 absent up to t=3, enters t=4
    out = forecast_latent(post, last_t=2, target_ts=[3, 4, 5], active_mask=active,
                          season_start_mask=np.zeros(T_full), season_start_sigma_mult=1.0,
                          rng=np.random.default_rng(0), active_last=np.array([1.0, 0.0]))
    assert np.allclose(out[3][0][:, 1], 0.0)                  # still absent
    assert np.allclose(out[4][0][:, 1], -0.4) and np.allclose(out[4][1][:, 1], -0.3)
    assert np.allclose(out[5][0][:, 1], -0.4)                 # persists, not re-applied
    assert np.allclose(out[4][0][:, 0], 0.0)                  # team 0 untouched


def test_forecast_latent_recentres_only_when_model_was_hard_centred():
    D, K = 50, 4
    rng = np.random.default_rng(3)
    raw = rng.normal(0, 0.3, (D, 3, K))
    post = dict(attack_raw=raw, defence_raw=raw, attack=raw - raw.mean(2, keepdims=True),
                defence=raw - raw.mean(2, keepdims=True),
                sigma_att=np.zeros(D), sigma_def=np.zeros(D))
    out = forecast_latent(post, last_t=2, target_ts=[3], active_mask=np.ones((5, K)),
                          season_start_mask=np.zeros(5), season_start_sigma_mult=1.0,
                          rng=np.random.default_rng(0))
    assert np.allclose(out[3][0].mean(axis=1), 0.0, atol=1e-12)


def test_posterior_lambdas_in_sample_matches_predict_match_lambdas(two_season_model_data):
    md = two_season_model_data
    post = _fake_posterior(D=5, T=md.n_time, K=md.n_teams)
    post["intercept"] = np.full(5, 0.2)
    rows = np.array([0, 5, 12])
    lam_h, lam_a = posterior_lambdas(md, rows, post, 5.0, last_t=md.n_time - 1,
                                     season_start_sigma_mult=3.0, n_draws=5)
    assert lam_h.shape == (5, 3)
    t, h, a = md.t_idx[rows], md.team_idx[rows], md.opp_idx[rows]
    for d in range(5):
        exp_h, exp_a = predict_match_lambdas(
            post["attack"][d, t, h], post["defence"][d, t, h], post["attack"][d, t, a],
            post["defence"][d, t, a], post["home_adv"][d, h], 5.0, intercept=0.2)
        assert np.allclose(lam_h[d], exp_h) and np.allclose(lam_a[d], exp_a)


def test_posterior_lambdas_out_of_sample_propagation_widens_but_does_not_bias(two_season_model_data):
    md = two_season_model_data
    D = 800
    last_t = int(md.t_idx.max()) - 2
    rows = np.where(md.t_idx > last_t)[0]
    assert len(rows) > 0
    post = dict(attack=np.zeros((D, md.n_time, md.n_teams)), defence=np.zeros((D, md.n_time, md.n_teams)),
                home_adv=np.zeros((D, md.n_teams)), sigma_att=np.full(D, 0.15), sigma_def=np.full(D, 0.15))
    frozen = posterior_lambdas(md, rows, post, 5.0, last_t=last_t, season_start_sigma_mult=1.0,
                               n_draws=D, propagate=False)[0]
    prop = posterior_lambdas(md, rows, post, 5.0, last_t=last_t, season_start_sigma_mult=1.0,
                             n_draws=D, propagate=True)[0]
    assert np.allclose(frozen.std(axis=0), 0.0)               # frozen state: no spread at all
    assert (prop.std(axis=0) > 0.05).all()                    # propagated: real forecast uncertainty


def test_posterior_outcome_probs_sums_to_one_and_averages_probabilities_not_lambdas():
    lam_h = np.array([[0.6], [2.6]]); lam_a = np.array([[1.4], [0.6]])     # two draws, one match
    p = posterior_outcome_probs(lam_h, lam_a)
    assert p.shape == (1, 3) and np.isclose(p.sum(), 1.0)
    p_mean_draw = np.array(dc_outcome_probs(lam_h.mean(), lam_a.mean(), max_goals=12))
    p_avg = np.mean([dc_outcome_probs(lam_h[d, 0], lam_a[d, 0], max_goals=12) for d in range(2)], axis=0)
    assert np.allclose(p[0], p_avg)
    assert not np.allclose(p[0], p_mean_draw, atol=1e-3)      # Jensen gap is real, not noise


def test_posterior_goals_loglik_matches_independent_poisson_when_rho_none():
    from scipy.stats import poisson

    lam_h = np.array([[1.4, 0.9]]); lam_a = np.array([[1.1, 1.3]])
    ll = posterior_goals_loglik(lam_h, lam_a, [2, 0], [1, 0])
    expected = np.log(poisson.pmf([2, 0], lam_h[0])) + np.log(poisson.pmf([1, 0], lam_a[0]))
    assert np.allclose(ll, expected, atol=1e-6)               # grid truncation at 12 goals is negligible


def test_ranked_probability_score_known_values():
    p = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1/3, 1/3, 1/3]])
    r = ranked_probability_score(p, [0, 0, 1])
    assert np.isclose(r[0], 0.0)
    assert np.isclose(r[1], 1.0)                              # certain away, home won
    assert np.isclose(r[2], 0.5 * ((1/3 - 1) ** 2 * 0 + (1/3) ** 2 + (2/3 - 1) ** 2))
