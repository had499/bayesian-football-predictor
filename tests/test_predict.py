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
    predict_match_lambdas,
    predict_rows,
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


def test_predict_match_lambdas_matches_build_model_with_lineup_xg(two_season_model_data):
    """WP008: written before this term was ever wired into run_cv_window.py
    or the CV harness — the exact gap (a new theta term added to training,
    prediction never updated to match) caused two real bugs earlier in this
    project (missing xG, un-evaluated Dixon-Coles). Test-first here."""
    import dataclasses

    md = two_season_model_data
    rng = np.random.default_rng(0)
    lineup_dev_home = rng.normal(scale=0.3, size=len(md.t_idx))
    lineup_dev_away = rng.normal(scale=0.3, size=len(md.t_idx))
    md = dataclasses.replace(md, lineup_dev_home=lineup_dev_home, lineup_dev_away=lineup_dev_away)

    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True, use_lineup_xg=True)
    model = build_model(md, config)

    with model:
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    attack = idata.prior["attack"].values[0, 0]
    defence = idata.prior["defence"].values[0, 0]
    home_adv = idata.prior["home_adv"].values[0, 0]
    beta_lineup = float(idata.prior["beta_lineup"].values[0, 0])
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
            beta_lineup=beta_lineup,
            lineup_dev_team=md.lineup_dev_home[row],
            lineup_dev_opp=md.lineup_dev_away[row],
        )
        assert np.isclose(lam_team, lambda_home_model[row], rtol=1e-5)
        assert np.isclose(lam_opp, lambda_away_model[row], rtol=1e-5)


def test_predict_rows_matches_build_model_with_lineup_xg(two_season_model_data):
    """Same cross-check as above, but through predict_rows (the batched
    path run_cv_window.py actually calls) rather than predict_match_lambdas
    directly — proving the ModelData plumbing (lineup_dev_home/away read off
    the struct by row index) is correct too, not just the formula."""
    import dataclasses

    md = two_season_model_data
    rng = np.random.default_rng(1)
    lineup_dev_home = rng.normal(scale=0.3, size=len(md.t_idx))
    lineup_dev_away = rng.normal(scale=0.3, size=len(md.t_idx))
    md = dataclasses.replace(md, lineup_dev_home=lineup_dev_home, lineup_dev_away=lineup_dev_away)

    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True, use_lineup_xg=True)
    model = build_model(md, config)

    with model:
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    attack = idata.prior["attack"].values[0, 0]
    defence = idata.prior["defence"].values[0, 0]
    home_adv = idata.prior["home_adv"].values[0, 0]
    beta_lineup = float(idata.prior["beta_lineup"].values[0, 0])
    lambda_home_model = idata.prior["lambda_home"].values[0, 0]
    lambda_away_model = idata.prior["lambda_away"].values[0, 0]

    rows = np.array([0, 5, min(20, len(md.t_idx) - 1)])
    lam_home, lam_away, returned_rows = predict_rows(
        md, rows, attack=attack, defense=defence, home_adv=home_adv,
        clip_theta=config.clip_theta, beta_lineup=beta_lineup,
    )
    assert np.array_equal(returned_rows, rows)
    assert np.allclose(lam_home, lambda_home_model[rows], rtol=1e-5)
    assert np.allclose(lam_away, lambda_away_model[rows], rtol=1e-5)


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


def test_predict_rows_matches_epl_slice_of_multileague_model(two_season_model_data, league2_model_data):
    """WP011's key claim: evaluation needs ZERO new predict.py code. A
    trained league's own attack_<name>/defence_<name>/home_adv_<name>
    posterior slice, read alongside that league's own (completely ordinary)
    ModelData, must match build_multileague_model's lambda_home/lambda_away
    for that league exactly through the SAME predict_rows/
    predict_match_lambdas this project has used since WP001 -- proving a
    multi-league-trained EPL slice is indistinguishable, at prediction time,
    from a model that was only ever trained on EPL."""
    from football_model.model.model import build_multileague_model

    leagues = {"EPL": two_season_model_data, "Bundesliga": league2_model_data}
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True)
    model = build_multileague_model(leagues, config)

    with model:
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    md = two_season_model_data
    attack = idata.prior["attack_EPL"].values[0, 0]
    defence = idata.prior["defence_EPL"].values[0, 0]
    home_adv = idata.prior["home_adv_EPL"].values[0, 0]
    lambda_home_model = idata.prior["lambda_home_EPL"].values[0, 0]
    lambda_away_model = idata.prior["lambda_away_EPL"].values[0, 0]

    rows = np.arange(len(md.t_idx))
    lam_home, lam_away, returned_rows = predict_rows(
        md, rows, attack=attack, defense=defence, home_adv=home_adv,
        clip_theta=config.clip_theta,
    )
    assert np.array_equal(returned_rows, rows)
    assert np.allclose(lam_home, lambda_home_model[rows], rtol=1e-5)
    assert np.allclose(lam_away, lambda_away_model[rows], rtol=1e-5)

    # and the OTHER league's slice, through the exact same unchanged code path
    md2 = league2_model_data
    attack2 = idata.prior["attack_Bundesliga"].values[0, 0]
    defence2 = idata.prior["defence_Bundesliga"].values[0, 0]
    home_adv2 = idata.prior["home_adv_Bundesliga"].values[0, 0]
    lambda_home_model2 = idata.prior["lambda_home_Bundesliga"].values[0, 0]
    lambda_away_model2 = idata.prior["lambda_away_Bundesliga"].values[0, 0]

    rows2 = np.arange(len(md2.t_idx))
    lam_home2, lam_away2, _ = predict_rows(
        md2, rows2, attack=attack2, defense=defence2, home_adv=home_adv2,
        clip_theta=config.clip_theta,
    )
    assert np.allclose(lam_home2, lambda_home_model2[rows2], rtol=1e-5)
    assert np.allclose(lam_away2, lambda_away_model2[rows2], rtol=1e-5)
