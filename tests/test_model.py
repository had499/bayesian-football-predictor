import numpy as np
import pymc as pm

from football_model.model.model import build_model
from football_model.model.predict import predict_match_lambdas
from football_model.types.model_data import ModelConfig


def test_build_model_prior_predictive_smoke(two_season_model_data):
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True)
    model = build_model(two_season_model_data, config)
    with model:
        idata = pm.sample_prior_predictive(draws=3, random_seed=0)

    assert idata.prior["attack"].shape[-2:] == (
        two_season_model_data.n_time,
        two_season_model_data.n_teams,
    )
    assert np.isfinite(idata.prior["lambda_home"].values).all()
    assert np.isfinite(idata.prior["lambda_away"].values).all()
    assert (idata.prior["lambda_home"].values > 0).all()
    assert (idata.prior["lambda_away"].values > 0).all()


def test_build_model_nuts_smoke(two_season_model_data):
    """A tiny real NUTS run — not for convergence, just to catch shape/graph
    errors that only show up once gradients are taken through the model."""
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True)
    model = build_model(two_season_model_data, config)
    with model:
        trace = pm.sample(
            draws=3, tune=3, chains=1, cores=1, progressbar=False, random_seed=0
        )
    assert np.isfinite(trace.posterior["lambda_home"].values).all()


def test_hard_centering_uses_defence_name_consistently(two_season_model_data):
    """Regression test: both the centered and uncentered branches must
    register the deterministic under the same name ('defence'), since
    predictor.py always reads trace.posterior['defence']."""
    config = ModelConfig(center_team_strength=True)
    model = build_model(two_season_model_data, config)
    var_names = {rv.name for rv in model.deterministics}
    assert "defence" in var_names
    assert "defense" not in var_names


def test_relegated_team_attack_frozen_within_trained_model(two_season_model_data):
    md = two_season_model_data
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True)
    model = build_model(md, config)
    with model:
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    d_idx = md.team_mapping["D"]
    attack_d = idata.prior["attack"].values[0, 0, :, d_idx]
    inactive_ts = np.where(md.active_mask[:, d_idx] == 0.0)[0]
    assert len(inactive_ts) > 0
    assert np.allclose(attack_d[inactive_ts], attack_d[inactive_ts[0]])


def test_dixon_coles_disabled_by_default(two_season_model_data):
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True)
    model = build_model(two_season_model_data, config)
    var_names = [rv.name for rv in model.free_RVs]
    potential_names = [p.name for p in model.potentials]
    assert "rho_dc" not in var_names
    assert "dixon_coles" not in potential_names


def test_dixon_coles_enabled_builds_and_samples(two_season_model_data):
    config = ModelConfig(
        center_team_strength=False, soft_center_team_strength=True, use_dixon_coles=True
    )
    model = build_model(two_season_model_data, config)
    var_names = [rv.name for rv in model.free_RVs]
    potential_names = [p.name for p in model.potentials]
    assert "rho_dc" in var_names
    assert "dixon_coles" in potential_names

    with model:
        idata = pm.sample_prior_predictive(draws=3, random_seed=0)
    assert np.isfinite(idata.prior["lambda_home"].values).all()

    with model:
        trace = pm.sample(draws=3, tune=3, chains=1, cores=1, progressbar=False, random_seed=0)
    assert np.isfinite(trace.posterior["rho_dc"].values).all()


def test_use_per_team_sigma_registers_hierarchical_priors_not_global_scalar(two_season_model_data):
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True,
                          use_per_team_sigma=True)
    model = build_model(two_season_model_data, config)
    free_names = {rv.name for rv in model.free_RVs}
    det_names = {d.name for d in model.deterministics}
    # per-team hierarchical names present, the old global-scalar names absent
    assert {"sigma_att_pop", "sigma_att_team_raw", "sigma_def_pop", "sigma_def_team_raw"} <= free_names
    assert {"sigma_att_team", "sigma_def_team"} <= det_names
    assert "sigma_att" not in free_names and "sigma_def" not in free_names


def test_use_per_team_sigma_false_keeps_original_global_scalar_priors(two_season_model_data):
    """Default behaviour (every config before WP006) must be unchanged."""
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True,
                          use_per_team_sigma=False)
    model = build_model(two_season_model_data, config)
    free_names = {rv.name for rv in model.free_RVs}
    assert {"sigma_att", "sigma_def"} <= free_names
    assert "sigma_att_pop" not in free_names and "sigma_att_team_raw" not in free_names


def test_use_per_team_sigma_builds_and_samples(two_season_model_data):
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True,
                          use_per_team_sigma=True)
    model = build_model(two_season_model_data, config)
    with model:
        idata = pm.sample_prior_predictive(draws=3, random_seed=0)
    assert np.isfinite(idata.prior["lambda_home"].values).all()
    assert idata.prior["sigma_att_team"].shape[-1] == two_season_model_data.n_teams

    with model:
        trace = pm.sample(draws=3, tune=3, chains=1, cores=1, progressbar=False, random_seed=0)
    assert np.isfinite(trace.posterior["lambda_home"].values).all()
    assert np.isfinite(trace.posterior["sigma_att_team"].values).all()
    assert (trace.posterior["sigma_att_team"].values > 0).all()


def test_form_decomposition_branch_builds(two_season_model_data):
    config = ModelConfig(use_form_decomposition=True, center_team_strength=False,
                          soft_center_team_strength=True)
    model = build_model(two_season_model_data, config)
    with model:
        idata = pm.sample_prior_predictive(draws=2, random_seed=0)
    assert np.isfinite(idata.prior["attack"].values).all()


def _cfg(**kw):
    return ModelConfig(center_team_strength=False, soft_center_team_strength=True, **kw)


def test_random_walk_dynamics_drops_static_level_and_rho(two_season_model_data):
    model = build_model(two_season_model_data, _cfg(dynamics="random_walk"))
    free = {rv.name for rv in model.free_RVs}
    assert not ({"att_0", "def_0", "rho_att", "rho_def"} & free)
    assert {"sigma_att", "sigma_def", "att_rw_std"} <= free


def test_default_dynamics_unchanged(two_season_model_data):
    free = {rv.name for rv in build_model(two_season_model_data, _cfg()).free_RVs}
    assert {"att_0", "def_0", "rho_att", "rho_def"} <= free
    assert not ({"promo_att", "intercept", "xg_shape"} & free)


def test_random_walk_promoted_effect_intercept_xg_channel_build_and_sample(two_season_model_data):
    config = ModelConfig(dynamics="random_walk", sigma_att=0.05, sigma_def=0.05, use_promoted_effect=True,
                         use_intercept=True, use_xg_channel=True, use_dixon_coles=True,
                         center_team_strength=True)
    model = build_model(two_season_model_data, config)
    free = {rv.name for rv in model.free_RVs}
    assert {"promo_att", "promo_def", "intercept", "xg_shape", "xg_kappa", "rho_dc"} <= free
    assert {"attack_raw", "defence_raw"} <= {d.name for d in model.deterministics}
    with model:
        trace = pm.sample(draws=3, tune=3, chains=1, cores=1, progressbar=False, random_seed=0)
    for v in ("promo_att", "intercept", "xg_shape", "lambda_home"):
        assert np.isfinite(trace.posterior[v].values).all()


def test_promoted_shift_reaches_the_newly_promoted_teams_latent_state(two_season_model_data):
    """E enters in season 2; with a big fixed shift its attack at entry must
    drop by exactly that amount relative to its (frozen) pre-entry value."""
    md = two_season_model_data
    model = build_model(md, _cfg(dynamics="random_walk", use_promoted_effect=True, sigma_att=1e-6))
    e = md.team_mapping["E"]
    entry_t = int(np.argmax((md.active_mask[:, e] > 0) & (np.arange(md.n_time) > 0)))
    with model:
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)
    att = idata.prior["attack"].values[0, 0, :, e]
    shift = float(idata.prior["promo_att"].values[0, 0])
    assert np.isclose(att[entry_t] - att[entry_t - 1], shift, atol=1e-3)


def test_xg_channel_without_realised_xg_raises_clearly(two_season_model_data):
    import dataclasses

    md = dataclasses.replace(two_season_model_data, xG_obs_home=None, xG_obs_away=None)
    try:
        build_model(md, _cfg(use_xg_channel=True))
    except ValueError as e:
        assert "xG_obs" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_form_decomposition_rejects_random_walk_options(two_season_model_data):
    try:
        build_model(two_season_model_data, _cfg(use_form_decomposition=True, use_promoted_effect=True))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_absence_covariate_registers_and_reaches_lambda(two_season_model_data):
    """beta_absence must actually move the predicted rate, not just exist."""
    import dataclasses

    md = dataclasses.replace(
        two_season_model_data,
        absent_home=np.full(len(two_season_model_data.goals_home), 0.4, dtype="float32"),
        absent_away=np.zeros(len(two_season_model_data.goals_home), dtype="float32"),
    )
    model = build_model(md, _cfg(use_absence=True))
    assert "beta_absence" in {rv.name for rv in model.free_RVs}
    with model:
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)
    beta = float(idata.prior["beta_absence"].values[0, 0])
    lam_h = idata.prior["lambda_home"].values[0, 0]
    lam_a = idata.prior["lambda_away"].values[0, 0]
    # home carries absence 0.4, away carries 0 -> the ratio must show exactly exp(beta*0.4)
    attack = idata.prior["attack"].values[0, 0]
    defence = idata.prior["defence"].values[0, 0]
    home_adv = idata.prior["home_adv"].values[0, 0]
    t, h, a = int(md.t_idx[0]), int(md.team_idx[0]), int(md.opp_idx[0])
    exp_h, exp_a = predict_match_lambdas(
        attack[t, h], defence[t, h], attack[t, a], defence[t, a], home_adv[h],
        clip_theta=_cfg().clip_theta,
        intercept=0.0, absence_team=beta * 0.4, absence_opp=0.0,
    )
    assert np.isclose(lam_h[0], exp_h, rtol=1e-5)
    assert np.isclose(lam_a[0], exp_a, rtol=1e-5)


def test_absence_disabled_by_default_and_errors_without_data(two_season_model_data):
    import dataclasses

    assert "beta_absence" not in {rv.name for rv in build_model(two_season_model_data, _cfg()).free_RVs}
    md = dataclasses.replace(two_season_model_data, absent_home=None, absent_away=None)
    try:
        build_model(md, _cfg(use_absence=True))
    except ValueError as e:
        assert "absent_home" in str(e)
    else:
        raise AssertionError("expected ValueError")
