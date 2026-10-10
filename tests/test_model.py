import numpy as np
import pymc as pm
import pytensor.tensor as pt

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


def test_npxg_is_used_for_the_channel_when_present(engineered_two_season_df):
    """If the df carries npxg_home/away, the measurement channel must observe
    THOSE, not raw xG — otherwise the whole point of WP014 is silently lost."""
    from football_model.data.prepare_model_data import prepare_model_data

    df = engineered_two_season_df.copy()
    df["npxg_home"] = df["xG_home"] * 0.5      # deliberately distinguishable
    df["npxg_away"] = df["xG_away"] * 0.5
    md = prepare_model_data(df, max_round=df["round"].max())
    md_raw = prepare_model_data(engineered_two_season_df, max_round=df["round"].max())
    assert np.allclose(md.xG_obs_home, np.clip(md_raw.xG_obs_home * 0.5, 0.02, None))
    # the rolling-baseline xG covariate must be untouched by this
    assert np.allclose(md.xG_home, md_raw.xG_home)


def test_season_idx_is_chronological_and_covers_every_row(engineered_two_season_df):
    from football_model.data.prepare_model_data import prepare_model_data

    md = prepare_model_data(engineered_two_season_df, max_round=engineered_two_season_df["round"].max())
    assert md.season_idx is not None and len(md.season_idx) == len(md.goals_home)
    assert md.season_idx.min() == 0 and md.n_seasons == md.season_idx.max() + 1
    assert md.n_seasons == 2                      # fixture has seasons 2023, 2024
    # earlier season must get the lower code
    assert md.season_idx[0] <= md.season_idx[-1]


def test_season_varying_kappa_registers_per_season_values(two_season_model_data):
    config = _cfg(use_xg_channel=True, xg_kappa_by_season=True)
    model = build_model(two_season_model_data, config)
    free = {rv.name for rv in model.free_RVs}
    assert {"xg_kappa_mu", "xg_kappa_tau", "xg_kappa_z"} <= free
    assert "xg_kappa" not in free                 # the single global value is replaced
    with model:
        idata = pm.sample_prior_predictive(draws=2, random_seed=0)
    ks = idata.prior["xg_kappa_season"].values
    assert ks.shape[-1] == two_season_model_data.n_seasons
    assert (ks > 0).all()


def test_global_kappa_still_the_default(two_season_model_data):
    free = {rv.name for rv in build_model(two_season_model_data, _cfg(use_xg_channel=True)).free_RVs}
    assert "xg_kappa" in free and "xg_kappa_z" not in free


def test_time_varying_hfa_and_level_register_walks_and_reach_lambda(two_season_model_data):
    md = two_season_model_data
    config = ModelConfig(center_team_strength=True, use_xg_channel=False,
                         time_varying_hfa=True, time_varying_level=True)
    model = build_model(md, config)
    free = {rv.name for rv in model.free_RVs}
    det = {d.name for d in model.deterministics}
    assert {"hfa_t_z", "level_t_z", "sigma_hfa_t", "sigma_level_t"} <= free
    assert {"hfa_t", "level_t", "home_adv_dev"} <= det
    assert "intercept" not in free          # replaced by the drifting level

    with model:
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)
    hfa_t = idata.prior["hfa_t"].values[0, 0]
    level_t = idata.prior["level_t"].values[0, 0]
    dev = idata.prior["home_adv_dev"].values[0, 0]
    attack = idata.prior["attack"].values[0, 0]
    defence = idata.prior["defence"].values[0, 0]
    assert len(hfa_t) == md.n_time and len(level_t) == md.n_time

    row = 3
    t, h, a = int(md.t_idx[row]), int(md.team_idx[row]), int(md.opp_idx[row])
    exp_h, exp_a = predict_match_lambdas(
        attack[t, h], defence[t, h], attack[t, a], defence[t, a],
        hfa_t[t] + dev[h], clip_theta=config.clip_theta, intercept=level_t[t],
    )
    assert np.isclose(idata.prior["lambda_home"].values[0, 0][row], exp_h, rtol=1e-5)
    assert np.isclose(idata.prior["lambda_away"].values[0, 0][row], exp_a, rtol=1e-5)


def test_time_varying_level_rejects_soft_centering_and_season_kappa(two_season_model_data):
    for kw, msg in [(dict(center_team_strength=False, soft_center_team_strength=True), "hard centering"),
                    (dict(center_team_strength=True, use_xg_channel=True, xg_kappa_by_season=True), "collinear")]:
        try:
            build_model(two_season_model_data, ModelConfig(time_varying_level=True, **kw))
        except ValueError as e:
            assert msg in str(e)
        else:
            raise AssertionError(f"expected ValueError containing {msg!r}")


def test_global_walk_is_flat_when_sigma_is_zero():
    from football_model.model.components import global_walk

    with pm.Model():
        global_walk("w", 12, sigma_scale=1e-9, init_mu=0.25, init_sd=1e-9)
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)
    w = idata.prior["w"].values[0, 0]
    assert len(w) == 12 and np.allclose(w, w[0], atol=1e-5) and np.isclose(w[0], 0.25, atol=1e-3)


def test_split_defence_registers_two_latents_and_keeps_defence_meaning(two_season_model_data):
    """`defence` must stay = suppression + stopping so every downstream consumer
    (predict.py, the harness, the service) is unaffected by the split."""
    config = _cfg(use_xg_channel=True, split_defence=True)
    model = build_model(two_season_model_data, config)
    assert "sigma_stop" in {rv.name for rv in model.free_RVs}
    det = {d.name for d in model.deterministics}
    assert {"suppress", "defence", "stop_rw", "lambda_xg_home", "lambda_xg_away"} <= det

    with model:
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)
    defence = idata.prior["defence"].values[0, 0]
    suppress = idata.prior["suppress"].values[0, 0]
    stop = idata.prior["stop_rw"].values[0, 0]
    # uncentred config: defence is exactly suppression + stopping
    assert np.allclose(defence, suppress + stop, atol=1e-5)


def test_split_defence_xg_channel_sees_suppression_only(two_season_model_data):
    """The identification strategy: the xG rate must exclude shot-stopping while
    the goals rate includes it. If both saw the same rate the latents collapse."""
    md = two_season_model_data
    # clip_theta wide so soft_clip's tanh is effectively the identity — otherwise the
    # two rates are each clipped non-linearly and the exact additive relation is hidden
    model = build_model(md, _cfg(use_xg_channel=True, split_defence=True, clip_theta=1e6))
    with model:
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)
    p = idata.prior
    stop = p["stop_rw"].values[0, 0]
    lam_g = p["lambda_away"].values[0, 0]      # away goals: conceded by the HOME team
    lam_x = p["lambda_xg_away"].values[0, 0]
    # ratio must be exp(stopping of the conceding side) — here the home team
    ratio = np.log(lam_x / lam_g)
    expected = stop[md.t_idx.astype(int), md.team_idx.astype(int)]
    assert np.allclose(ratio, expected, atol=1e-4), "xG rate must differ from goals rate by exactly stopping"
    assert not np.allclose(stop, 0.0), "stopping latent should be non-zero in a prior draw"


def test_split_defence_requires_the_xg_channel(two_season_model_data):
    try:
        build_model(two_season_model_data, _cfg(split_defence=True, use_xg_channel=False))
    except ValueError as e:
        assert "use_xg_channel" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_split_defence_off_by_default_leaves_model_unchanged(two_season_model_data):
    model = build_model(two_season_model_data, _cfg(use_xg_channel=True))
    assert "sigma_stop" not in {rv.name for rv in model.free_RVs}
    assert "lambda_xg_home" not in {d.name for d in model.deterministics}


def test_outcome_likelihood_matches_the_numpy_scoreline_grid(two_season_model_data):
    """The Potential must equal weight * sum(log P(observed outcome)) computed
    from predict.py's own grid — otherwise training optimises something other
    than what we score."""
    from football_model.model.components import outcome_likelihood
    from football_model.model.predict import dc_outcome_probs

    lam_h = np.array([1.4, 0.8, 2.2])
    lam_a = np.array([1.1, 1.6, 0.7])
    gh = np.array([2, 0, 1]); ga = np.array([1, 1, 1])     # H, A, D
    with pm.Model():
        pot = outcome_likelihood(pt.as_tensor_variable(lam_h), pt.as_tensor_variable(lam_a),
                                 gh, ga, weight=0.5, max_goals=10)
        actual = float(pot.eval())
    want = 0.0
    for i, (h, a) in enumerate(zip(lam_h, lam_a)):
        ph, pdw, pa = dc_outcome_probs(h, a, max_goals=10)
        want += np.log([ph, pdw, pa][[0, 2, 1][i]])
    assert np.isclose(actual, 0.5 * want, rtol=1e-6)


def test_outcome_likelihood_weight_zero_is_a_noop(two_season_model_data):
    from football_model.model.components import outcome_likelihood

    with pm.Model():
        pot = outcome_likelihood(pt.as_tensor_variable(np.array([1.4])),
                                 pt.as_tensor_variable(np.array([1.1])),
                                 np.array([1]), np.array([0]), weight=0.0)
        assert np.isclose(float(pot.eval()), 0.0)


def test_outcome_likelihood_registers_and_samples(two_season_model_data):
    config = _cfg(use_xg_channel=True, use_outcome_likelihood=True, outcome_weight=1.0)
    model = build_model(two_season_model_data, config)
    assert "outcome_likelihood" in [p.name for p in model.potentials]
    with model:
        trace = pm.sample(draws=3, tune=3, chains=1, cores=1, progressbar=False, random_seed=0)
    assert np.isfinite(trace.posterior["lambda_home"].values).all()


def test_outcome_likelihood_off_by_default(two_season_model_data):
    model = build_model(two_season_model_data, _cfg(use_xg_channel=True))
    assert "outcome_likelihood" not in [p.name for p in model.potentials]
