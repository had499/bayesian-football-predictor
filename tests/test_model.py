import numpy as np
import pymc as pm
import pytest

from football_model.model.model import build_model, build_multileague_model
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


def _with_lineup_dev(model_data, seed=0):
    import dataclasses
    rng = np.random.default_rng(seed)
    n = len(model_data.t_idx)
    return dataclasses.replace(
        model_data,
        lineup_dev_home=rng.normal(scale=0.3, size=n),
        lineup_dev_away=rng.normal(scale=0.3, size=n),
    )


def test_use_lineup_xg_registers_beta_lineup(two_season_model_data):
    md = _with_lineup_dev(two_season_model_data)
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True, use_lineup_xg=True)
    model = build_model(md, config)
    free_names = {rv.name for rv in model.free_RVs}
    assert "beta_lineup" in free_names


def test_use_lineup_xg_false_registers_no_beta_lineup(two_season_model_data):
    """Default behaviour (every config before WP008) must be unchanged."""
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True, use_lineup_xg=False)
    model = build_model(two_season_model_data, config)
    free_names = {rv.name for rv in model.free_RVs}
    assert "beta_lineup" not in free_names


def test_use_lineup_xg_builds_and_samples(two_season_model_data):
    md = _with_lineup_dev(two_season_model_data)
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True, use_lineup_xg=True)
    model = build_model(md, config)
    with model:
        idata = pm.sample_prior_predictive(draws=3, random_seed=0)
    assert np.isfinite(idata.prior["lambda_home"].values).all()

    with model:
        trace = pm.sample(draws=3, tune=3, chains=1, cores=1, progressbar=False, random_seed=0)
    assert np.isfinite(trace.posterior["lambda_home"].values).all()
    assert np.isfinite(trace.posterior["beta_lineup"].values).all()
    assert (trace.posterior["beta_lineup"].values >= 0).all()  # HalfNormal


# --- WP011: multi-league hierarchical pooling ---

def _multileague_config(**overrides):
    return ModelConfig(center_team_strength=False, soft_center_team_strength=True, **overrides)


def test_build_multileague_model_registers_shared_and_per_league_names(two_season_model_data, league2_model_data):
    leagues = {"EPL": two_season_model_data, "Bundesliga": league2_model_data}
    model = build_multileague_model(leagues, _multileague_config())
    free_names = [rv.name for rv in model.free_RVs]
    det_names = {d.name for d in model.deterministics}
    observed_names = {rv.name for rv in model.observed_RVs}

    # shared hyperparameters: exactly one of each, regardless of league count
    for shared_name in ["rho_att", "rho_def", "sigma_att", "sigma_def",
                         "home_mu_global", "home_mu_between_league_sd", "home_sd"]:
        assert free_names.count(shared_name) == 1, shared_name

    # per-league
    for name in leagues:
        assert f"attack_{name}" in det_names
        assert f"defence_{name}" in det_names
        assert f"home_adv_{name}" in det_names
        assert f"lambda_home_{name}" in det_names
        assert f"lambda_away_{name}" in det_names
        assert f"goals_home_{name}" in observed_names
        assert f"goals_away_{name}" in observed_names


def test_build_multileague_model_handles_different_league_shapes(two_season_model_data, league2_model_data):
    """The two synthetic leagues deliberately have different team counts
    (4 vs 6) and different n_time (different rounds_per_season) -- proves
    nothing assumes leagues share shape."""
    leagues = {"EPL": two_season_model_data, "Bundesliga": league2_model_data}
    assert two_season_model_data.n_teams != league2_model_data.n_teams
    model = build_multileague_model(leagues, _multileague_config())
    with model:
        idata = pm.sample_prior_predictive(draws=3, random_seed=0)
    assert idata.prior["attack_EPL"].shape[-2:] == (two_season_model_data.n_time, two_season_model_data.n_teams)
    assert idata.prior["attack_Bundesliga"].shape[-2:] == (league2_model_data.n_time, league2_model_data.n_teams)
    assert np.isfinite(idata.prior["lambda_home_EPL"].values).all()
    assert np.isfinite(idata.prior["lambda_home_Bundesliga"].values).all()


def test_build_multileague_model_nuts_smoke(two_season_model_data, league2_model_data):
    """A tiny real NUTS run, same standard as build_model's own smoke test --
    catches shape/graph errors gradients would surface but prior-predictive
    sampling wouldn't."""
    leagues = {"EPL": two_season_model_data, "Bundesliga": league2_model_data}
    model = build_multileague_model(leagues, _multileague_config())
    with model:
        trace = pm.sample(draws=3, tune=3, chains=1, cores=1, progressbar=False, random_seed=0)
    assert np.isfinite(trace.posterior["lambda_home_EPL"].values).all()
    assert np.isfinite(trace.posterior["lambda_home_Bundesliga"].values).all()


def test_build_multileague_model_with_xg_and_dixon_coles(two_season_model_data, league2_model_data):
    leagues = {"EPL": two_season_model_data, "Bundesliga": league2_model_data}
    config = _multileague_config(use_xG=True, use_dixon_coles=True)
    model = build_multileague_model(leagues, config)
    free_names = [rv.name for rv in model.free_RVs]
    potential_names = [p.name for p in model.potentials]
    # ONE beta_xG, ONE rho_dc shared across every league, not one per league
    assert free_names.count("beta_xG") == 1
    assert free_names.count("rho_dc") == 1
    assert "dixon_coles" in potential_names
    with model:
        idata = pm.sample_prior_predictive(draws=3, random_seed=0)
    assert np.isfinite(idata.prior["lambda_home_EPL"].values).all()
    assert np.isfinite(idata.prior["lambda_home_Bundesliga"].values).all()


def test_build_multileague_model_rejects_out_of_scope_config(two_season_model_data, league2_model_data):
    leagues = {"EPL": two_season_model_data, "Bundesliga": league2_model_data}
    for kwargs in [dict(use_lineup_xg=True), dict(use_continuity=True), dict(use_per_team_sigma=True), dict(use_opponent_adjusted_xG=True)]:
        with pytest.raises(NotImplementedError):
            build_multileague_model(leagues, _multileague_config(**kwargs))


def test_build_multileague_model_league_independence_graph_structure(two_season_model_data, league2_model_data):
    """The core structural claim of WP011's architecture: one league's
    attack/defence must never depend on another league's data/noise, only
    on the explicitly shared hyperparameters (rho_att/rho_def/sigma_att/
    sigma_def). Checked directly on the PyTensor computation graph -- a
    deterministic proof, not one that depends on sampling noise."""
    import pytensor.graph.traversal as graph_basic

    leagues = {"EPL": two_season_model_data, "Bundesliga": league2_model_data}
    model = build_multileague_model(leagues, _multileague_config(use_xG=True, use_dixon_coles=True))

    for own, other in [("EPL", "Bundesliga"), ("Bundesliga", "EPL")]:
        for det_name in (f"attack_{own}", f"defence_{own}"):
            var = model.named_vars[det_name]
            ancestor_names = {v.name for v in graph_basic.ancestors([var]) if v.name is not None}
            other_only = {n for n in ancestor_names if n.endswith(f"_{other}") or f"_{other}_" in n}
            assert other_only == set(), f"{det_name} depends on {other}-specific nodes: {other_only}"

        # sanity: it DOES depend on its own league's nodes and the shared hyperparameters
        att_ancestors = {v.name for v in graph_basic.ancestors([model.named_vars[f"attack_{own}"]]) if v.name is not None}
        assert f"att_0_{own}" in att_ancestors
        assert "rho_att" in att_ancestors
        assert "sigma_att" in att_ancestors


# --- WP013: lineup-continuity covariate ---

def _with_continuity(model_data, seed=0):
    import dataclasses
    rng = np.random.default_rng(seed)
    n = len(model_data.t_idx)
    return dataclasses.replace(
        model_data,
        defence_cont_home=rng.normal(size=n).astype("float32"),
        defence_cont_away=rng.normal(size=n).astype("float32"),
    )


def test_use_continuity_registers_beta_continuity(two_season_model_data):
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True, use_continuity=True)
    model = build_model(_with_continuity(two_season_model_data), config)
    assert "beta_continuity" in {rv.name for rv in model.free_RVs}


def test_use_continuity_false_registers_no_beta_continuity(two_season_model_data):
    """Default behaviour (every config before WP013) must be unchanged."""
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True)
    model = build_model(two_season_model_data, config)
    assert "beta_continuity" not in {rv.name for rv in model.free_RVs}


def test_use_continuity_builds_and_samples_with_either_sign(two_season_model_data):
    config = ModelConfig(center_team_strength=False, soft_center_team_strength=True, use_continuity=True)
    model = build_model(_with_continuity(two_season_model_data), config)
    with model:
        idata = pm.sample_prior_predictive(draws=200, random_seed=0)
        trace = pm.sample(draws=3, tune=3, chains=1, cores=1, progressbar=False, random_seed=0)
    assert np.isfinite(idata.prior["lambda_home"].values).all()
    assert np.isfinite(trace.posterior["beta_continuity"].values).all()
    b = idata.prior["beta_continuity"].values
    assert (b < 0).any() and (b > 0).any()   # Normal prior: the data, not the prior, decides the sign
