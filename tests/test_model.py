import numpy as np
import pymc as pm

from football_model.model.model import build_model
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


def test_form_decomposition_branch_builds(two_season_model_data):
    config = ModelConfig(use_form_decomposition=True, center_team_strength=False,
                          soft_center_team_strength=True)
    model = build_model(two_season_model_data, config)
    with model:
        idata = pm.sample_prior_predictive(draws=2, random_seed=0)
    assert np.isfinite(idata.prior["attack"].values).all()
