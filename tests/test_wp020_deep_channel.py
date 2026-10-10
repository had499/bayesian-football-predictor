"""WP020 — the deep-completion measurement channel.

Same guarantees WP019's tests enforce, plus one specific to the refactor: the
generic count channel must not have changed `shot_count_channel`'s behaviour.
"""
import numpy as np
import pymc as pm
import pytest

from football_model.model.components import (count_measurement_channel,
                                             deep_completion_channel,
                                             shot_count_channel)
from football_model.model.model import build_model
from football_model.types.model_data import ModelConfig


@pytest.fixture
def deep_model_data(engineered_two_season_df):
    from football_model.data.prepare_model_data import prepare_model_data

    df = engineered_two_season_df.copy()
    rng = np.random.default_rng(0)
    df["deep_home"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["deep_away"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    md = prepare_model_data(df, max_round=df["round"].max())
    assert md.deep_home is not None and len(md.deep_home) == len(md.goals_home)
    return md


def test_off_by_default(deep_model_data):
    assert ModelConfig().use_deep_channel is False
    assert "deep_channel" not in build_model(deep_model_data, ModelConfig()).named_vars


def test_registers_channel_and_scalars(deep_model_data):
    m = build_model(deep_model_data, ModelConfig(use_deep_channel=True))
    for name in ("deep_channel", "deep_kappa", "deep_alpha"):
        assert name in m.named_vars, name


def test_missing_deep_data_raises(two_season_model_data):
    assert two_season_model_data.deep_home is None
    with pytest.raises(ValueError, match="use_deep_channel"):
        build_model(two_season_model_data, ModelConfig(use_deep_channel=True))


def _channel_value(md, weight):
    m = build_model(md, ModelConfig(use_deep_channel=True, deep_channel_weight=weight))
    fn = m.compile_fn(m.named_vars["deep_channel"], inputs=m.value_vars,
                      point_fn=True, on_unused_input="ignore")
    return float(fn(m.initial_point()))


def test_weight_zero_contributes_nothing(deep_model_data):
    assert _channel_value(deep_model_data, 0.0) == 0.0
    assert _channel_value(deep_model_data, 0.5) != 0.0


def test_channel_depends_on_lambda():
    """If the density doesn't move with the latent rate, the arm measures nothing."""
    obs = np.full(30, 7.0)
    logps = []
    for rate in (1.0, 1.6):
        with pm.Model() as mdl:
            lam = pm.math.constant(np.full(30, rate))
            deep_completion_channel(lam, lam, obs, obs, weight=1.0)
            logps.append(mdl.compile_logp()(mdl.initial_point()))
    assert logps[0] != pytest.approx(logps[1], abs=1e-6)


def test_weight_scales_density_linearly():
    obs = np.full(25, 7.0)
    contrib = {}
    for w in (1.0, 0.5):
        with pm.Model() as mdl:
            lam = pm.math.constant(np.full(25, 1.4))
            deep_completion_channel(lam, lam, obs, obs, weight=w)
            full = mdl.compile_logp()(mdl.initial_point())
        with pm.Model() as bare:
            lam = pm.math.constant(np.full(25, 1.4))
            deep_completion_channel(lam, lam, obs, obs, weight=0.0)
            contrib[w] = full - bare.compile_logp()(bare.initial_point())
    assert contrib[0.5] == pytest.approx(0.5 * contrib[1.0], rel=1e-6)


def test_shot_channel_unchanged_by_the_refactor():
    """shot_count_channel now delegates to count_measurement_channel. It must keep
    its variable NAMES and its exact density, or WP019's arms stop being comparable."""
    obs = np.full(20, 13.0)
    with pm.Model() as viaw:
        lam = pm.math.constant(np.full(20, 1.4))
        shot_count_channel(lam, lam, obs, obs, weight=0.5)
        a = viaw.compile_logp()(viaw.initial_point())
    with pm.Model() as direct:
        lam = pm.math.constant(np.full(20, 1.4))
        count_measurement_channel("shot", lam, lam, obs, obs, weight=0.5,
                                  kappa_mu=2.175, kappa_sd=0.2)
        b = direct.compile_logp()(direct.initial_point())
    assert set(viaw.named_vars) == {"shot_kappa", "shot_alpha", "shot_channel"}
    assert a == pytest.approx(b, rel=1e-9)


def test_stacks_with_xg_channel_and_samples(deep_model_data):
    cfg = ModelConfig(use_xg_channel=True, use_deep_channel=True)
    m = build_model(deep_model_data, cfg)
    assert {"xg_channel", "deep_channel"} <= set(m.named_vars)
    with m:
        pm.sample(draws=8, tune=8, chains=1, progressbar=False, random_seed=0,
                  compute_convergence_checks=False)


def test_goals_likelihood_untouched(deep_model_data):
    for cfg in (ModelConfig(use_deep_channel=True), ModelConfig()):
        m = build_model(deep_model_data, cfg)
        assert {v.name for v in m.observed_RVs} == {"goals_home", "goals_away"}
