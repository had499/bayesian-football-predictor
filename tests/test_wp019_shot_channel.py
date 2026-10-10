"""WP019 — the shot-count measurement channel.

Guards the three things that have actually gone wrong in this project when a
channel was added: the feature silently defaulting on, the channel not reaching
lambda (so the arm measures nothing), and a missing-data config raising nothing.
"""
import dataclasses

import numpy as np
import pymc as pm
import pytest

from football_model.model.components import shot_count_channel
from football_model.model.model import build_model
from football_model.types.model_data import ModelConfig


@pytest.fixture
def shot_model_data(engineered_two_season_df):
    """The shared two-season fixture, plus per-side shot counts on the home rows
    (the perspective prepare_model_data reads them from)."""
    from football_model.data.prepare_model_data import prepare_model_data

    df = engineered_two_season_df.copy()
    rng = np.random.default_rng(0)
    df["shots_home"] = rng.negative_binomial(9, 9 / 22, len(df)) + 1.0
    df["shots_away"] = rng.negative_binomial(9, 9 / 22, len(df)) + 1.0
    md = prepare_model_data(df, max_round=df["round"].max())
    assert md.shots_home is not None and len(md.shots_home) == len(md.goals_home)
    return md


def test_off_by_default(shot_model_data):
    assert ModelConfig().use_shot_channel is False
    assert "shot_channel" not in build_model(shot_model_data, ModelConfig()).named_vars


def test_registers_channel_and_scalars(shot_model_data):
    m = build_model(shot_model_data, ModelConfig(use_shot_channel=True))
    for name in ("shot_channel", "shot_kappa", "shot_alpha"):
        assert name in m.named_vars, name


def test_missing_shot_data_raises(two_season_model_data):
    """The base fixture carries no shot columns, so the config must refuse rather
    than silently building a model whose new channel observes nothing."""
    assert two_season_model_data.shots_home is None
    with pytest.raises(ValueError, match="use_shot_channel"):
        build_model(two_season_model_data, ModelConfig(use_shot_channel=True))


def _channel_value(md, weight):
    m = build_model(md, ModelConfig(use_shot_channel=True, shot_channel_weight=weight))
    # inputs=value_vars so the point function accepts every key in initial_point,
    # not just the 13 the channel's subgraph happens to depend on
    fn = m.compile_fn(m.named_vars["shot_channel"], inputs=m.value_vars,
                      point_fn=True, on_unused_input="ignore")
    return float(fn(m.initial_point()))


def test_weight_zero_contributes_nothing(shot_model_data):
    """Zero weight must contribute exactly nothing, so an arm can be turned off
    without changing the posterior. Linearity in the weight is checked
    deterministically in test_weight_scales_density_linearly — Model.initial_point()
    is stochastic for these latents, so values aren't comparable across builds.
    """
    assert _channel_value(shot_model_data, 0.0) == 0.0
    assert _channel_value(shot_model_data, 0.5) != 0.0


def test_channel_depends_on_lambda():
    """The whole point of a measurement channel: its density must move with the
    latent rate. If it doesn't, the arm is measuring nothing."""
    shots = np.full(30, 13.0)
    logps = []
    for rate in (1.0, 1.6):
        with pm.Model() as mdl:
            lam = pm.math.constant(np.full(30, rate))
            shot_count_channel(lam, lam, shots, shots, weight=1.0)
            logps.append(mdl.compile_logp()(mdl.initial_point()))
    assert logps[0] != pytest.approx(logps[1], abs=1e-6)


def test_weight_scales_density_linearly():
    shots = np.full(25, 12.0)
    contrib = {}
    for w in (1.0, 0.5):
        with pm.Model() as mdl:
            lam = pm.math.constant(np.full(25, 1.4))
            shot_count_channel(lam, lam, shots, shots, weight=w)
            full = mdl.compile_logp()(mdl.initial_point())
        with pm.Model() as bare:
            lam = pm.math.constant(np.full(25, 1.4))
            shot_count_channel(lam, lam, shots, shots, weight=0.0)
            contrib[w] = full - bare.compile_logp()(bare.initial_point())
    assert contrib[0.5] == pytest.approx(0.5 * contrib[1.0], rel=1e-6)


def test_stacks_with_xg_channel_and_samples(shot_model_data):
    """WP019's headline arm runs both channels at once, so they must coexist and
    the joint must be samplable."""
    cfg = ModelConfig(use_xg_channel=True, use_shot_channel=True)
    m = build_model(shot_model_data, cfg)
    assert {"xg_channel", "shot_channel"} <= set(m.named_vars)
    with m:
        pm.sample(draws=8, tune=8, chains=1, progressbar=False, random_seed=0,
                  compute_convergence_checks=False)


def test_goals_likelihood_untouched_by_the_channel(shot_model_data):
    """The channel must sharpen the latents without altering the goals
    likelihood — same guarantee the xG channel gives."""
    on = build_model(shot_model_data, ModelConfig(use_shot_channel=True))
    off = build_model(shot_model_data, ModelConfig())
    for m in (on, off):
        names = {v.name for v in m.observed_RVs}
        assert names == {"goals_home", "goals_away"}, names
