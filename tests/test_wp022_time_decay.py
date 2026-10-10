"""WP022 — Dixon-Coles time decay.

The half Dixon & Coles (1997) that this project never implemented. Unlike the
channels, this one rewrites the GOALS likelihood (observed RV -> weighted
Potential), so the tests focus on that switch being exact and inert when off.
"""
import numpy as np
import pymc as pm
import pytest

from football_model.model.components import time_decay_weights
from football_model.model.model import build_model
from football_model.types.model_data import ModelConfig


@pytest.fixture
def decay_model_data(engineered_two_season_df):
    from football_model.data.prepare_model_data import prepare_model_data

    df = engineered_two_season_df.copy()
    rng = np.random.default_rng(0)
    df["deep_home"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["deep_away"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["stakes_home"] = rng.uniform(0, 0.6, len(df))
    df["stakes_away"] = rng.uniform(0, 0.6, len(df))
    df["stakes_match"] = np.minimum(df["stakes_home"], df["stakes_away"])
    md = prepare_model_data(df, max_round=df["round"].max())
    assert md.match_day is not None and md.match_day.min() == 0.0
    return md


def test_weights_decay_with_age_and_newest_is_one():
    day = np.array([0.0, 100.0, 200.0, 300.0])
    w = time_decay_weights(day, 100.0, normalise=False).eval()
    assert np.all(np.diff(w) > 0), "older matches must get SMALLER weight"
    assert w[-1] == pytest.approx(1.0), "the most recent match is the reference"
    # a match one half-life old must count exactly half
    assert w[-2] == pytest.approx(0.5, rel=1e-9)
    assert w[-3] == pytest.approx(0.25, rel=1e-9)


def test_normalise_preserves_total_likelihood_mass():
    """The point of normalising: isolate RECENCY WEIGHTING from 'less effective
    data'. Raw weights shrink total mass and would confound the two."""
    day = np.linspace(0, 3000, 500)
    assert time_decay_weights(day, 270.0, normalise=True).eval().mean() == pytest.approx(1.0)
    assert time_decay_weights(day, 270.0, normalise=False).eval().mean() < 0.2


def test_long_half_life_approaches_uniform():
    day = np.linspace(0, 3000, 200)
    w = time_decay_weights(day, 1e7, normalise=True).eval()
    assert np.allclose(w, 1.0, atol=1e-3)


def test_off_by_default_and_keeps_observed_rvs(decay_model_data):
    assert ModelConfig().use_time_decay is False
    m = build_model(decay_model_data, ModelConfig.champion())
    assert {v.name for v in m.observed_RVs} == {"goals_home", "goals_away"}
    assert "goals_decayed" not in m.named_vars


def test_on_switches_goals_to_a_weighted_potential(decay_model_data):
    m = build_model(decay_model_data, ModelConfig.champion(use_time_decay=True))
    assert "goals_decayed" in m.named_vars
    assert not m.observed_RVs, "goals must no longer be an observed RV when decayed"


def test_missing_match_day_raises(decay_model_data):
    import dataclasses
    md = dataclasses.replace(decay_model_data, match_day=None)
    with pytest.raises(ValueError, match="use_time_decay"):
        build_model(md, ModelConfig.champion(use_time_decay=True))


def test_effectively_infinite_half_life_reproduces_the_undecayed_logp(decay_model_data):
    """A decay so slow it is uniform must give the SAME joint density as no decay.
    This is the guarantee that the Potential rewrite is exact, not merely similar."""
    off = build_model(decay_model_data, ModelConfig.champion())
    on = build_model(decay_model_data, ModelConfig.champion(
        use_time_decay=True, decay_half_life_days=1e9))
    pt = off.initial_point()
    pt_on = on.initial_point()
    pt_on.update({k: v for k, v in pt.items() if k in pt_on})
    assert on.compile_logp()(pt_on) == pytest.approx(off.compile_logp()(pt), rel=1e-8)


def test_short_half_life_changes_the_density(decay_model_data):
    off = build_model(decay_model_data, ModelConfig.champion())
    on = build_model(decay_model_data, ModelConfig.champion(
        use_time_decay=True, decay_half_life_days=60.0))
    pt = off.initial_point()
    pt_on = on.initial_point()
    pt_on.update({k: v for k, v in pt.items() if k in pt_on})
    assert on.compile_logp()(pt_on) != pytest.approx(off.compile_logp()(pt), rel=1e-6)


def test_decay_reaches_the_channels_too(decay_model_data):
    """Decay must apply to the whole likelihood. If the channels ignored it, old
    matches would still count fully through xG/deep and the arm would be diluted."""
    vals = {}
    for hl in (60.0, 1e9):
        mm = build_model(decay_model_data, ModelConfig.champion(
            use_time_decay=True, decay_half_life_days=hl))
        fn = mm.compile_fn([mm.named_vars["xg_channel"], mm.named_vars["deep_channel"]],
                           inputs=mm.value_vars, point_fn=True, on_unused_input="ignore")
        vals[hl] = [float(x) for x in fn(mm.initial_point())]
    assert vals[60.0][0] != pytest.approx(vals[1e9][0], rel=1e-6), "xG channel ignored decay"
    assert vals[60.0][1] != pytest.approx(vals[1e9][1], rel=1e-6), "deep channel ignored decay"


def test_samples(decay_model_data):
    m = build_model(decay_model_data, ModelConfig.champion(use_time_decay=True))
    with m:
        pm.sample(draws=8, tune=8, chains=1, progressbar=False, random_seed=0,
                  compute_convergence_checks=False)
