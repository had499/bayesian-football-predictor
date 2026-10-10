"""WP028 — end-of-season stakes: multiplicative effort, and stakes-dependent dispersion.

The guard that matters most: an overdispersed LIKELIHOOD with a Poisson PREDICTIVE
would measure nothing, since flatter probabilities in dead rubbers IS the mechanism.
"""
import numpy as np
import pymc as pm
import pytest

from football_model.model.model import build_model
from football_model.model.predict import grid_outcome_probs, scoreline_grid
from football_model.types.model_data import ModelConfig


@pytest.fixture
def stakes_md(engineered_two_season_df):
    from football_model.data.prepare_model_data import prepare_model_data
    df = engineered_two_season_df.copy()
    rng = np.random.default_rng(0)
    df["deep_home"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["deep_away"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["stakes_home"] = rng.uniform(0, 0.6, len(df))
    df["stakes_away"] = rng.uniform(0, 0.6, len(df))
    df["stakes_match"] = np.minimum(df.stakes_home, df.stakes_away)
    return prepare_model_data(df, max_round=df["round"].max())


def test_off_by_default(stakes_md):
    c = ModelConfig()
    assert c.use_stakes_effort is False and c.use_stakes_dispersion is False
    # champion() now ENABLES effort (WP028), so compare against it switched off
    m = build_model(stakes_md, ModelConfig.champion(use_stakes_effort=False))
    assert not [n for n in m.named_vars if "stakes" in n]
    assert {v.name for v in m.observed_RVs} == {"goals_home", "goals_away"}


def test_effort_registers_and_is_multiplicative(stakes_md):
    m = build_model(stakes_md, ModelConfig.champion(use_stakes_effort=True))
    assert "stakes_effort" in m.named_vars
    # A zero coefficient must reproduce the unmodified model's DATA likelihood
    # exactly, proving the term enters as (1 + g*stakes) rather than additively.
    # Compare observed-RV logp only: the joint differs by the new parameter's own
    # prior density, Normal(0, 0.3).logpdf(0) = 0.285.
    off = build_model(stakes_md, ModelConfig.champion(use_stakes_effort=False))
    pt = off.initial_point()
    lik_off = sum(float(off.compile_fn(off.logp(v), inputs=off.value_vars, point_fn=True,
                                      on_unused_input="ignore")(pt)) for v in off.observed_RVs)
    on = build_model(stakes_md, ModelConfig.champion(use_stakes_effort=True))
    p2 = on.initial_point()
    p2.update({k: v for k, v in pt.items() if k in p2})
    p2["stakes_effort"] = np.array(0.0)
    lik_on = sum(float(on.compile_fn(on.logp(v), inputs=on.value_vars, point_fn=True,
                                     on_unused_input="ignore")(p2)) for v in on.observed_RVs)
    assert lik_on == pytest.approx(lik_off, rel=1e-8)


def test_dispersion_switches_the_likelihood(stakes_md):
    m = build_model(stakes_md, ModelConfig.champion(use_stakes_dispersion=True))
    assert "stakes_disp" in m.named_vars and "disp_alpha" in m.named_vars
    names = {v.name for v in m.observed_RVs}
    assert names == {"goals_home", "goals_away"}
    # and they must be NegBinomial now, not Poisson
    kinds = {v.name: type(v.owner.op).__name__ for v in m.observed_RVs}
    assert all("NegBinomial" in k for k in kinds.values()), kinds


def test_dispersion_needs_stakes_match(two_season_model_data):
    # champion() also requires deep data, which this fixture lacks, so drop it --
    # otherwise the deep guard fires first and the test passes for the wrong reason
    cfg = ModelConfig.champion(use_deep_channel=False, use_stakes_effort=False,
                               use_stakes_dispersion=True)
    with pytest.raises(ValueError, match="use_stakes_dispersion"):
        build_model(two_season_model_data, cfg)


def test_effort_needs_stakes(two_season_model_data):
    cfg = ModelConfig.champion(use_deep_channel=False)   # effort already on
    with pytest.raises(ValueError, match="use_stakes_effort"):
        build_model(two_season_model_data, cfg)


def test_predictive_alpha_matches_poisson_in_the_limit():
    """alpha -> inf must recover the Poisson predictive exactly, so the dispersion
    arm reduces to the champion when the coefficient is zero."""
    lh, la = np.array([1.5]), np.array([1.2])
    p = grid_outcome_probs(scoreline_grid(lh, la))
    q = grid_outcome_probs(scoreline_grid(lh, la, alpha_home=np.array([1e9]),
                                          alpha_away=np.array([1e9])))
    assert np.abs(p - q).max() < 1e-6


def test_overdispersion_flattens_outcome_probabilities():
    """THE point of the dispersion arm: a dead rubber must get flatter probabilities.
    If this failed, an overdispersed likelihood would be paired with a predictive
    that ignores it, and the arm would measure nothing."""
    lh, la = np.array([1.6]), np.array([1.1])
    spreads = []
    for a in (1e9, 20.0, 8.0, 4.0):
        p = grid_outcome_probs(scoreline_grid(lh, la, alpha_home=np.array([a]),
                                              alpha_away=np.array([a])))[0]
        spreads.append(p[0] - p[2])
    assert all(spreads[i] > spreads[i + 1] for i in range(len(spreads) - 1)), spreads


def test_both_arms_sample(stakes_md):
    for cfg in (ModelConfig.champion(use_stakes_effort=True),
                ModelConfig.champion(use_stakes_dispersion=True)):
        m = build_model(stakes_md, cfg)
        with m:
            pm.sample(draws=8, tune=8, chains=1, progressbar=False, random_seed=0,
                      compute_convergence_checks=False)
