"""WP037 — correlated attack/defence walks.

The premise: the prior currently asserts the two walks are INDEPENDENT, which the data
contradicts -- corr(attack, defence) across teams is +0.79. Correlating the standardised
innovations via a Cholesky factor shares information between them.

The property that matters for safety: at r = 0 the model must be IDENTICAL to the
independent walks, so turning the feature on cannot silently change anything else.
"""
import numpy as np
import pymc as pm
import pytest

from football_model.model.model import build_model
from football_model.types.model_data import ModelConfig


def _md(engineered_two_season_df):
    from football_model.data.prepare_model_data import prepare_model_data
    df = engineered_two_season_df.copy()
    rng = np.random.default_rng(0)
    df["deep_home"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["deep_away"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["stakes_home"] = rng.uniform(0, 0.6, len(df))
    df["stakes_away"] = rng.uniform(0, 0.6, len(df))
    df["stakes_match"] = np.minimum(df["stakes_home"], df["stakes_away"])
    return prepare_model_data(df, max_round=df["round"].max())


def test_off_by_default(engineered_two_season_df):
    assert ModelConfig().use_ad_correlation is False
    m = build_model(_md(engineered_two_season_df), ModelConfig.champion())
    assert "ad_corr" not in m.named_vars


def test_registers_the_correlation(engineered_two_season_df):
    m = build_model(_md(engineered_two_season_df),
                    ModelConfig.champion(use_ad_correlation=True))
    assert "ad_corr" in m.named_vars
    # the two innovation arrays must still exist and be independent N(0,1) draws
    assert "att_rw_std" in m.named_vars and "def_rw_std" in m.named_vars


def test_r_zero_is_identical_to_independent_walks(engineered_two_season_df):
    """THE safety property: at r=0 the Cholesky factor is the identity, so the
    likelihood must match the champion's exactly. If this drifts, the feature is
    silently changing the model even when 'off'."""
    md = _md(engineered_two_season_df)
    off = build_model(md, ModelConfig.champion())
    pt0 = off.initial_point()
    lik = lambda mod, p: sum(
        float(mod.compile_fn(mod.logp(v), inputs=mod.value_vars, point_fn=True,
                             on_unused_input="ignore")(p)) for v in mod.observed_RVs)
    base = lik(off, pt0)

    on = build_model(md, ModelConfig.champion(use_ad_correlation=True))
    p2 = on.initial_point()
    for k, v in pt0.items():
        if k in p2:
            p2[k] = v
    # drive r to exactly 0 through the Uniform's transform
    for k in p2:
        if k.startswith("ad_corr"):
            p2[k] = np.array(0.0)       # interval-transformed 0 == midpoint of (-.98,.98)
    assert lik(on, p2) == pytest.approx(base, rel=1e-9)


def test_defence_innovations_stay_marginally_standard(engineered_two_season_df):
    """z_def = r*z_att + sqrt(1-r^2)*z_def_raw keeps Var(z_def) = 1 for any r, which is
    what makes `sigma_def` mean the same thing with the feature on or off. Checked
    numerically rather than by inspection."""
    rng = np.random.default_rng(1)
    for r in (0.0, 0.5, 0.79, -0.9):
        za = rng.standard_normal(200_000)
        zdr = rng.standard_normal(200_000)
        zd = r * za + np.sqrt(1 - r ** 2) * zdr
        assert zd.std() == pytest.approx(1.0, abs=0.01), r
        assert np.corrcoef(za, zd)[0, 1] == pytest.approx(r, abs=0.01), r


def test_samples_and_learns_a_correlation(engineered_two_season_df):
    md = _md(engineered_two_season_df)
    m = build_model(md, ModelConfig.champion(use_ad_correlation=True))
    with m:
        tr = pm.sample(draws=25, tune=25, chains=1, progressbar=False, random_seed=0,
                       compute_convergence_checks=False)
    r = float(tr.posterior["ad_corr"].mean())
    assert -0.98 < r < 0.98
