"""WP054 — the GBM residual correction as an in-model covariate.

WP053 measured a post-hoc GBM wrapper at -0.00062 RPS, but a wrapper carries NO
uncertainty: the Bayesian model says "lambda is 1.50 +- 0.14" and the GBM says "multiply
by 1.03" as a bare point estimate. Entering it as `theta += beta * z` with z on the
log-rate scale gives the correction a posterior AND lets the model decline it.

The property that matters: at beta = 1 the in-model version must reproduce the wrapper
exactly, because z is already log(multiplicative correction). If that identity breaks,
the two results are not comparable and the whole WP053 -> WP054 chain is broken.

Unlike WP030's intactness, this covariate IS knowable in advance (built from strictly
earlier matches), so predict.py MIRRORS it -- tested here.
"""
import numpy as np
import pymc as pm
import pytest

from football_model.model.model import build_model
from football_model.model.predict import posterior_lambdas
from football_model.types.model_data import ModelConfig


def _md(engineered_two_season_df, z=None):
    from football_model.data.prepare_model_data import prepare_model_data
    df = engineered_two_season_df.copy()
    rng = np.random.default_rng(0)
    df["deep_home"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["deep_away"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["stakes_home"] = rng.uniform(0, 0.6, len(df))
    df["stakes_away"] = rng.uniform(0, 0.6, len(df))
    df["stakes_match"] = np.minimum(df["stakes_home"], df["stakes_away"])
    if z is not None:
        df["gbm_z_home"] = z if np.isscalar(z) else z
        df["gbm_z_away"] = z if np.isscalar(z) else z
    return prepare_model_data(df, max_round=df["round"].max())


def test_off_by_default(engineered_two_season_df):
    assert ModelConfig().use_gbm_covariate is False
    m = build_model(_md(engineered_two_season_df, 0.0), ModelConfig.champion())
    assert "gbm_beta" not in m.named_vars


def test_registers_one_coefficient(engineered_two_season_df):
    m = build_model(_md(engineered_two_season_df, 0.05),
                    ModelConfig.champion(use_gbm_covariate=True))
    assert "gbm_beta" in m.named_vars


def test_missing_data_raises(engineered_two_season_df):
    md = _md(engineered_two_season_df)
    assert md.gbm_z_home is None
    with pytest.raises(ValueError, match="use_gbm_covariate"):
        build_model(md, ModelConfig.champion(use_gbm_covariate=True))


def test_zero_z_is_exactly_inert(engineered_two_season_df):
    """z = 0 marks matches with no composite (the earliest windows have no training
    base). Those rows must contribute EXACTLY nothing, whatever beta is."""
    md = _md(engineered_two_season_df, 0.0)
    off = build_model(md, ModelConfig.champion())
    pt = off.initial_point()
    lik = lambda mod, p: sum(
        float(mod.compile_fn(mod.logp(v), inputs=mod.value_vars, point_fn=True,
                             on_unused_input="ignore")(p)) for v in mod.observed_RVs)
    base = lik(off, pt)
    on = build_model(md, ModelConfig.champion(use_gbm_covariate=True))
    p2 = on.initial_point(); p2.update({k: v for k, v in pt.items() if k in p2})
    p2["gbm_beta"] = np.array(0.9)          # a large beta must still not bite
    assert lik(on, p2) == pytest.approx(base, rel=1e-9)


def test_beta_one_reproduces_the_wrapper(engineered_two_season_df):
    """THE identity that makes WP053 and WP054 comparable: z is log(m), so beta = 1
    must multiply the rate by exactly m."""
    md = _md(engineered_two_season_df, 0.0)
    rows = np.arange(6)
    D, T, K = 4, md.n_time, md.n_teams
    post = {"attack": np.zeros((D, T, K)), "defence": np.zeros((D, T, K)),
            "intercept": np.full(D, 0.3), "home_adv": np.full((D, K), 0.2)}
    kw = dict(clip_theta=5.0, last_t=int(md.t_walk.max()),
              season_start_sigma_mult=1.0, n_draws=D, propagate=False)
    lh0, la0 = posterior_lambdas(md, rows, post, **kw)

    m_corr = 1.08
    md2 = _md(engineered_two_season_df, float(np.log(m_corr)))
    lh1, la1 = posterior_lambdas(md2, rows, {**post, "gbm_beta": np.ones(D)}, **kw)
    # soft_clip is mildly non-linear even far below the clip, hence rtol not exact
    assert np.allclose(lh1 / lh0, m_corr, rtol=0.01), lh1 / lh0
    assert np.allclose(la1 / la0, m_corr, rtol=0.01)


def test_predict_mirrors_it_and_scales_with_beta(engineered_two_season_df):
    md = _md(engineered_two_season_df, 0.1)
    rows = np.arange(6)
    D, T, K = 4, md.n_time, md.n_teams
    post = {"attack": np.zeros((D, T, K)), "defence": np.zeros((D, T, K)),
            "intercept": np.full(D, 0.3), "home_adv": np.full((D, K), 0.2)}
    kw = dict(clip_theta=5.0, last_t=int(md.t_walk.max()),
              season_start_sigma_mult=1.0, n_draws=D, propagate=False)
    lh0, _ = posterior_lambdas(md, rows, post, **kw)
    lh_half, _ = posterior_lambdas(md, rows, {**post, "gbm_beta": np.full(D, 0.5)}, **kw)
    lh_full, _ = posterior_lambdas(md, rows, {**post, "gbm_beta": np.ones(D)}, **kw)
    assert np.all(lh_half > lh0) and np.all(lh_full > lh_half), "beta must scale it"


def test_samples(engineered_two_season_df):
    m = build_model(_md(engineered_two_season_df, 0.05),
                    ModelConfig.champion(use_gbm_covariate=True))
    with m:
        pm.sample(draws=8, tune=8, chains=1, progressbar=False, random_seed=0,
                  compute_convergence_checks=False)
