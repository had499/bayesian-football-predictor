"""WP031 — crowdless-match shift in home advantage.

CONTRAST WITH WP030, which is the whole point. Squad intactness is NOT knowable at the
opening price, so predict.py deliberately omits it. Crowd status IS knowable -- a
scheduling fact published months ahead -- so predict.py MUST mirror it. Withholding it
would make the model predict crowdless matches as if crowds were present, which is the
very bias the term exists to remove.
"""
import numpy as np
import pymc as pm
import pytest

from football_model.model.model import build_model
from football_model.model.predict import posterior_lambdas
from football_model.types.model_data import ModelConfig


def _md(engineered_two_season_df, crowd=None):
    from football_model.data.prepare_model_data import prepare_model_data
    df = engineered_two_season_df.copy()
    rng = np.random.default_rng(0)
    df["deep_home"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["deep_away"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["stakes_home"] = rng.uniform(0, 0.6, len(df))
    df["stakes_away"] = rng.uniform(0, 0.6, len(df))
    df["stakes_match"] = np.minimum(df["stakes_home"], df["stakes_away"])
    if crowd is not None:
        df["crowdless"] = crowd if np.isscalar(crowd) else crowd
    return prepare_model_data(df, max_round=df["round"].max())


def test_off_by_default(engineered_two_season_df):
    assert ModelConfig().use_crowd_regime is False
    m = build_model(_md(engineered_two_season_df, 0.0), ModelConfig.champion())
    assert "crowd_hfa" not in m.named_vars


def test_registers_coefficient(engineered_two_season_df):
    m = build_model(_md(engineered_two_season_df, 0.0),
                    ModelConfig.champion(use_crowd_regime=True))
    assert "crowd_hfa" in m.named_vars


def test_missing_data_raises(engineered_two_season_df):
    md = _md(engineered_two_season_df)
    assert md.crowdless is None
    with pytest.raises(ValueError, match="use_crowd_regime"):
        build_model(md, ModelConfig.champion(use_crowd_regime=True))


def test_inert_when_no_match_is_crowdless(engineered_two_season_df):
    md = _md(engineered_two_season_df, 0.0)
    off = build_model(md, ModelConfig.champion())
    pt = off.initial_point()
    lik = lambda mod, p: sum(
        float(mod.compile_fn(mod.logp(v), inputs=mod.value_vars, point_fn=True,
                             on_unused_input="ignore")(p)) for v in mod.observed_RVs)
    base = lik(off, pt)
    on = build_model(md, ModelConfig.champion(use_crowd_regime=True))
    p2 = on.initial_point(); p2.update({k: v for k, v in pt.items() if k in p2})
    p2["crowd_hfa"] = np.array(0.7)       # even a large coefficient must not bite
    assert lik(on, p2) == pytest.approx(base, rel=1e-9)


def test_crowdless_rows_do_bite(engineered_two_season_df):
    md = _md(engineered_two_season_df, 1.0)
    on = build_model(md, ModelConfig.champion(use_crowd_regime=True))
    pt = on.initial_point()
    base = on.compile_logp()(pt)
    pt2 = dict(pt); pt2["crowd_hfa"] = np.array(-0.3)
    assert on.compile_logp()(pt2) != pytest.approx(base, rel=1e-6)


def test_predict_MIRRORS_the_term(engineered_two_season_df):
    """The contrast with WP030: a NEGATIVE crowd_hfa must LOWER the predicted home
    rate on a crowdless row, and leave the away rate alone. If predict.py ignored
    crowd_hfa (as it correctly ignores intactness) this test would fail -- and the CV
    arm would be scored as if crowdless matches had crowds."""
    md = _md(engineered_two_season_df, 1.0)
    rows = np.arange(6)
    D, T, K = 5, md.n_time, md.n_teams
    base_post = {
        "attack": np.zeros((D, T, K)), "defence": np.zeros((D, T, K)),
        "intercept": np.full(D, 0.2), "home_adv": np.full((D, K), 0.25),
    }
    kw = dict(clip_theta=5.0, last_t=int(md.t_walk.max()), season_start_sigma_mult=1.0,
              n_draws=D, propagate=False)
    lh0, la0 = posterior_lambdas(md, rows, base_post, **kw)
    lh1, la1 = posterior_lambdas(md, rows, {**base_post, "crowd_hfa": np.full(D, -0.30)}, **kw)
    assert np.all(lh1 < lh0 * 0.99), "crowd_hfa did not reach the predicted home rate"
    assert np.allclose(la1, la0), "crowd_hfa must not touch the away rate"
    # and the size must be ~exp(-0.30), i.e. the term rides inside home_adv rather
    # than being damped or doubled somewhere. Not exact: theta passes through
    # soft_clip, which is mildly nonlinear even far below the clip (0.7417 vs 0.7408).
    assert np.allclose(lh1 / lh0, np.exp(-0.30), rtol=0.005)


def test_predict_ignores_it_when_crowds_present(engineered_two_season_df):
    md = _md(engineered_two_season_df, 0.0)
    rows = np.arange(6)
    D, T, K = 4, md.n_time, md.n_teams
    post = {"attack": np.zeros((D, T, K)), "defence": np.zeros((D, T, K)),
            "intercept": np.full(D, 0.2), "home_adv": np.full((D, K), 0.25)}
    kw = dict(clip_theta=5.0, last_t=int(md.t_walk.max()), season_start_sigma_mult=1.0,
              n_draws=D, propagate=False)
    lh0, _ = posterior_lambdas(md, rows, post, **kw)
    lh1, _ = posterior_lambdas(md, rows, {**post, "crowd_hfa": np.full(D, -0.30)}, **kw)
    assert np.allclose(lh0, lh1)


def test_samples(engineered_two_season_df):
    md = _md(engineered_two_season_df, 1.0)
    m = build_model(md, ModelConfig.champion(use_crowd_regime=True))
    with m:
        pm.sample(draws=8, tune=8, chains=1, progressbar=False, random_seed=0,
                  compute_convergence_checks=False)
