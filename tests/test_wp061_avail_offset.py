"""WP061: the availability offset must be IDENTICAL in model.py and predict.py.

Those two are synced BY HAND (model.py says so in a comment), and two real bugs have
already come from the gap: a missing xG term, then a pair of teams swapped between
independently computed mappings. So these tests EXECUTE both paths and compare numbers --
they never inspect source text.
"""
import numpy as np
import pytest

from football_model.data.prepare_model_data import prepare_model_data
from football_model.model.components import avail_offset_terms
from football_model.model.predict import _avail_offset, predict_rows
from football_model.types.model_data import ModelConfig


def _md(df_in, with_avail=True, seed=0):
    df = df_in.copy()
    rng = np.random.default_rng(seed)
    df["deep_home"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["deep_away"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["stakes_home"] = rng.uniform(0, 0.6, len(df))
    df["stakes_away"] = rng.uniform(0, 0.6, len(df))
    df["stakes_match"] = np.minimum(df["stakes_home"], df["stakes_away"])
    if with_avail:
        df["avail_out_h"] = rng.uniform(0, 0.3, len(df))
        df["avail_out_a"] = rng.uniform(0, 0.3, len(df))
    return prepare_model_data(df, max_round=df["round"].max())


def test_offset_formula_is_the_documented_poisson_offset(engineered_two_season_df):
    """log(1-k*own) + log(1+k*opp), and the two sides are mirror images."""
    md = _md(engineered_two_season_df)
    idx = np.arange(len(md.avail_out_h))
    k = 0.2
    oh = md.avail_out_h.astype(float)
    oa = md.avail_out_a.astype(float)
    np.testing.assert_allclose(_avail_offset(md, idx, k, side="h"),
                               np.log(1 - k * oh) + np.log(1 + k * oa), rtol=1e-5)
    np.testing.assert_allclose(_avail_offset(md, idx, k, side="a"),
                               np.log(1 - k * oa) + np.log(1 + k * oh), rtol=1e-5)


def test_offset_is_zero_when_arm_off_or_data_missing(engineered_two_season_df):
    md = _md(engineered_two_season_df)
    assert _avail_offset(md, np.arange(5), None, side="h") == 0.0
    md2 = _md(engineered_two_season_df, with_avail=False)
    assert md2.avail_out_h is None
    assert _avail_offset(md2, np.arange(5), 0.2, side="h") == 0.0


def test_sign_own_absence_lowers_own_rate_and_lifts_opponent(engineered_two_season_df):
    import dataclasses
    md = _md(engineered_two_season_df)
    n = len(md.avail_out_h)
    md = dataclasses.replace(md, avail_out_h=np.full(n, 0.3, dtype="float32"),
                             avail_out_a=np.zeros(n, dtype="float32"))
    assert _avail_offset(md, np.arange(n), 0.2, side="h").mean() < 0
    assert _avail_offset(md, np.arange(n), 0.2, side="a").mean() > 0


def test_predict_rows_shifts_lambda_by_exactly_the_offset(engineered_two_season_df):
    """end-to-end: log of the lambda ratio with/without the offset IS the offset."""
    md = _md(engineered_two_season_df)
    nt = int(md.t_walk.max()) + 1
    rng = np.random.default_rng(1)
    kw = dict(attack=rng.normal(0, .2, (nt, md.n_teams)),
              defense=rng.normal(0, .2, (nt, md.n_teams)),
              home_adv=rng.normal(.25, .05, md.n_teams), clip_theta=50.0)
    rows = np.arange(len(md.t_walk))
    lh0, la0, _ = predict_rows(md, rows, **kw)
    lh1, la1, _ = predict_rows(md, rows, avail_k=0.2, **kw)
    # atol 1e-5: prepare_model_data stores these as float32, so the two paths agree
    # only to single precision -- 1e-6 fails on ~17% of rows by ~3e-6.
    np.testing.assert_allclose(np.log(lh1 / lh0),
                               _avail_offset(md, rows, 0.2, side="h"), atol=1e-5)
    np.testing.assert_allclose(np.log(la1 / la0),
                               _avail_offset(md, rows, 0.2, side="a"), atol=1e-5)


def test_numpy_and_pytensor_give_the_same_offset():
    """The real sync risk, tested directly on the SHARED implementation.

    model.py calls avail_offset_terms with pm.math.log, predict.py with np.log. Evaluating
    it under both namespaces is the equivalence that matters, and the offset is a pure
    function of data (no random variables), so this is exactly reproducible.

    I originally tried to compare model.py's `lambda_home` against predict.py's. That was
    a bad test twice over: `model.initial_point()` is RANDOM here (two builds of identical
    data gave lambdas differing by 0.72), and compiling a deterministic over
    `model.value_vars` with on_unused_input="ignore" silently produced a graph that did not
    depend on the inputs at all -- it "passed" its own sanity check while measuring
    nothing. The duplication is now removed at the source instead.
    """
    pt = pytest.importorskip("pytensor")
    import pytensor.tensor as ptt
    rng = np.random.default_rng(0)
    oh = rng.uniform(0, 0.35, 64)
    oa = rng.uniform(0, 0.35, 64)
    for k in (0.05, 0.2, 0.5):
        n_h, n_a = avail_offset_terms(oh, oa, k, np.log)
        s_h, s_a = avail_offset_terms(ptt.as_tensor(oh), ptt.as_tensor(oa), k, ptt.log)
        np.testing.assert_allclose(n_h, s_h.eval(), rtol=1e-12)
        np.testing.assert_allclose(n_a, s_a.eval(), rtol=1e-12)


def test_model_builds_with_the_arm_on(engineered_two_season_df):
    """Cheap smoke: the PyMC path registers the data and builds without error."""
    pytest.importorskip("pymc")
    from football_model.model.model import build_model
    md = _md(engineered_two_season_df)
    with build_model(md, ModelConfig(use_avail_offset=True, avail_k=0.2)) as m:
        names = {d.name for d in m.data_vars}
    assert {"avail_oh", "avail_oa"} <= names


def test_missing_data_raises_rather_than_silently_passing(engineered_two_season_df):
    pytest.importorskip("pymc")
    from football_model.model.model import build_model
    md = _md(engineered_two_season_df, with_avail=False)
    with pytest.raises(ValueError, match="use_avail_offset needs"):
        build_model(md, ModelConfig(use_avail_offset=True))
