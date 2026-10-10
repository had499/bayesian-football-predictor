"""WP030 — squad intactness as a training-row-only covariate.

The guarantee that matters: the term is b*(intact - 1), which is EXACTLY ZERO when a
team is at full strength. Prediction assumes a normal XI, so no team-sheet information
about the match being priced is ever consumed -- that is what keeps the model usable
against OPENING prices. `predict.py` deliberately omits the term for the same reason.
"""
import numpy as np
import pymc as pm
import pytest

from football_model.model.model import build_model
from football_model.types.model_data import ModelConfig


def _md(engineered_two_season_df, with_intact=True, intact_value=None):
    from football_model.data.prepare_model_data import prepare_model_data
    df = engineered_two_season_df.copy()
    rng = np.random.default_rng(0)
    df["deep_home"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["deep_away"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["stakes_home"] = rng.uniform(0, 0.6, len(df))
    df["stakes_away"] = rng.uniform(0, 0.6, len(df))
    df["stakes_match"] = np.minimum(df["stakes_home"], df["stakes_away"])
    if with_intact:
        if intact_value is None:
            df["intact_home"] = rng.uniform(0.75, 1.15, len(df))
            df["intact_away"] = rng.uniform(0.75, 1.15, len(df))
        else:
            df["intact_home"] = intact_value
            df["intact_away"] = intact_value
    return prepare_model_data(df, max_round=df["round"].max())


def test_off_by_default(engineered_two_season_df):
    assert ModelConfig().use_intactness is False
    m = build_model(_md(engineered_two_season_df), ModelConfig.champion())
    assert not [n for n in m.named_vars if "intact" in n]


def test_registers_two_coefficients(engineered_two_season_df):
    m = build_model(_md(engineered_two_season_df), ModelConfig.champion(use_intactness=True))
    assert "intact_att" in m.named_vars and "intact_def" in m.named_vars


def test_missing_data_raises(engineered_two_season_df):
    md = _md(engineered_two_season_df, with_intact=False)
    assert md.intact_home is None
    with pytest.raises(ValueError, match="use_intactness"):
        build_model(md, ModelConfig.champion(use_intactness=True))


def test_full_strength_is_exactly_inert(engineered_two_season_df):
    """intact == 1 everywhere must reproduce the champion's DATA likelihood exactly.
    This is the property that makes prediction safe: a normal XI contributes nothing,
    so the predicted match needs no lineup information."""
    md = _md(engineered_two_season_df, intact_value=1.0)
    off = build_model(md, ModelConfig.champion())
    pt = off.initial_point()
    lik_off = sum(float(off.compile_fn(off.logp(v), inputs=off.value_vars, point_fn=True,
                                      on_unused_input="ignore")(pt)) for v in off.observed_RVs)
    on = build_model(md, ModelConfig.champion(use_intactness=True))
    p2 = on.initial_point()
    p2.update({k: v for k, v in pt.items() if k in p2})
    lik_on = sum(float(on.compile_fn(on.logp(v), inputs=on.value_vars, point_fn=True,
                                     on_unused_input="ignore")(p2)) for v in on.observed_RVs)
    assert lik_on == pytest.approx(lik_off, rel=1e-9)


def test_depletion_does_move_the_density(engineered_two_season_df):
    """With intact != 1 the term must bite, or the arm measures nothing."""
    md = _md(engineered_two_season_df, intact_value=0.8)
    on = build_model(md, ModelConfig.champion(use_intactness=True))
    pt = on.initial_point()
    base = on.compile_logp()(pt)
    pt2 = dict(pt)
    pt2["intact_att"] = np.array(0.6)
    assert on.compile_logp()(pt2) != pytest.approx(base, rel=1e-6)


def test_predict_mirror_omits_the_term_deliberately():
    """predict.py must NOT implement intactness. That asymmetry is the design: adding
    it would make the model consume the predicted match's team sheet, which is
    unavailable at the opening price. Guarding it so nobody 'fixes' the desync."""
    import inspect
    from football_model.model import predict
    # Strip comments and docstrings properly: predict.py legitimately MENTIONS
    # intactness in prose (WP031's crowd note contrasts itself against it). Only
    # executable code matters here.
    import ast
    tree = ast.parse(inspect.getsource(predict))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            node.value.value = ""          # blank out docstrings
    code = ast.unparse(tree)
    assert "intact" not in code, \
        "predict.py must not IMPLEMENT intactness — see model.py's note on why"


def test_samples(engineered_two_season_df):
    m = build_model(_md(engineered_two_season_df), ModelConfig.champion(use_intactness=True))
    with m:
        pm.sample(draws=8, tune=8, chains=1, progressbar=False, random_seed=0,
                  compute_convergence_checks=False)


def test_mean_centring_makes_the_average_match_inert(engineered_two_season_df):
    """WP030b. With intactness_center_mean=True the term is zero at AVERAGE
    availability, not at full strength. A constant intactness of 0.8 therefore
    contributes nothing -- whereas centring on 1.0 would shift every rate.
    This is the fix for the uniform +0.00090 loss the first arm took: prediction
    assumes an unknown XI, which means AVERAGE availability, not a full squad."""
    md = _md(engineered_two_season_df, intact_value=0.8)
    champ = build_model(md, ModelConfig.champion())
    pt = champ.initial_point()
    lik = lambda mod, p: sum(
        float(mod.compile_fn(mod.logp(v), inputs=mod.value_vars, point_fn=True,
                             on_unused_input="ignore")(p)) for v in mod.observed_RVs)
    base = lik(champ, pt)

    on = build_model(md, ModelConfig.champion(use_intactness=True,
                                              intactness_center_mean=True))
    p2 = on.initial_point()
    p2.update({k: v for k, v in pt.items() if k in p2})
    assert lik(on, p2) == pytest.approx(base, rel=1e-9)

    # and centring on 1.0 (the first arm) must NOT be inert at 0.8
    off = build_model(md, ModelConfig.champion(use_intactness=True))
    p3 = off.initial_point()
    p3.update({k: v for k, v in pt.items() if k in p3})
    p3["intact_att"] = np.array(0.5)
    assert lik(off, p3) != pytest.approx(base, rel=1e-6)


def test_mean_centring_still_bites_on_variation(engineered_two_season_df):
    md = _md(engineered_two_season_df)   # intact varies 0.75-1.15
    on = build_model(md, ModelConfig.champion(use_intactness=True,
                                              intactness_center_mean=True))
    pt = on.initial_point()
    base = on.compile_logp()(pt)
    pt2 = dict(pt); pt2["intact_att"] = np.array(0.6)
    assert on.compile_logp()(pt2) != pytest.approx(base, rel=1e-6)
