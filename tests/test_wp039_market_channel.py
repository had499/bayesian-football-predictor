"""WP039 — the market as a measurement channel.

Premise: Pinnacle's closing price, inverted to implied rates, adds +0.0159 OOS R^2 for
next-match goals on top of xG -- 3.3x the +0.00484 at which WP019's shot count failed.

Two properties matter here. (1) The `name=` refactor of xg_measurement_channel must NOT
have renamed any xG variable, or every earlier result becomes unreproducible. (2) Rows
with no price (about 4% of matches) carry market_obs_w = 0 and must contribute EXACTLY
nothing, since their placeholder rate of 1.0 is fiction.
"""
import numpy as np
import pymc as pm
import pytest

from football_model.model.model import build_model
from football_model.types.model_data import ModelConfig


def _md(engineered_two_season_df, with_market=True, weight=None):
    from football_model.data.prepare_model_data import prepare_model_data
    df = engineered_two_season_df.copy()
    rng = np.random.default_rng(0)
    df["deep_home"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["deep_away"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["stakes_home"] = rng.uniform(0, 0.6, len(df))
    df["stakes_away"] = rng.uniform(0, 0.6, len(df))
    df["stakes_match"] = np.minimum(df["stakes_home"], df["stakes_away"])
    if with_market:
        df["market_lh"] = rng.uniform(0.8, 2.4, len(df))
        df["market_la"] = rng.uniform(0.6, 2.0, len(df))
        df["market_obs_w"] = 1.0 if weight is None else weight
    return prepare_model_data(df, max_round=df["round"].max())


def test_xg_variable_names_unchanged_by_the_refactor(engineered_two_season_df):
    """The name= parameter defaults to "xg". If this drifts, every result from WP007
    onward becomes unreproducible, because scalar recording keys off these names."""
    m = build_model(_md(engineered_two_season_df), ModelConfig.champion())
    for v in ("xg_shape", "xg_kappa"):
        assert v in m.named_vars, f"{v} disappeared in the WP039 refactor"
    assert any("xg_channel" in str(p) for p in m.potentials)


def test_off_by_default(engineered_two_season_df):
    assert ModelConfig().use_market_channel is False
    m = build_model(_md(engineered_two_season_df), ModelConfig.champion())
    assert "mkt_kappa" not in m.named_vars


def test_registers_its_own_scalars(engineered_two_season_df):
    m = build_model(_md(engineered_two_season_df),
                    ModelConfig.champion(use_market_channel=True))
    assert "mkt_kappa" in m.named_vars and "mkt_shape" in m.named_vars
    # and must NOT have clobbered the xG channel's
    assert "xg_kappa" in m.named_vars and "xg_shape" in m.named_vars


def test_missing_data_raises(engineered_two_season_df):
    md = _md(engineered_two_season_df, with_market=False)
    assert md.market_lh is None
    with pytest.raises(ValueError, match="use_market_channel"):
        build_model(md, ModelConfig.champion(use_market_channel=True))


def test_unpriced_rows_contribute_exactly_nothing(engineered_two_season_df):
    """market_obs_w = 0 everywhere must reproduce the champion's density exactly. The
    placeholder rate of 1.0 on unpriced rows is fiction and must never reach the logp."""
    md = _md(engineered_two_season_df, weight=0.0)
    off = build_model(md, ModelConfig.champion())
    pt0 = off.initial_point()
    lp_off = off.compile_logp()(pt0)
    on = build_model(md, ModelConfig.champion(use_market_channel=True))
    p2 = on.initial_point()
    for k, v in pt0.items():
        if k in p2:
            p2[k] = v
    # the market channel's own priors add a constant; isolate the POTENTIAL itself
    pot = [p for p in on.potentials if "mkt_channel" in str(p)]
    assert len(pot) == 1
    val = float(on.compile_fn(pot[0], inputs=on.value_vars, point_fn=True,
                              on_unused_input="ignore")(p2))
    assert val == pytest.approx(0.0, abs=1e-9), f"unpriced rows leaked {val}"
    assert np.isfinite(lp_off)


def test_priced_rows_do_bite(engineered_two_season_df):
    md = _md(engineered_two_season_df, weight=1.0)
    on = build_model(md, ModelConfig.champion(use_market_channel=True))
    pt0 = on.initial_point()
    base = on.compile_logp()(pt0)
    p2 = dict(pt0)
    k = [x for x in p2 if "mkt_kappa" in x][0]
    p2[k] = np.array(float(p2[k]) + 0.4)
    assert on.compile_logp()(p2) != pytest.approx(base, rel=1e-6)


def test_weight_zero_config_disables_it(engineered_two_season_df):
    md = _md(engineered_two_season_df)
    on = build_model(md, ModelConfig.champion(use_market_channel=True,
                                              market_channel_weight=0.0))
    pot = [p for p in on.potentials if "mkt_channel" in str(p)]
    val = float(on.compile_fn(pot[0], inputs=on.value_vars, point_fn=True,
                              on_unused_input="ignore")(on.initial_point()))
    assert val == pytest.approx(0.0, abs=1e-9)


def test_samples(engineered_two_season_df):
    m = build_model(_md(engineered_two_season_df),
                    ModelConfig.champion(use_market_channel=True))
    with m:
        pm.sample(draws=8, tune=8, chains=1, progressbar=False, random_seed=0,
                  compute_convergence_checks=False)
