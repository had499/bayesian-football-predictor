"""WP025 — Elo-scaled entry level for promoted teams.

The defect: `entry_shift` is one global scalar, fitted at -0.06 in WP007 against a
real promotion gap of -0.37. Measured pre-debut Elo spans 1475 (Huddersfield 2017)
to 1726 (Burnley 2023), so one scalar cannot represent it.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from football_model.model.components import ar1_team_process
from football_model.model.model import build_model
from football_model.types.model_data import ModelConfig


def _md(engineered_two_season_df, with_elo=True):
    from football_model.data.prepare_model_data import prepare_model_data
    import pandas as pd
    df = engineered_two_season_df.copy()
    rng = np.random.default_rng(0)
    df["deep_home"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["deep_away"] = rng.negative_binomial(4, 4 / 11, len(df)) + 0.0
    df["stakes_home"] = rng.uniform(0, 0.6, len(df))
    df["stakes_away"] = rng.uniform(0, 0.6, len(df))
    df["stakes_match"] = np.minimum(df["stakes_home"], df["stakes_away"])
    grid = None
    if with_elo:
        teams = sorted(set(df.team) | set(df.opp_team))
        rounds = sorted(df.cum_round.unique())
        grid = pd.DataFrame(rng.normal(0, 1, (len(rounds), len(teams))),
                            index=rounds, columns=teams)
    return prepare_model_data(df, max_round=df["round"].max(), elo_z_grid=grid)


def test_off_by_default(engineered_two_season_df):
    md = _md(engineered_two_season_df)
    assert ModelConfig().use_elo_entry is False
    m = build_model(md, ModelConfig.champion())
    assert not [n for n in m.named_vars if "elo_entry" in n]


def test_registers_coefficients(engineered_two_season_df):
    m = build_model(_md(engineered_two_season_df), ModelConfig.champion(use_elo_entry=True))
    assert "elo_entry_att" in m.named_vars and "elo_entry_def" in m.named_vars


def test_missing_elo_raises(engineered_two_season_df):
    md = _md(engineered_two_season_df, with_elo=False)
    assert md.entry_elo_z is None
    with pytest.raises(ValueError, match="use_elo_entry"):
        build_model(md, ModelConfig.champion(use_elo_entry=True))


def test_entry_scale_none_is_bit_identical():
    """The default path must reproduce the pre-WP025 recursion exactly."""
    n_time, n_teams = 12, 4
    act = np.ones((n_time, n_teams), "float32")
    act[:5, 3] = 0.0                      # team 3 enters at t=5
    outs = []
    for scale in (None, np.zeros((n_time, n_teams), "float32")):
        with pm.Model() as mdl:
            x = ar1_team_process("w", n_time, n_teams, 0.02, 0.98,
                                 active_mask=act, entry_shift=0.3, entry_scale=scale)
            fn = mdl.compile_fn(x, inputs=mdl.value_vars, point_fn=True,
                                on_unused_input="ignore")
            outs.append(np.asarray(fn(mdl.initial_point())))
    # a zero scale kills the shift; None keeps it -> they must DIFFER, proving the
    # scale is actually applied rather than silently ignored
    assert not np.allclose(outs[0], outs[1])


def test_scale_reaches_the_latent_only_at_entry():
    n_time, n_teams = 12, 3
    act = np.ones((n_time, n_teams), "float32")
    act[:5, 2] = 0.0                      # team 2 enters at t=5
    scale = np.zeros((n_time, n_teams), "float32")
    scale[:, 2] = 2.0                     # strong Elo weight for the entrant
    vals = {}
    for shift in (0.0, 1.0):
        with pm.Model() as mdl:
            x = ar1_team_process("w", n_time, n_teams, 1e-9, 1.0, active_mask=act,
                                 entry_shift=shift, entry_scale=scale, init_scale=1e-9)
            fn = mdl.compile_fn(x, inputs=mdl.value_vars, point_fn=True,
                                on_unused_input="ignore")
            vals[shift] = np.asarray(fn(mdl.initial_point()))
    d = vals[1.0] - vals[0.0]
    assert abs(d[5:, 2].mean() - 2.0) < 1e-3, "entrant should shift by scale*shift = 2.0"
    assert np.abs(d[:, :2]).max() < 1e-6, "teams already present must not move"
    assert np.abs(d[:5, 2]).max() < 1e-6, "no shift before the entry round"


def test_repromotion_gets_distinct_values():
    """Teams re-enter the league (Norwich 2015/2019/2021). A per-team value would be
    wrong; the grid must supply a different weight at each entry."""
    n_time, n_teams = 20, 2
    act = np.ones((n_time, n_teams), "float32")
    act[5:10, 1] = 0.0                    # team 1 leaves then returns at t=10
    scale = np.zeros((n_time, n_teams), "float32")
    scale[0, 1] = 1.0
    scale[10, 1] = -3.0                   # a different strength on re-entry
    with pm.Model() as mdl:
        x = ar1_team_process("w", n_time, n_teams, 1e-9, 1.0, active_mask=act,
                             entry_shift=1.0, entry_scale=scale, init_scale=1e-9)
        fn = mdl.compile_fn(x, inputs=mdl.value_vars, point_fn=True, on_unused_input="ignore")
        out = np.asarray(fn(mdl.initial_point()))
    assert out[10:, 1].mean() < -2.0, "re-entry must use the RE-ENTRY weight, not the first"


def test_samples(engineered_two_season_df):
    m = build_model(_md(engineered_two_season_df), ModelConfig.champion(use_elo_entry=True))
    with m:
        pm.sample(draws=8, tune=8, chains=1, progressbar=False, random_seed=0,
                  compute_convergence_checks=False)
