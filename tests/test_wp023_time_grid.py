"""WP023 — coarsening the latent walk's time grid (a performance change).

The bug this guards against is specific and was hit for real: `t_idx` was doing
two jobs — indexing the latent walk AND labelling which round a match is in.
Coarsening it silently mislabelled rounds, corrupting test-row selection and the
downstream odds join. The two indices are now separate and must stay separate.
"""
import numpy as np
import pytest

from football_model.data.prepare_model_data import prepare_model_data
from football_model.model.model import build_model
from football_model.types.model_data import ModelConfig, ModelData


def test_k1_is_bit_identical(engineered_two_season_df):
    """The default must reproduce every pre-WP023 result exactly."""
    a = prepare_model_data(engineered_two_season_df, max_round=engineered_two_season_df["round"].max())
    b = prepare_model_data(engineered_two_season_df, max_round=engineered_two_season_df["round"].max(),
                           time_grid_step=1)
    assert np.array_equal(a.t_idx, b.t_idx)
    assert np.array_equal(a.t_walk, b.t_walk)
    assert np.array_equal(a.t_idx, a.t_walk), "at k=1 the walk index IS the round"
    assert a.n_time == b.n_time


def test_coarsening_shrinks_the_walk_but_not_the_data(engineered_two_season_df):
    mr = engineered_two_season_df["round"].max()
    base = prepare_model_data(engineered_two_season_df, max_round=mr)
    for k in (2, 3):
        md = prepare_model_data(engineered_two_season_df, max_round=mr, time_grid_step=k)
        assert len(md.goals_home) == len(base.goals_home), "observations must not be dropped"
        assert md.n_time <= base.n_time // k + 1
        assert md.t_walk.max() == md.n_time - 1


def test_round_labels_survive_coarsening(engineered_two_season_df):
    """The regression that motivated this file: t_idx must stay the TRUE round,
    because test rows are selected and odds are joined on it."""
    mr = engineered_two_season_df["round"].max()
    base = prepare_model_data(engineered_two_season_df, max_round=mr)
    for k in (2, 3, 4):
        md = prepare_model_data(engineered_two_season_df, max_round=mr, time_grid_step=k)
        assert np.array_equal(md.t_idx, base.t_idx), f"t_idx changed at k={k}"
        assert np.array_equal(md.t_walk, base.t_idx // k)


def test_masks_are_built_on_the_walk_grid(engineered_two_season_df):
    mr = engineered_two_season_df["round"].max()
    for k in (1, 2, 3):
        md = prepare_model_data(engineered_two_season_df, max_round=mr, time_grid_step=k)
        assert md.active_mask.shape == (md.n_time, md.n_teams)
        assert md.season_start_mask.shape == (md.n_time,)


def test_model_builds_and_latents_shrink(engineered_two_season_df):
    mr = engineered_two_season_df["round"].max()
    sizes = {}
    for k in (1, 2):
        md = prepare_model_data(engineered_two_season_df, max_round=mr, time_grid_step=k)
        m = build_model(md, ModelConfig())
        name = next(n for n in m.named_vars if n.endswith("_std") and "att" in n)
        sizes[k] = m.named_vars[name].type.shape[0]
    assert sizes[2] < sizes[1], "coarsening must actually reduce the latent count"


def test_t_walk_defaults_to_t_idx_when_absent():
    """A ModelData built by hand (tests, older callers) must behave as before."""
    n = 6
    md = ModelData(
        n_teams=2, n_matches=n, n_time=n,
        t_idx=np.arange(n, dtype="int32"),
        team_idx=np.zeros(n, "int32"), opp_idx=np.ones(n, "int32"),
        match_idx=np.arange(n, dtype="int32"), home=np.ones(n, "int32"),
        goals_home=np.ones(n, "int32"), goals_away=np.ones(n, "int32"),
        xG_home=np.ones(n, "float32"), xG_away=np.ones(n, "float32"),
    )
    assert md.t_walk is not None
    assert np.array_equal(md.t_walk, md.t_idx)
