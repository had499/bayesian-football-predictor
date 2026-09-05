import numpy as np

from football_model.data.prepare_model_data import prepare_model_data


def test_team_union_spans_all_seasons(two_season_model_data):
    md = two_season_model_data
    assert set(md.team_mapping) == {"A", "B", "C", "D", "E"}
    assert md.n_teams == 5


def test_relegated_team_inactive_after_leaving(engineered_two_season_df, two_season_model_data):
    md = two_season_model_data
    d_idx = md.team_mapping["D"]

    season1_ts = engineered_two_season_df.loc[
        engineered_two_season_df["season"] == "2023", "cum_round"
    ].astype(int)
    season2_ts = engineered_two_season_df.loc[
        engineered_two_season_df["season"] == "2024", "cum_round"
    ].astype(int)

    # D played every round of season 1 -> active
    assert (md.active_mask[sorted(season1_ts.unique()), d_idx] == 1.0).all()
    # D is relegated for season 2 -> inactive throughout
    assert (md.active_mask[sorted(season2_ts.unique()), d_idx] == 0.0).all()


def test_promoted_team_inactive_before_arriving(engineered_two_season_df, two_season_model_data):
    md = two_season_model_data
    e_idx = md.team_mapping["E"]

    season1_ts = engineered_two_season_df.loc[
        engineered_two_season_df["season"] == "2023", "cum_round"
    ].astype(int)
    season2_ts = engineered_two_season_df.loc[
        engineered_two_season_df["season"] == "2024", "cum_round"
    ].astype(int)

    # E doesn't exist in the top flight until season 2
    assert (md.active_mask[sorted(season1_ts.unique()), e_idx] == 0.0).all()
    assert (md.active_mask[sorted(season2_ts.unique()), e_idx] == 1.0).all()


def test_continuing_teams_active_throughout(engineered_two_season_df, two_season_model_data):
    md = two_season_model_data
    played_ts = sorted(engineered_two_season_df["cum_round"].astype(int).unique())
    for team in ("A", "B", "C"):
        idx = md.team_mapping[team]
        assert (md.active_mask[played_ts, idx] == 1.0).all()


def test_season_start_window_flags_first_k_rounds_of_each_season(
    engineered_two_season_df,
):
    window = 3
    md = prepare_model_data(
        engineered_two_season_df,
        max_round=engineered_two_season_df["round"].max(),
        season_start_window=window,
    )

    season1_ts = sorted(engineered_two_season_df.loc[engineered_two_season_df["season"] == "2023", "cum_round"].astype(int).unique())
    season2_ts = sorted(engineered_two_season_df.loc[engineered_two_season_df["season"] == "2024", "cum_round"].astype(int).unique())

    assert list(md.season_start_mask[season1_ts[:window]]) == [1.0] * window
    assert list(md.season_start_mask[season1_ts[window:]]) == [0.0] * (len(season1_ts) - window)
    assert list(md.season_start_mask[season2_ts[:window]]) == [1.0] * window
    assert list(md.season_start_mask[season2_ts[window:]]) == [0.0] * (len(season2_ts) - window)


def test_max_round_excludes_future_rounds_no_leakage(engineered_two_season_df):
    max_round = engineered_two_season_df["round"].max() - 2
    md = prepare_model_data(engineered_two_season_df, max_round=max_round)
    assert md.t_idx.max() <= max_round


def test_active_mask_and_season_start_mask_shapes(two_season_model_data):
    md = two_season_model_data
    assert md.active_mask.shape == (md.n_time, md.n_teams)
    assert md.season_start_mask.shape == (md.n_time,)
    # every row has at least one active team (no all-zero rounds)
    assert (md.active_mask.sum(axis=1) > 0).all()


def test_one_observation_row_per_match_not_two(two_season_model_data):
    """Regression test: each match must contribute exactly one row to the
    training arrays. Keeping both the home- and away-perspective row would
    feed every match's outcome into the Poisson likelihood twice — once
    correctly (home team's goals include their home_adv bonus) and once
    under a mismatched mean function that omits it."""
    md = two_season_model_data
    assert len(md.t_idx) == md.n_matches
    assert len(md.goals_home) == md.n_matches
    # every surviving observation is a home-perspective row
    assert (md.home == 1.0).all()


def test_goals_home_matches_true_home_team_score(engineered_two_season_df, two_season_model_data):
    """Cross-check against the raw dataframe: goals_home for each surviving
    observation must equal what the actual home team scored in that match,
    not a duplicate/swapped value from the dropped away-perspective row."""
    md = two_season_model_data
    idx_to_team = {v: k for k, v in md.team_mapping.items()}

    home_rows = engineered_two_season_df[engineered_two_season_df["is_home"] == 1]
    true_goals_by_match = home_rows.set_index("match_id")["goals_home"]

    for i in range(len(md.match_idx)):
        match_id = md.match_idx[i]
        home_team = idx_to_team[md.team_idx[i]]
        assert md.goals_home[i] == true_goals_by_match.loc[match_id]
        assert home_rows.set_index("match_id").loc[match_id, "team"] == home_team
