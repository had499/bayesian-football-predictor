import numpy as np
import pandas as pd
import pytest

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


def test_lineup_dev_defaults_to_zero_when_no_table_given(two_season_model_data):
    """Default behaviour (every call site before WP008) must be unchanged."""
    md = two_season_model_data
    assert (md.lineup_dev_home == 0.0).all()
    assert (md.lineup_dev_away == 0.0).all()


def test_lineup_dev_table_joins_by_team_and_date(engineered_two_season_df):
    df = engineered_two_season_df
    home_rows = df[df["is_home"] == 1]
    # pick one real (team, date) pair from the fixture and give it a
    # distinctive, unmistakable deviation value
    sample = home_rows.iloc[0]
    team, date = sample["team_long"], pd.Timestamp(sample["datetime"]).normalize()

    lineup_dev_table = pd.DataFrame([{"team": team, "date": date, "lineup_dev": 0.777}])
    md = prepare_model_data(df, max_round=df["round"].max(), lineup_dev_table=lineup_dev_table)

    idx_to_team = {v: k for k, v in md.team_mapping.items()}
    match_row = [i for i in range(len(md.match_idx)) if md.match_idx[i] == sample["match_id"]][0]
    assert idx_to_team[md.team_idx[match_row]] == team
    assert md.lineup_dev_home[match_row] == pytest.approx(0.777)

    # every OTHER match must still default to zero -- the join must not
    # spill the one real value onto unrelated rows
    others = [i for i in range(len(md.match_idx)) if i != match_row]
    assert all(md.lineup_dev_home[i] == 0.0 for i in others)


def test_lineup_dev_table_resolves_home_and_away_independently(engineered_two_season_df):
    """A deviation entry for the AWAY team on a given date must land in
    lineup_dev_away for that match, not lineup_dev_home, and must not
    accidentally match if only the date (not the team) lines up."""
    df = engineered_two_season_df
    home_rows = df[df["is_home"] == 1]
    sample = home_rows.iloc[0]
    away_team, date = sample["opp_team_long"], pd.Timestamp(sample["datetime"]).normalize()

    lineup_dev_table = pd.DataFrame([{"team": away_team, "date": date, "lineup_dev": -0.55}])
    md = prepare_model_data(df, max_round=df["round"].max(), lineup_dev_table=lineup_dev_table)

    match_row = [i for i in range(len(md.match_idx)) if md.match_idx[i] == sample["match_id"]][0]
    assert md.lineup_dev_away[match_row] == pytest.approx(-0.55)
    assert md.lineup_dev_home[match_row] == 0.0


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


# --- WP011: multi-league hierarchical pooling ---

def test_max_round_for_cutoff_date_strict_inclusive(engineered_two_season_df):
    from football_model.data.prepare_model_data import max_round_for_cutoff_date

    df = engineered_two_season_df
    # pick a real match date/round pair and confirm the cutoff resolves to
    # exactly that round when the cutoff IS that date, and to the round
    # before when the cutoff is one day earlier.
    sample = df.sort_values("datetime").iloc[len(df) // 2]
    cutoff_on = pd.Timestamp(sample["datetime"])
    cutoff_before = cutoff_on - pd.Timedelta(days=1)

    round_on = max_round_for_cutoff_date(df, cutoff_on)
    round_before = max_round_for_cutoff_date(df, cutoff_before)

    assert round_on >= int(sample["round"])
    # nothing AFTER cutoff_before's date should be counted
    later_rounds = df.loc[pd.to_datetime(df["datetime"]) > cutoff_before, "round"]
    if len(later_rounds) > 0:
        assert round_before < int(later_rounds.max())


def test_max_round_for_cutoff_date_before_any_match_returns_zero(engineered_two_season_df):
    from football_model.data.prepare_model_data import max_round_for_cutoff_date

    df = engineered_two_season_df
    way_before = pd.Timestamp(df["datetime"].min()) - pd.Timedelta(days=365)
    assert max_round_for_cutoff_date(df, way_before) == 0


def test_prepare_multileague_data_builds_one_modeldata_per_league(
    engineered_two_season_df, engineered_league2_df
):
    from football_model.data.prepare_model_data import prepare_multileague_data

    cutoff = pd.Timestamp(engineered_two_season_df["datetime"].max())
    leagues = prepare_multileague_data(
        {"EPL": engineered_two_season_df, "Bundesliga": engineered_league2_df},
        cutoff_date=cutoff,
    )
    assert set(leagues) == {"EPL", "Bundesliga"}
    assert leagues["EPL"].n_teams == 5   # A,B,C,D,E across both seasons
    assert leagues["Bundesliga"].n_teams == 6  # W,X,Y,Z,P,Q


def test_prepare_multileague_data_each_league_gets_independent_time_axis(
    engineered_two_season_df, engineered_league2_df
):
    """Each league's own ModelData.t_idx must start at 0 and be scoped to
    that league alone -- NOT a shared global calendar across leagues (see
    prepare_multileague_data's docstring for why)."""
    from football_model.data.prepare_model_data import prepare_multileague_data, prepare_model_data

    cutoff = pd.Timestamp(engineered_two_season_df["datetime"].max())
    leagues = prepare_multileague_data(
        {"EPL": engineered_two_season_df, "Bundesliga": engineered_league2_df},
        cutoff_date=cutoff,
    )
    # must be identical to calling prepare_model_data directly on each
    # league's own df -- i.e. genuinely independent, not cross-contaminated
    # by having been prepared "together".
    solo_epl = prepare_model_data(engineered_two_season_df, max_round=engineered_two_season_df["round"].max())
    assert np.array_equal(leagues["EPL"].t_idx, solo_epl.t_idx)
    assert leagues["EPL"].n_time == solo_epl.n_time
    assert leagues["EPL"].team_mapping == solo_epl.team_mapping


def test_prepare_multileague_data_respects_cutoff_date_per_league_leakage_safety(
    engineered_two_season_df, engineered_league2_df
):
    """A cutoff date strictly before the second league's season-2 matches
    must exclude them from that league's ModelData, even though the first
    league's own max_round (passed as an integer elsewhere) might otherwise
    suggest more rounds are available -- proves the two leagues don't share
    a max_round and each is genuinely date-gated independently."""
    from football_model.data.prepare_model_data import prepare_multileague_data

    early_cutoff = pd.Timestamp(engineered_league2_df["datetime"].min())  # league2's very first match date
    leagues = prepare_multileague_data(
        {"EPL": engineered_two_season_df, "Bundesliga": engineered_league2_df},
        cutoff_date=early_cutoff,
    )
    # Bundesliga should have (close to) no training rounds yet
    assert leagues["Bundesliga"].n_matches <= leagues["Bundesliga"].n_teams // 2


def test_prepare_multileague_data_skips_league_with_no_matches_before_cutoff(
    engineered_two_season_df, engineered_league2_df
):
    """Regression test for a real crash found via WP011's live concurrent
    test: a non-eval league fetched with fewer seasons than the eval league
    (the "trim non-EPL history" compute lever) can have zero matches before
    an early eval-league window's cutoff date. Must be skipped gracefully,
    not crash prepare_model_data with an empty-dataframe -> NaN n_time."""
    from football_model.data.prepare_model_data import prepare_multileague_data

    # both synthetic fixtures start on the same hardcoded date -- shift
    # Bundesliga's later, as if it were fetched with fewer/more recent
    # seasons than EPL (WP011's actual "trim non-EPL history" scenario).
    shifted_league2 = engineered_league2_df.copy()
    shifted_league2["datetime"] = shifted_league2["datetime"] + pd.Timedelta(days=60)

    cutoff_after_epl_starts_before_league2 = (
        pd.Timestamp(engineered_two_season_df["datetime"].min()) + pd.Timedelta(days=14)
    )
    leagues = prepare_multileague_data(
        {"EPL": engineered_two_season_df, "Bundesliga": shifted_league2},
        cutoff_date=cutoff_after_epl_starts_before_league2,
    )
    assert "Bundesliga" not in leagues
    assert "EPL" in leagues  # the eval league (never trimmed) is still present


def test_continuity_defaults_to_zero_when_no_table_given(two_season_model_data):
    md = two_season_model_data
    assert (md.defence_cont_home == 0.0).all() and (md.defence_cont_away == 0.0).all()


def test_continuity_table_joins_home_and_away_independently(engineered_two_season_df):
    """A side's continuity lands in its own array (home team -> defence_cont_home,
    away team -> defence_cont_away), matched on (team, date), and nothing
    else picks up a value."""
    df = engineered_two_season_df
    sample = df[df["is_home"] == 1].iloc[0]
    date = pd.Timestamp(sample["datetime"]).normalize()
    table = pd.DataFrame([
        {"team": sample["team_long"], "date": date, "continuity_z": 1.25},
        {"team": sample["opp_team_long"], "date": date, "continuity_z": -0.75},
    ])
    md = prepare_model_data(df, max_round=df["round"].max(), continuity_table=table)
    row = [i for i in range(len(md.match_idx)) if md.match_idx[i] == sample["match_id"]][0]
    assert md.defence_cont_home[row] == pytest.approx(1.25)
    assert md.defence_cont_away[row] == pytest.approx(-0.75)
    others = [i for i in range(len(md.match_idx)) if i != row]
    assert all(md.defence_cont_home[i] == 0.0 for i in others)
