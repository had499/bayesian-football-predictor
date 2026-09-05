import pandas as pd

from football_model.features.add_metadata import (
    add_rounds_to_data,
    add_match_ids,
    add_home_away_goals_xg,
)


def test_cum_round_is_monotonic_and_continuous_across_seasons(raw_two_season_df):
    df = add_rounds_to_data(raw_two_season_df)

    # Season 1 occupies rounds 1..6, season 2 continues at 7..12 (no gap, no overlap)
    season1_rounds = sorted(df[df["season"] == "2023"]["round"].unique())
    season2_rounds = sorted(df[df["season"] == "2024"]["round"].unique())

    assert season1_rounds == list(range(1, 7))
    assert season2_rounds == list(range(7, 13))


def test_add_match_ids_gives_each_fixture_one_id_shared_by_both_rows(raw_two_season_df):
    df = add_rounds_to_data(raw_two_season_df)
    df = add_match_ids(df)

    # Every match appears as two rows (home perspective + away perspective)
    # sharing a match_id, and every match_id covers exactly two rows.
    counts = df.groupby("match_id").size()
    assert (counts == 2).all()


def test_add_home_away_goals_xg_is_row_relative(raw_two_season_df):
    """goals_home/goals_away (and xG_home/xG_away) are relative to *this
    row's* team/opponent, not the literal home/away side of the match —
    the model's linear predictor (theta_home/theta_away in model.py) is
    built the same way, gating the home-advantage bonus with `is_home`
    rather than assuming goals_home always means the true home team. Every
    row should carry its own team's goals as goals_home and its opponent's
    as goals_away, regardless of is_home."""
    df = add_rounds_to_data(raw_two_season_df)
    df = add_match_ids(df)
    df = add_home_away_goals_xg(df)

    assert (df["goals_home"] == df["goals"]).all()
    assert (df["goals_away"] == df["goals_against"]).all()
    assert (df["xG_home"] == df["xG"]).all()
    assert (df["xG_away"] == df["xGA"]).all()
