import numpy as np
import pandas as pd
import pytest

from football_model.features.add_metadata import (
    add_rounds_to_data,
    add_match_ids,
    add_home_away_goals_xg,
)


def make_synthetic_league_df(season_team_lists, rounds_per_season=6, seed=0):
    """Build a synthetic multi-season match dataframe in the same shape
    get_understat_data() produces, before feature engineering.

    season_team_lists: list of (season_label, [team names]) pairs, one per
    season, in chronological order. Each season round-robins its own team
    list, so teams that don't appear in a given season are absent from it
    (mirroring what happens with real relegation/promotion).
    """
    rng = np.random.default_rng(seed)
    rows = []
    dt = pd.Timestamp("2023-08-01")
    for season, teams in season_team_lists:
        assert len(teams) % 2 == 0, "synthetic fixture needs an even team count"
        for rnd in range(1, rounds_per_season + 1):
            # All fixtures in a round share one date, so add_rounds_to_data's
            # ISO-week grouping sees exactly one "round" per week, matching
            # the round number we intend here.
            for i in range(0, len(teams), 2):
                home, away = teams[i], teams[i + 1]
                gh, ga = rng.poisson(1.4), rng.poisson(1.1)
                for team, opp, is_home, goals, goals_against in [
                    (home, away, 1, gh, ga),
                    (away, home, 0, ga, gh),
                ]:
                    rows.append(
                        dict(
                            team=team,
                            opp_team=opp,
                            is_home=is_home,
                            goals=goals,
                            goals_against=goals_against,
                            xG=goals + 0.1,
                            xGA=goals_against + 0.1,
                            datetime=dt,
                            season=season,
                            round=rnd,
                        )
                    )
            dt += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


@pytest.fixture
def two_season_teams():
    """Season 1: A, B, C, D. Season 2: D relegated, E promoted."""
    return [
        ("2023", ["A", "B", "C", "D"]),
        ("2024", ["A", "B", "C", "E"]),
    ]


@pytest.fixture
def raw_two_season_df(two_season_teams):
    return make_synthetic_league_df(two_season_teams, rounds_per_season=6, seed=0)


@pytest.fixture
def engineered_two_season_df(raw_two_season_df):
    df = add_rounds_to_data(raw_two_season_df)
    df = add_match_ids(df)
    df = add_home_away_goals_xg(df)
    return df


@pytest.fixture
def two_season_model_data(engineered_two_season_df):
    from football_model.data.prepare_model_data import prepare_model_data

    return prepare_model_data(
        engineered_two_season_df, max_round=engineered_two_season_df["round"].max()
    )
