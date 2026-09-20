import numpy as np
import pandas as pd
import pytest

from football_model.features.continuity_features import build_continuity_table

# A fixed XI: 1 GK, 4 defenders, 3 midfielders (one attacking), 3 forwards -> 11.
XI = [
    ("gk", "GK"), ("d1", "DC"), ("d2", "DC"), ("d3", "DR"), ("d4", "DL"),
    ("m1", "MC"), ("m2", "MC"), ("m3", "AMC"),
    ("f1", "FW"), ("f2", "FWL"), ("f3", "FWR"),
]


def make_logs(n_matches, team="A", start="2023-08-05", spacing_days=7, season="2023",
              lineup_for=None, season_for=None):
    """Player-match logs for one team. `lineup_for(i)` returns the list of
    (player_id, position) that started match i (default: the fixed XI)."""
    rows = []
    for i in range(n_matches):
        date = pd.Timestamp(start) + pd.Timedelta(days=spacing_days * i)
        lineup = lineup_for(i) if lineup_for else XI
        for pid, pos in lineup:
            rows.append({"team": team, "date": date, "season": season_for(i) if season_for else season,
                         "player_id": pid, "position": pos, "started": True})
        rows.append({"team": team, "date": date, "season": season, "player_id": "bench",
                     "position": "Sub", "started": False})
    return pd.DataFrame(rows)


def row_at(table, i, start="2023-08-05", spacing_days=7, team="A"):
    date = pd.Timestamp(start) + pd.Timedelta(days=spacing_days * i)
    return table[(table["team"] == team) & (table["date"] == date)].iloc[0]


def test_unchanged_xi_gives_continuity_one_after_min_history():
    t = build_continuity_table(make_logs(20), window=10, min_history=5, neutral_first_n_of_season=0)
    r = row_at(t, 12)
    for unit in ("defence", "midfield", "attack", "all"):
        assert r[f"continuity_{unit}"] == pytest.approx(1.0)
    assert r["n_history"] == 10


def test_early_rows_are_nan_until_min_history():
    t = build_continuity_table(make_logs(10), window=10, min_history=5, neutral_first_n_of_season=0)
    assert [np.isnan(row_at(t, i)["continuity_all"]) for i in range(7)] == [True] * 5 + [False] * 2
    assert row_at(t, 3)["n_history"] == 3


def test_one_new_defender_lowers_only_the_defence_unit():
    def lineup(i):
        if i == 8:  # d4 is replaced by a first-time starter
            return [(p if p != "d4" else "new", pos) for p, pos in XI]
        return XI
    t = build_continuity_table(make_logs(12, lineup_for=lineup), min_history=5, neutral_first_n_of_season=0)
    r = row_at(t, 8)
    # defence unit = gk, d1, d2, d3 (share 1.0 each) + new (share 0.0) -> 4/5
    assert r["continuity_defence"] == pytest.approx(0.8)
    assert r["continuity_midfield"] == pytest.approx(1.0)
    assert r["continuity_attack"] == pytest.approx(1.0)
    assert r["continuity_all"] == pytest.approx(10 / 11)


def test_start_share_counts_starts_in_any_position():
    """A player who always started as a DC but plays DMC today still has share
    1.0 and now counts in the midfield unit — formation changes do not make
    a regular look 'unusual'."""
    def lineup(i):
        if i == 8:
            return [(p, "DMC" if p == "d2" else pos) for p, pos in XI]
        return XI
    t = build_continuity_table(make_logs(12, lineup_for=lineup), min_history=5, neutral_first_n_of_season=0)
    r = row_at(t, 8)
    assert r["continuity_midfield"] == pytest.approx(1.0)
    assert r["continuity_defence"] == pytest.approx(1.0)


def test_only_the_last_window_matches_count():
    """A player who started matches 0-9 but none of 10-19 has share 0 at match
    20 with window=10 (only matches 10-19 count)."""
    def lineup(i):
        return XI if i < 10 else [(p if p != "d1" else "sub_d1", pos) for p, pos in XI]
    def lineup2(i):
        return lineup(i) if i < 20 else XI          # d1 returns at match 20
    t = build_continuity_table(make_logs(21, lineup_for=lineup2), window=10, min_history=5,
                               neutral_first_n_of_season=0)
    # at match 20: d1 share = 0/10; defence unit gk, d1, d2, d3, d4 -> 4/5
    assert row_at(t, 20)["continuity_defence"] == pytest.approx(0.8)


def test_no_leakage_a_later_match_never_changes_an_earlier_row():
    def lineup(i):
        return XI if i < 15 else [(p if p != "gk" else "other_gk", pos) for p, pos in XI]
    full = build_continuity_table(make_logs(20, lineup_for=lineup), neutral_first_n_of_season=0)
    truncated = build_continuity_table(make_logs(15, lineup_for=lineup), neutral_first_n_of_season=0)
    pd.testing.assert_frame_equal(
        full.iloc[:15].reset_index(drop=True), truncated.reset_index(drop=True))
    assert full.iloc[15]["continuity_defence"] < 1.0        # ...and the change itself is visible


def test_first_matches_of_a_new_season_are_neutral_nan():
    seasons = lambda i: "2023" if i < 12 else "2024"
    t = build_continuity_table(make_logs(20, season_for=seasons), min_history=5, neutral_first_n_of_season=5)
    later_season = [row_at(t, i)["continuity_all"] for i in range(12, 20)]
    assert all(np.isnan(v) for v in later_season[:5])       # 5 matches of new season: NaN
    assert not np.isnan(later_season[5])                    # then it is measured again
    assert not np.isnan(row_at(t, 8)["continuity_all"])     # previous season unaffected


def test_a_long_gap_resets_history():
    logs = pd.concat([
        make_logs(8, start="2021-08-14"),
        make_logs(8, start="2023-08-05", season="2023"),   # returns 2 years later
    ])
    t = build_continuity_table(logs, min_history=5, neutral_first_n_of_season=0, max_gap_days=150)
    first_after_gap = t[t["date"] == pd.Timestamp("2023-08-05")].iloc[0]
    assert first_after_gap["n_history"] == 0 and np.isnan(first_after_gap["continuity_all"])
    assert not np.isnan(t[t["date"] == pd.Timestamp("2023-08-05") + pd.Timedelta(days=42)].iloc[0]["continuity_all"])


def test_substitutes_are_ignored_and_teams_are_independent():
    logs = pd.concat([make_logs(12, team="A"), make_logs(12, team="B")])
    t = build_continuity_table(logs, min_history=5, neutral_first_n_of_season=0)
    assert set(t["team"]) == {"A", "B"} and len(t) == 24
    # the 'bench' player never started, so nothing should ever count him
    assert (t["continuity_all"].dropna() == 1.0).all()


def test_continuity_feature_table_standardises_and_maps_nan_to_zero():
    from football_model.features.continuity_features import continuity_feature_table
    table = pd.DataFrame({
        "team": ["A", "A", "A"], "date": pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-15"]),
        "continuity_defence": [0.714 + 0.14, np.nan, 0.714 - 0.28],
    })
    out = continuity_feature_table(table)
    assert out["continuity_z"].tolist() == pytest.approx([1.0, 0.0, -2.0])
    assert list(out.columns) == ["team", "date", "continuity_z"]
