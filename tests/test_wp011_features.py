"""Tests for WP011's leak-safety and feature logic — the parts that would
silently invalidate every conclusion if wrong."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work_products" / "wp011_squad_instability"))
import features as F


def _roster(rows):
    df = pd.DataFrame(rows)
    df["contrib"] = df.xG + df.xA
    return df.sort_values("datetime").reset_index(drop=True)


def _match(mid, day, team, players, is_home=True):
    """players: list of (player_id, minutes, xG, position)"""
    return [dict(match_id=mid, datetime=pd.Timestamp(f"2021-01-{day:02d}"), season="2021",
                 team=team, is_home=is_home, player_id=pid, position=pos,
                 minutes=mins, xG=xg, xA=0.0)
            for pid, mins, xg, pos in players]


def _settled(pid_offset=0):
    return [(f"p{pid_offset+i}", 90.0, 0.1, "GK" if i == 0 else "DC") for i in range(11)] + \
           [(f"s{pid_offset+i}", 0.0, 0.0, "Sub") for i in range(3)]


def test_features_use_only_prior_matches_not_the_target_match():
    """The leak-safety guarantee: a wildly different XI in the TARGET match must
    not change that match's own features."""
    rows = []
    for d in range(1, 8):
        rows += _match(f"m{d}", d, "AAA", _settled())
    base = F.team_match_features(_roster(rows))
    f_before = base[base.match_id == "m7"].iloc[0]

    # same history, but the target match m7 fields an entirely different XI
    rows2 = [r for r in rows if r["match_id"] != "m7"] + _match("m7", 7, "AAA", _settled(pid_offset=100))
    f_after = F.team_match_features(_roster(rows2))
    f_after = f_after[f_after.match_id == "m7"].iloc[0]

    for c in ("instability", "absent_share", "staleness"):
        assert np.isclose(f_before[c], f_after[c]), f"{c} leaked from the target match"


def test_absent_share_flags_a_contributor_who_stopped_playing():
    """A striker who produced most of the xG then stopped featuring must push
    absent_share up — that's the 'key man is out' signal this WP is testing."""
    rows = []
    for d in range(1, 6):   # striker plays and scores
        rows += _match(f"m{d}", d, "AAA", _settled() + [("star", 90.0, 1.5, "FW")])
    for d in range(6, 10):  # striker disappears
        rows += _match(f"m{d}", d, "AAA", _settled())
    f = F.team_match_features(_roster(rows)).set_index("match_id")
    assert f.loc["m6", "absent_share"] == pytest.approx(0.0), "nothing absent while he's still playing"
    # must register after ONE missed match — the hard-threshold version needed four,
    # long after the market would have repriced (that's why it was replaced)
    assert f.loc["m7", "absent_share"] > 0.1
    assert f.loc["m7", "absent_share"] < f.loc["m8", "absent_share"] < f.loc["m9", "absent_share"]


def test_instability_is_higher_for_a_rotating_side_than_a_settled_one():
    settled, rotating = [], []
    for d in range(1, 10):
        settled += _match(f"s{d}", d, "SET", _settled())
        # rotating side swaps 5 of its XI every match
        off = 0 if d % 2 else 50
        rotating += _match(f"r{d}", d, "ROT", _settled(pid_offset=off))
    fs = F.team_match_features(_roster(settled))
    fr = F.team_match_features(_roster(rotating))
    assert fr.instability.mean() > fs.instability.mean()
    assert fs.instability.dropna().max() < 0.05   # a truly settled XI is ~0


def test_staleness_is_zero_for_a_settled_side_and_positive_after_turnover():
    settled = []
    for d in range(1, 12):
        settled += _match(f"m{d}", d, "AAA", _settled())
    fs = F.team_match_features(_roster(settled)).set_index("match_id")
    assert fs.loc["m11", "staleness"] == pytest.approx(0.0, abs=1e-9)

    turnover = [r for r in settled if int(r["match_id"][1:]) <= 6]
    for d in range(7, 12):          # squad turns over completely from m7
        turnover += _match(f"m{d}", d, "AAA", _settled(pid_offset=100))
    ft = F.team_match_features(_roster(turnover)).set_index("match_id")
    assert ft.loc["m7", "staleness"] == pytest.approx(0.0)      # turnover hasn't entered history yet
    assert ft.loc["m8", "staleness"] > 0.0                       # starts registering
    assert ft.loc["m10", "staleness"] >= 0.5                     # peaks mid-transition
    assert ft.loc["m11", "staleness"] < ft.loc["m10", "staleness"]  # settles as the new XI takes over


def test_attach_to_matches_does_not_change_row_count_or_silently_misjoin():
    feats = pd.DataFrame([
        dict(team="AAA", match_id="m1", datetime=pd.Timestamp("2021-01-05"),
             instability=0.2, absent_share=0.1, absent_drop=0.05, staleness=0.3),
        dict(team="BBB", match_id="m1", datetime=pd.Timestamp("2021-01-05"),
             instability=0.4, absent_share=0.2, absent_drop=0.15, staleness=0.5),
    ])
    matches = pd.DataFrame([dict(home="AAA", away="BBB", date=pd.Timestamp("2021-01-05"))])
    out = F.attach_to_matches(matches, feats)
    assert len(out) == 1
    assert np.isclose(out.h_instability.iloc[0], 0.2) and np.isclose(out.a_instability.iloc[0], 0.4)
    assert np.isclose(out.d_absent_share.iloc[0], 0.1 - 0.2)
    assert np.isclose(out.d_absent_drop.iloc[0], 0.05 - 0.15)
    assert np.isclose(out.max_instability.iloc[0], 0.4)
