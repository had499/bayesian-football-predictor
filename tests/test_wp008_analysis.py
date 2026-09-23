"""WP008's odds-joining logic — the part that's new here (bootstrap/RPS/de-vig
statistics are WP007's, already tested in test_wp007_analysis.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work_products" / "wp008_market_benchmark"))
import market_analysis as MA  # noqa: E402


def _model_row(window, round_, home, away, date, gh, ga, result):
    return dict(window=window, round=round_, home=home, away=away, date=pd.Timestamp(date),
               season="2021", result=result, goals_home=gh, goals_away=ga)


def _odds_row(date, home_fd, away_fd, fthg, ftag, pin=(1.8, 3.6, 4.6)):
    row = dict(date=pd.Timestamp(date), home_team=home_fd, away_team=away_fd, fthg=fthg, ftag=ftag)
    for col, val in zip(MA.PIN, pin):
        row[col] = val
    for cols in (MA.B365, MA.AVG, MA.MAX):
        for col in cols:
            row[col] = np.nan
    return row


def test_join_market_matches_on_mapped_team_names_and_keeps_all_books_separate():
    model = pd.DataFrame([_model_row(1, 5, "ARS", "MUN", "2021-01-01", 2, 1, "H")])
    odds = pd.DataFrame([_odds_row("2021-01-01", "Arsenal", "Man United", 2, 1)])
    j, unmatched, mismatched = MA.join_market(model, odds)
    assert len(j) == 1 and len(unmatched) == 0 and len(mismatched) == 0
    assert np.isclose(j["pinnacle_h"].iloc[0] + j["pinnacle_d"].iloc[0] + j["pinnacle_a"].iloc[0], 1.0)
    assert j["bet365_h"].isna().all()  # no bet365 odds supplied — must stay NaN, not silently borrow pinnacle


def test_join_market_drops_score_mismatches_instead_of_silently_mis_scoring():
    model = pd.DataFrame([_model_row(1, 5, "ARS", "MUN", "2021-01-01", 2, 1, "H")])
    odds = pd.DataFrame([_odds_row("2021-01-01", "Arsenal", "Man United", 3, 1)])  # wrong score at that date
    j, unmatched, mismatched = MA.join_market(model, odds)
    assert len(j) == 0 and len(mismatched) == 1


def test_join_market_reports_unmatched_when_no_odds_row_exists():
    model = pd.DataFrame([_model_row(1, 5, "ARS", "MUN", "2021-01-01", 2, 1, "H")])
    odds = pd.DataFrame([_odds_row("2021-01-01", "Arsenal", "Chelsea", 2, 1)])  # different fixture
    j, unmatched, mismatched = MA.join_market(model, odds)
    assert len(j) == 0 and len(unmatched) == 1


def test_join_market_raises_on_an_unknown_team_code():
    model = pd.DataFrame([_model_row(1, 5, "ZZZ", "MUN", "2021-01-01", 2, 1, "H")])
    odds = pd.DataFrame([_odds_row("2021-01-01", "Arsenal", "Man United", 2, 1)])
    with pytest.raises(ValueError, match="CODE_TO_FD"):
        MA.join_market(model, odds)


def test_add_devig_probs_leaves_missing_odds_as_nan_not_zero():
    df = pd.DataFrame({"a": [1.8, np.nan], "b": [3.6, 3.6], "c": [4.6, 4.6]})
    out = MA.add_devig_probs(df.copy(), ["a", "b", "c"], "x")
    assert np.isclose(out.loc[0, ["x_h", "x_d", "x_a"]].sum(), 1.0)
    assert out.loc[1, ["x_h", "x_d", "x_a"]].isna().all()
    assert np.isclose(out.loc[0, "overround_x"], 1 / 1.8 + 1 / 3.6 + 1 / 4.6 - 1.0)
    assert np.isnan(out.loc[1, "overround_x"])


def test_book_rps_only_uses_rows_with_that_books_odds():
    joined = pd.DataFrame({
        "pinnacle_h": [0.5, np.nan], "pinnacle_d": [0.3, np.nan], "pinnacle_a": [0.2, np.nan],
        "result": ["H", "A"],
    })
    r, sub = MA.book_rps(joined, "pinnacle")
    assert len(r) == 1 and len(sub) == 1
    assert np.isclose(r[0], 0.5 * ((0.5 - 1) ** 2 + (0.8 - 1) ** 2))
