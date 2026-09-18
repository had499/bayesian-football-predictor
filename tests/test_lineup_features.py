import numpy as np
import pandas as pd
import pytest

from football_model.features.lineup_features import (
    add_lineup_deviation,
    add_rolling_player_rate,
    build_lineup_deviation_table,
    compute_starting_xi_rate,
)


def _log(player_id, date, xG, xA, time, started=True, team="TeamA"):
    return {
        "player_id": player_id, "date": pd.Timestamp(date), "xG": xG, "xA": xA,
        "time": time, "started": started, "team": team,
    }


# ---------------------------------------------------------------------------
# add_rolling_player_rate — the leakage-critical stage
# ---------------------------------------------------------------------------

def test_rolling_rate_uses_only_strictly_earlier_matches():
    """The core leakage test: a player with a deliberately huge, obvious
    performance spike in their LAST match. An early match's rolling_rate
    must not be affected by it at all."""
    logs = pd.DataFrame([
        _log("p1", "2024-01-01", xG=0.1, xA=0.0, time=90),
        _log("p1", "2024-01-08", xG=0.1, xA=0.0, time=90),
        _log("p1", "2024-01-15", xG=0.1, xA=0.0, time=90),
        _log("p1", "2024-01-22", xG=5.0, xA=5.0, time=90),  # huge future spike
    ])
    out = add_rolling_player_rate(logs, population_rate=0.5)
    out = out.sort_values("date").reset_index(drop=True)

    # row 0 (the player's very first match): no history at all -> pure population rate
    assert out.loc[0, "trust_weight"] == 0.0
    assert out.loc[0, "rolling_rate"] == pytest.approx(0.5)

    # row 1: history is only row 0 (xG+xA=0.1 over 90 min -> 0.1/90*90 = 0.1 per-90),
    # nowhere near the future spike at row 3
    assert out.loc[1, "past_minutes"] == 90
    assert out.loc[1, "rolling_rate"] < 0.5  # pulled toward 0.1, not spiked

    # row 3 (the spike match itself): its OWN rolling_rate reflects rows 0-2 only
    # (contribution 0.1 each over 90 min -> own_rate=0.1, trust=270/900=0.3,
    # blended = 0.3*0.1 + 0.7*0.5 = 0.38), proving the spike hasn't leaked into
    # its own predictor value. A leaked version would include the spike's
    # contribution (5.0+5.0=10 over the extra 90 min) and come out far higher.
    assert out.loc[3, "past_minutes"] == 270
    assert out.loc[3, "rolling_rate"] == pytest.approx(0.38)


def test_rolling_rate_matches_hand_computation():
    logs = pd.DataFrame([
        _log("p1", "2024-01-01", xG=0.3, xA=0.1, time=90),   # contribution 0.4
        _log("p1", "2024-01-08", xG=0.2, xA=0.2, time=45),   # contribution 0.4
        _log("p1", "2024-01-15", xG=0.5, xA=0.0, time=90),   # contribution 0.5 (this row's own target)
    ])
    out = add_rolling_player_rate(logs, min_minutes_for_full_trust=900.0, population_rate=0.5)
    out = out.sort_values("date").reset_index(drop=True)

    # row 2's history = rows 0+1: past_minutes = 90+45=135, past_contribution = 0.4+0.4=0.8
    # own_rate = 0.8 / 135 * 90 = 0.5333...
    expected_own_rate = 0.8 / 135 * 90
    trust = 135 / 900.0
    expected_blended = trust * expected_own_rate + (1 - trust) * 0.5
    assert out.loc[2, "past_minutes"] == 135
    assert out.loc[2, "rolling_rate"] == pytest.approx(expected_blended)


def test_rolling_rate_debutant_is_pure_population_mean():
    logs = pd.DataFrame([_log("new_guy", "2024-01-01", xG=0.0, xA=0.0, time=90)])
    out = add_rolling_player_rate(logs, population_rate=0.42)
    assert out.loc[0, "trust_weight"] == 0.0
    assert out.loc[0, "rolling_rate"] == pytest.approx(0.42)


def test_trust_weight_caps_at_one():
    rows = [_log("p1", f"2024-01-{i+1:02d}", xG=0.2, xA=0.0, time=90) for i in range(20)]
    out = add_rolling_player_rate(pd.DataFrame(rows), min_minutes_for_full_trust=900.0)
    out = out.sort_values("date").reset_index(drop=True)
    # by row 10 (900 minutes of history behind it), trust should be exactly 1.0
    assert out.loc[10, "past_minutes"] == 900
    assert out.loc[10, "trust_weight"] == 1.0
    assert out["trust_weight"].max() <= 1.0


# ---------------------------------------------------------------------------
# compute_starting_xi_rate — substitute exclusion + trust-weighted aggregation
# ---------------------------------------------------------------------------

def test_starting_xi_rate_excludes_substitutes():
    logs = pd.DataFrame([
        _log("starter1", "2024-02-01", xG=0, xA=0, time=90, started=True),
        _log("starter2", "2024-02-01", xG=0, xA=0, time=90, started=True),
        _log("sub1", "2024-02-01", xG=0, xA=0, time=15, started=False),
    ])
    logs["rolling_rate"] = [1.0, 1.0, 100.0]  # sub has a wildly different rate
    logs["trust_weight"] = [1.0, 1.0, 1.0]

    out = compute_starting_xi_rate(logs)
    assert len(out) == 1
    assert out.loc[0, "n_starters"] == 2
    assert out.loc[0, "today_rate"] == pytest.approx(1.0)  # sub's 100.0 must not appear at all


def test_starting_xi_rate_is_trust_weighted():
    logs = pd.DataFrame([
        _log("a", "2024-02-01", xG=0, xA=0, time=90, started=True),
        _log("b", "2024-02-01", xG=0, xA=0, time=90, started=True),
    ])
    logs["rolling_rate"] = [1.0, 3.0]
    logs["trust_weight"] = [1.0, 0.0]  # b is a debutant, fully untrusted -> floored to 0.05

    out = compute_starting_xi_rate(logs)
    w_a, w_b = 1.0, 0.05
    expected = (1.0 * w_a + 3.0 * w_b) / (w_a + w_b)
    assert out.loc[0, "today_rate"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# add_lineup_deviation — team-level leakage boundary
# ---------------------------------------------------------------------------

def test_lineup_deviation_zero_for_teams_first_match():
    df = pd.DataFrame([{"team": "TeamA", "date": pd.Timestamp("2024-01-01"), "today_rate": 0.7, "n_starters": 11}])
    out = add_lineup_deviation(df)
    assert out.loc[0, "lineup_dev"] == pytest.approx(0.0, abs=1e-9)


def test_lineup_deviation_normal_rate_excludes_current_match():
    """The team-level leakage check: match 3's normal_rate must be the
    average of matches 1-2 ONLY, not influenced by match 3's own rate."""
    df = pd.DataFrame([
        {"team": "TeamA", "date": pd.Timestamp("2024-01-01"), "today_rate": 0.4, "n_starters": 11},
        {"team": "TeamA", "date": pd.Timestamp("2024-01-08"), "today_rate": 0.6, "n_starters": 11},
        {"team": "TeamA", "date": pd.Timestamp("2024-01-15"), "today_rate": 10.0, "n_starters": 11},  # big spike
    ])
    out = add_lineup_deviation(df).sort_values("date").reset_index(drop=True)
    assert out.loc[2, "normal_rate"] == pytest.approx((0.4 + 0.6) / 2)  # NOT influenced by 10.0
    assert out.loc[2, "lineup_dev"] > 0  # today's spike correctly reads as far above normal


def test_lineup_deviation_sign_matches_direction_of_change():
    df = pd.DataFrame([
        {"team": "TeamA", "date": pd.Timestamp("2024-01-01"), "today_rate": 0.5, "n_starters": 11},
        {"team": "TeamA", "date": pd.Timestamp("2024-01-08"), "today_rate": 0.5, "n_starters": 11},
        {"team": "TeamA", "date": pd.Timestamp("2024-01-15"), "today_rate": 0.1, "n_starters": 11},  # weaker XI
    ])
    out = add_lineup_deviation(df).sort_values("date").reset_index(drop=True)
    assert out.loc[2, "lineup_dev"] < 0  # a weaker-than-normal lineup reads negative


# ---------------------------------------------------------------------------
# end-to-end smoke
# ---------------------------------------------------------------------------

def test_build_lineup_deviation_table_end_to_end_smoke():
    logs = pd.DataFrame([
        _log("p1", "2024-01-01", xG=0.3, xA=0.1, time=90, started=True, team="TeamA"),
        _log("p2", "2024-01-01", xG=0.2, xA=0.0, time=90, started=True, team="TeamA"),
        _log("p3", "2024-01-01", xG=0.0, xA=0.0, time=10, started=False, team="TeamA"),
        _log("p1", "2024-01-08", xG=0.4, xA=0.0, time=90, started=True, team="TeamA"),
        _log("p2", "2024-01-08", xG=0.1, xA=0.1, time=90, started=True, team="TeamA"),
    ])
    out = build_lineup_deviation_table(logs, population_rate=0.3)
    assert set(out.columns) >= {"team", "date", "today_rate", "normal_rate", "lineup_dev", "n_starters"}
    assert len(out) == 2
    assert np.isfinite(out["lineup_dev"]).all()
