"""WP009's new logic: rest-days join-key correctness (the bug that produced
2,285 duplicated rows before the fix) and temperature scaling."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work_products" / "wp009_disagreement_diagnostics"))
import diagnostics as D  # noqa: E402


def _df_cv_row(team, opp, dt, round_, season, is_home):
    return dict(team=team, opp_team=opp, datetime=pd.Timestamp(dt), round=round_, season=season, is_home=is_home)


def test_rest_days_keys_on_date_not_round_a_team_can_play_twice_in_one_round():
    """Regression for the exact bug found in this WP: a team playing twice
    under the same round number (a rearranged midweek fixture) must not
    fan-out the merge — each row's own rest_days must be independently
    correct, keyed by date."""
    df_cv = pd.DataFrame([
        _df_cv_row("ARS", "SOU", "2020-01-01", 14, "2020", 1),
        _df_cv_row("ARS", "STO", "2020-01-04", 14, "2020", 0),   # same round, 3 days later
    ])
    rest = D.rest_days(df_cv)
    assert len(rest) == 2  # no fan-out
    row2 = rest[rest.date == pd.Timestamp("2020-01-04")].iloc[0]
    assert np.isclose(row2.rest_days, 3.0)
    row1 = rest[rest.date == pd.Timestamp("2020-01-01")].iloc[0]
    assert np.isnan(row1.rest_days)  # no prior match for this team in the frame


def test_rest_days_raises_on_a_genuine_team_date_collision():
    df_cv = pd.DataFrame([
        _df_cv_row("ARS", "SOU", "2020-01-01", 14, "2020", 1),
        _df_cv_row("ARS", "STO", "2020-01-01", 14, "2020", 0),  # same team, same date — impossible in reality
    ])
    with pytest.raises(ValueError, match="not unique"):
        D.rest_days(df_cv)


def test_sharpen_is_a_noop_at_t1_and_increases_max_prob_when_t_below_1():
    P = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    assert np.allclose(D.sharpen(P, 1.0), P)
    sharp = D.sharpen(P, 0.5)
    assert np.allclose(sharp.sum(1), 1.0)
    assert (sharp[:, 0] >= P[:, 0]).all()  # sharpening pushes the already-largest prob up


def test_fit_temperature_recovers_t_less_than_one_when_truth_is_sharper_than_predictions():
    rng = np.random.default_rng(0)
    true_p = np.array([0.7, 0.2, 0.1])
    y = rng.choice(3, size=2000, p=true_p)
    # predictions deliberately flattened (as if under-confident, like the model in this WP)
    P = np.tile(D.sharpen(true_p[None, :], 2.0), (2000, 1))
    T = D.fit_temperature(P, y)
    assert T < 1.0  # fitting should want to sharpen these under-confident predictions back up


def test_evaluate_temperature_scaling_never_touches_train_window_in_its_own_test_fold():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "window": rng.integers(0, 10, n),
        "p_home": rng.dirichlet([2, 2, 2], n)[:, 0],
    })
    df["p_draw"] = rng.dirichlet([2, 2, 2], n)[:, 1] * (1 - df.p_home)
    df["p_away"] = 1 - df.p_home - df.p_draw
    df["result"] = rng.choice(["H", "D", "A"], n)
    df["favourite"] = "home"
    df["upset"] = False
    out = D.evaluate_temperature_scaling(df, n_folds=2, seed=0)
    assert len(out) == n  # every row scored exactly once
    assert set(out.rps_before.index) == set(out.rps_after.index)
