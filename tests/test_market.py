import numpy as np
import pandas as pd
import pytest

from football_model.evaluation import market as mk


# ---------------------------------------------------------------- basics

def test_devig_rows_sum_to_one_and_keep_ordering():
    odds = np.array([[2.0, 3.5, 4.0], [1.5, 4.0, 7.0]])
    p = mk.devig(odds)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert (np.argsort(p, axis=1) == np.argsort(-odds, axis=1)).all()


def test_devig_fair_book_is_identity_of_implied_probs():
    p = np.array([[0.5, 0.3, 0.2]])
    np.testing.assert_allclose(mk.devig(1.0 / p), p)


def test_outcome_onehot_and_rejects_bad_labels():
    np.testing.assert_array_equal(mk.outcome_onehot(["H", "D", "A"]), np.eye(3))
    with pytest.raises(ValueError):
        mk.outcome_onehot(["H", "X"])


def test_rps_hand_computed_values():
    onehot = mk.outcome_onehot(["H", "A"])
    uniform = np.full((2, 3), 1 / 3)
    # H with uniform: 0.5 * ((1/3-1)^2 + (2/3-1)^2) = 5/18; A is the mirror case.
    np.testing.assert_allclose(mk.rps(uniform, onehot), [5 / 18, 5 / 18])
    perfect = onehot.copy()
    np.testing.assert_allclose(mk.rps(perfect, onehot), 0.0)


def test_rps_penalises_far_misses_more_than_near_misses():
    home = mk.outcome_onehot(["H"])
    near = np.array([[0.0, 1.0, 0.0]])  # predicted draw, was home
    far = np.array([[0.0, 0.0, 1.0]])   # predicted away, was home
    assert mk.rps(far, home)[0] > mk.rps(near, home)[0]


def test_blend_endpoints():
    a, b = np.array([[0.6, 0.3, 0.1]]), np.array([[0.2, 0.3, 0.5]])
    np.testing.assert_allclose(mk.blend(a, b, 0.0), a)
    np.testing.assert_allclose(mk.blend(a, b, 1.0), b)
    np.testing.assert_allclose(mk.blend(a, b, 0.5).sum(axis=1), 1.0)


# ------------------------------------------------------------- intervals

def test_wilson_interval_known_value():
    lo, hi = mk.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    assert all(np.isnan(mk.wilson_interval(0, 0)))


def test_bootstrap_ci_contains_mean_and_is_seeded():
    v = np.random.default_rng(1).normal(0.3, 1.0, 400)
    m, lo, hi = mk.bootstrap_ci(v, n_boot=2000, seed=7)
    assert lo < m < hi
    assert (m, lo, hi) == mk.bootstrap_ci(v, n_boot=2000, seed=7)


def test_ratio_bootstrap_degenerate_and_empty_cases():
    m, lo, hi = mk.ratio_bootstrap(np.full(50, 0.4), np.ones(50), n_boot=500)
    assert m == pytest.approx(0.4) and lo == pytest.approx(0.4) and hi == pytest.approx(0.4)
    assert all(np.isnan(x) for x in mk.ratio_bootstrap(np.zeros(10), np.zeros(10)))


def test_half_masks_partition_by_median_date():
    dates = pd.date_range("2021-01-01", periods=10)
    first, second = mk.half_masks(dates)
    assert (first ^ second).all()
    assert first.sum() == 5 and dates[first].max() < dates[second].min()


# ----------------------------------------------------------- bet settling

def test_settle_hand_computed():
    odds = np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    onehot = mk.outcome_onehot(["D", "A"])
    mask = np.array([[True, True, False], [False, False, True]])
    profit, stake = mk.settle(mask, odds, onehot)
    # match 0: home bet loses (-1), draw bet wins (+2); match 1: away bet wins (+3)
    np.testing.assert_allclose(profit, [1.0, 3.0])
    np.testing.assert_allclose(stake, [2.0, 1.0])


def test_edge_and_threshold_direction():
    p, odds = np.array([[0.5, 0.3, 0.2]]), np.array([[2.2, 3.0, 4.0]])
    np.testing.assert_allclose(mk.edge(p, odds), [[0.1, -0.1, -0.2]])


def test_roi_table_no_bets_gives_nan_not_a_crash():
    p = np.full((20, 3), 1 / 3)
    odds = np.full((20, 3), 2.5)  # -16.7% edge everywhere
    onehot = mk.outcome_onehot(["H"] * 20)
    row = mk.roi_table(p, odds, onehot, taus=[0.05], n_boot=100).iloc[0]
    assert row["n_bets"] == 0 and np.isnan(row["roi"])


def _simulate_market(n, book_multiplier, seed):
    """Matches whose true probabilities are `p`; the book quotes odds
    book_multiplier / p (so its true edge to a bettor is exactly
    book_multiplier - 1 on EVERY outcome)."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([6, 4, 5], size=n)
    u = rng.random(n)
    result = np.where(u < p[:, 0], "H", np.where(u < p[:, 0] + p[:, 1], "D", "A"))
    return p, book_multiplier / p, mk.outcome_onehot(result)


def test_pipeline_recovers_a_known_positive_edge():
    p, odds, onehot = _simulate_market(30000, 1.05, seed=3)
    row = mk.roi_table(p, odds, onehot, taus=[0.02], n_boot=1000).iloc[0]
    assert row["n_bets"] == 3 * 30000            # 5% edge clears a 2% threshold everywhere
    assert row["lo"] < 0.05 < row["hi"]          # truth inside the CI
    assert row["claimed_edge"] == pytest.approx(0.05)


def test_pipeline_recovers_a_known_margin_and_bets_nothing_on_a_negative_market():
    p, odds, onehot = _simulate_market(30000, 0.95, seed=4)
    everything = mk.roi_table(p, odds, onehot, taus=[-np.inf], n_boot=1000).iloc[0]
    assert everything["lo"] < -0.05 < everything["hi"]
    assert mk.roi_table(p, odds, onehot, taus=[0.02], n_boot=100).iloc[0]["n_bets"] == 0


def test_a_wrong_model_does_not_manufacture_an_edge():
    """Bettor believes probabilities drawn independently of the truth, against
    a fair (multiplier 1.0) book. Odds are fair, so whatever the selection
    rule picks, expected profit per bet is exactly zero: the ROI CI must
    contain 0. This is the guard against the selection/bootstrap code
    flattering a model that has no information."""
    p, odds, onehot = _simulate_market(30000, 1.0, seed=5)
    belief = np.random.default_rng(99).dirichlet([6, 4, 5], size=len(p))
    row = mk.roi_table(belief, odds, onehot, taus=[0.05], n_boot=1000).iloc[0]
    assert row["n_bets"] > 1000                      # it really did bet
    assert row["claimed_edge"] > 0.05                # ...believing it had an edge
    assert row["lo"] < 0.0 < row["hi"]               # ...that the data does not support


def test_odds_band_table_partitions_all_bets():
    p, odds, onehot = _simulate_market(2000, 1.0, seed=6)
    edges = [1.0, 2.0, 4.0, 10.0, 1e9]
    t = mk.odds_band_table(odds, onehot, edges, n_boot=200)
    assert t["n_bets"].sum() == odds.size


def test_clv_positive_when_bets_are_placed_at_better_than_closing_prices():
    p_close = np.full((100, 3), 1 / 3)
    open_odds = np.full((100, 3), 3.3)   # 10% better than the fair 3.0 close
    mask = np.ones((100, 3), bool)
    res = mk.clv_table(open_odds, p_close, mask, n_boot=200)
    assert res["clv"] == pytest.approx(0.1)
    assert res["clv_lo"] == pytest.approx(0.1) and res["n_bets"] == 300


def test_calibration_table_counts_and_rates():
    p = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
    hit = np.array([0, 1, 1, 1, 1, 1])
    t = mk.calibration_table(p, hit, [0.0, 0.3, 0.7, 1.0])
    assert t["n"].tolist() == [2, 2, 2]
    assert t["rate"].tolist() == [0.5, 1.0, 1.0]


# ---------------------------------------------------- residual information

def test_logodds_features_shape_and_value():
    f = mk.logodds_features(np.array([[0.5, 0.25, 0.25]]))
    np.testing.assert_allclose(f, [[np.log(2), 0.0]])


def test_leave_group_out_logloss_prefers_informative_features():
    rng = np.random.default_rng(0)
    n = 900
    p = rng.dirichlet([3, 2, 3], size=n)
    y = np.array([rng.choice(3, p=row) for row in p])
    groups = np.repeat(np.arange(9), 100)
    informative = mk.leave_group_out_logloss(mk.logodds_features(p), y, groups)
    noise = mk.leave_group_out_logloss(rng.normal(size=(n, 2)), y, groups)
    assert informative.mean() < noise.mean()
    assert len(informative) == n and np.isfinite(informative).all()


# ------------------------------------------------------------ odds joining

def _fixtures_and_odds(home_goals_in_odds=1):
    fixtures = pd.DataFrame({
        "date": [pd.Timestamp("2022-01-01")], "home_fd": ["Arsenal"], "away_fd": ["Chelsea"],
        "goals_home": [1], "goals_away": [0],
    })
    odds = pd.DataFrame({
        "Date": [pd.Timestamp("2022-01-01")], "HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"],
        "FTHG": [home_goals_in_odds], "FTAG": [0], "PSCH": [2.0],
    })
    return fixtures, odds


def test_join_odds_happy_path_and_required_cols():
    fixtures, odds = _fixtures_and_odds()
    assert len(mk.join_odds(fixtures, odds, required_cols=["PSCH"])) == 1
    odds.loc[0, "PSCH"] = np.nan
    assert len(mk.join_odds(fixtures, odds, required_cols=["PSCH"])) == 0


def test_join_odds_raises_when_scores_disagree():
    fixtures, odds = _fixtures_and_odds(home_goals_in_odds=3)
    with pytest.raises(ValueError, match="disagrees"):
        mk.join_odds(fixtures, odds)
