"""WP018's state-dependent joint distribution. Two correctness properties matter:
it must reduce to independent Poisson when the state effect is switched off, and
it must preserve the input lambdas as marginal means — otherwise any measured
change would be a shifted mean rather than the dependence under test."""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work_products" / "wp018_score_state"))
import state_model as sm  # noqa: E402

LH = np.array([1.5, 2.4, 0.8, 1.0])
LA = np.array([1.2, 0.7, 1.6, 1.0])


def _exact(lh, la, K=sm.MAX_GOALS):
    g = np.arange(K + 1)
    P = poisson.pmf(g[None, :], lh[:, None])[:, :, None] * poisson.pmf(g[None, :], la[:, None])[:, None, :]
    return P / P.sum(axis=(1, 2), keepdims=True)


def test_reduces_to_independent_poisson_when_state_effect_is_off():
    got = sm.outcome_probs(sm.joint_distribution(LH, LA, mult=None))
    want = sm.outcome_probs(_exact(LH, LA))
    # first-order time discretisation; 16 steps/min leaves ~1e-4, far below the
    # ~1.4pp effects this work product examines
    assert np.abs(got - want).max() < 5e-4


def test_marginal_means_are_calibrated_back_to_the_input_lambdas():
    P = sm.joint_distribution(LH, LA, mult=sm.state_multipliers())
    gi, gj = np.meshgrid(np.arange(sm.MAX_GOALS + 1), np.arange(sm.MAX_GOALS + 1), indexing="ij")
    assert np.allclose((P * gi).sum(axis=(1, 2)), LH, atol=2e-3)
    assert np.allclose((P * gj).sum(axis=(1, 2)), LA, atol=2e-3)


def test_state_dependence_moves_probability_toward_draws():
    """A trailing side scoring faster pulls scorelines back toward level, so the
    draw mass must rise relative to independent Poisson."""
    base = sm.outcome_probs(sm.joint_distribution(LH, LA, mult=None))
    sd = sm.outcome_probs(sm.joint_distribution(LH, LA, mult=sm.state_multipliers()))
    assert (sd[:, 1] > base[:, 1]).all()


def test_outcome_probs_sum_to_one_and_order_is_home_draw_away():
    P = sm.joint_distribution(np.array([2.5]), np.array([0.5]), mult=sm.state_multipliers())
    p = sm.outcome_probs(P)[0]
    assert np.isclose(p.sum(), 1.0)
    assert p[0] > p[2]          # a much stronger home side must be favoured


def test_multipliers_are_flat_beyond_the_cap():
    m, cap = sm.state_multipliers(lead_mult=1.03, trail_mult=1.15, cap=2)
    assert m[cap] == m[2] and m[-cap] == m[-2] and m[0] == 1.0
