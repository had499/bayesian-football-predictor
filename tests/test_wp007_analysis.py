"""The WP007 readout's statistics — a wrong CI here would silently mislead every
conclusion the work product draws, so the helpers get real tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work_products" / "wp007_structural_fixes"))
import analysis as A  # noqa: E402


def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small, big = rng.normal(0.02, 0.1, 50), rng.normal(0.02, 0.1, 5000)
    m, lo, hi = A.bootstrap_mean_ci(small)
    assert lo < m < hi
    _, lo2, hi2 = A.bootstrap_mean_ci(big)
    assert (hi2 - lo2) < (hi - lo)


def test_cluster_bootstrap_is_wider_when_matches_within_a_group_are_correlated():
    # 20 windows x 10 matches; each window shares one shift, so matches aren't independent
    rng = np.random.default_rng(1)
    shifts = rng.normal(0, 0.05, 20)
    groups = np.repeat(np.arange(20), 10)
    v = shifts[groups] + rng.normal(0, 0.01, 200)
    _, lo_m, hi_m = A.bootstrap_mean_ci(v)
    _, lo_c, hi_c = A.bootstrap_mean_ci(v, groups=groups)
    assert (hi_c - lo_c) > 2 * (hi_m - lo_m)


def test_cluster_bootstrap_matches_plain_mean_and_handles_unequal_groups():
    v = np.array([1.0, 1.0, 1.0, 5.0])
    m, lo, hi = A.bootstrap_mean_ci(v, groups=np.array([0, 0, 0, 1]))
    assert np.isclose(m, 2.0) and lo <= m <= hi


def test_paired_diff_sign_and_verdict():
    a = np.full(200, 0.19) + np.random.default_rng(2).normal(0, 0.01, 200)
    b = np.full(200, 0.20) + np.random.default_rng(3).normal(0, 0.01, 200)
    d, lo, hi = A.paired_diff(a, b)
    assert d < 0 and hi < 0 and A.verdict(lo, hi) == "better"      # a has the lower RPS
    assert A.verdict(*A.paired_diff(b, a)[1:]) == "worse"
    assert A.verdict(-0.01, 0.01) == "no clear difference"


def test_devig_methods_return_probabilities_that_sum_to_one_and_keep_ordering():
    odds = [1.8, 3.6, 4.6]                                   # ~5% overround
    for f in (A.devig_proportional, A.devig_shin):
        p = f(odds)
        assert np.isclose(p.sum(), 1.0) and (p > 0).all()
        assert p[0] > p[1] > p[2]
    assert np.isclose(A.devig_proportional([2.0, 4.0, 4.0]).sum(), 1.0)


def test_rps_hda_known_values():
    p = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1/3, 1/3, 1/3]])
    r = A.rps_hda(p, ["H", "H", "D"])
    assert np.isclose(r[0], 0.0) and np.isclose(r[1], 1.0)
    assert np.isclose(r[2], 0.5 * ((1/3) ** 2 + (2/3 - 1) ** 2))


def test_wide_refuses_arms_with_different_match_sets():
    rows = [dict(arm="a", window=1, round=1, home="X", away="Y", rps=0.2),
            dict(arm="a", window=1, round=1, home="Z", away="W", rps=0.3),
            dict(arm="b", window=1, round=1, home="X", away="Y", rps=0.25)]   # b missing Z-W
    with pytest.raises(AssertionError, match="different matches"):
        A.wide(pd.DataFrame(rows), "rps", ["a", "b"])


def test_complete_windows_is_the_intersection_across_arms():
    m = pd.DataFrame({"arm": ["a", "a", "b"], "window": [1, 2, 1]})
    assert A.complete_windows(m, ["a", "b"]) == [1]


def test_reliability_table_bins_predicted_vs_actual():
    p = np.array([0.1, 0.12, 0.9, 0.88]); y = np.array([0, 0, 1, 1])
    t = A.reliability_table(p, y, n_bins=2)
    assert list(t["n"]) == [2, 2] and list(t["actual"]) == [0.0, 1.0]
