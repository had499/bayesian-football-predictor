"""WP036 — can shrinking the lambda spread exploit the home over-dispersion?

The dispersion audit found the champion's home lambdas spread too WIDE (ratio 0.9132
[0.8624,0.9648], CI excluding 1.0; away fine at 0.9913), and the strength-quintile
tilts are non-monotone -- over-rating home teams at BOTH extremes -- which is the
signature of too much spread rather than a strength interaction.

This tests the implied fix POST HOC on the 3,780 existing predictions, so it costs
minutes instead of a 3-hour sweep. If a post-hoc shrink cannot find a gain, a fitted
in-model version will not either, and the direction is closed.

    log lambda  ->  mean + k * (log lambda - mean),  k <= 1

k is applied per side, with the mean taken WITHIN each window so it cannot peek across
the walk-forward boundary. k=1 reproduces the champion exactly (the control).

PRIOR EXPECTATION, and the reason to be pessimistic: WP009 tested post-hoc SHARPENING of
the outcome probabilities and found it nets to zero -- it helps non-upsets and hurts
upsets, and the two cancel. Shrinking is the same axis in the opposite direction, so
WP009 predicts this nets to zero too. It is not identical (shrinking lambdas changes the
whole scoreline distribution, including total goals, not just a 3-way temperature), which
is the only reason it is worth the few minutes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import load_checkpoints, paired_diff  # noqa: E402

MAXG = 12


def outcome_probs(lh, la):
    ph = poisson.pmf(np.arange(MAXG + 1)[None, :], lh[:, None])
    pa = poisson.pmf(np.arange(MAXG + 1)[None, :], la[:, None])
    grid = ph[:, :, None] * pa[:, None, :]
    iu = np.triu_indices(MAXG + 1, 1)
    il = np.tril_indices(MAXG + 1, -1)
    p_away = grid[:, iu[0], iu[1]].sum(1)
    p_home = grid[:, il[0], il[1]].sum(1)
    p_draw = np.einsum("ikk->i", grid)
    p = np.stack([p_home, p_draw, p_away], 1)
    return p / p.sum(1, keepdims=True)


def rps(p, outcome):
    y = np.zeros_like(p)
    y[np.arange(len(p)), outcome] = 1.0
    return ((p.cumsum(1) - y.cumsum(1))[:, :2] ** 2).sum(1) / 2.0


def main():
    m, _ = load_checkpoints(WP / "wp032_xg_weight/checkpoints")
    m = m[m.arm == "W075"].copy()
    gh, ga = m.goals_home.to_numpy(int), m.goals_away.to_numpy(int)
    outcome = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))
    win = m.window.to_numpy()

    base = rps(outcome_probs(m.lambda_home.to_numpy(), m.lambda_away.to_numpy()), outcome)
    print(f"reconstructed champion RPS {base.mean():.5f}  "
          f"(checkpoint says {m.rps.mean():.5f}, diff {base.mean()-m.rps.mean():+.5f})")
    print("  a small diff is expected: this grid is plug-in Poisson on posterior-MEAN")
    print("  lambdas, while the harness averages over posterior draws.\n")

    lh0, la0 = m.lambda_home.to_numpy(), m.lambda_away.to_numpy()
    llh, lla = np.log(lh0), np.log(la0)
    # window-local means, so nothing peeks across the walk-forward boundary
    mh = pd.Series(llh).groupby(win).transform("mean").to_numpy()
    ma = pd.Series(lla).groupby(win).transform("mean").to_numpy()

    print("=== post-hoc shrink of the lambda spread (k<1 shrinks, k=1 = champion) ===")
    print(f"  {'k_home':>7s} {'k_away':>7s} {'RPS':>9s} {'dRPS':>10s}  {'CI':>22s}")
    grid = [(1.0, 1.0), (0.95, 1.0), (0.90, 1.0), (0.85, 1.0), (0.70, 1.0),
            (0.95, 0.95), (0.90, 0.90), (1.05, 1.0)]
    for kh, ka in grid:
        lh = np.exp(mh + kh * (llh - mh))
        la = np.exp(ma + ka * (lla - ma))
        r = rps(outcome_probs(lh, la), outcome)
        d, lo, hi = paired_diff(pd.Series(r), pd.Series(base), groups=win)
        tag = "  <- champion (control)" if (kh, ka) == (1.0, 1.0) else ""
        print(f"  {kh:7.2f} {ka:7.2f} {r.mean():9.5f} {d:+10.5f}  "
              f"[{lo:+.5f},{hi:+.5f}]{tag}")

    print("\n  the dispersion audit implied k_home ~ 0.70. If that row is not clearly")
    print("  negative, the over-dispersion is NOT exploitable through the lambda spread")
    print("  and this direction is closed -- consistent with WP009's sharpening null.")


if __name__ == "__main__":
    main()
