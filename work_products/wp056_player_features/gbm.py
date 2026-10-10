"""WP056b -- does the NEW feature information help the GBM, on the horizon-1 grid?

ABLATION, not a single number. Four feature sets, each tuned on windows 70-200 and
LOCKED before windows 201-343 are touched:

  OLD    WP053's 253 team-match aggregates (exact parity, asserted in build.py)
  NEW    the 228 never-used features alone -- does this information exist at all?
  BOTH   481 features
  +MKT   BOTH plus the opening O/U 2.5 and Asian-handicap lines. CIRCULAR: it uses the
         market to beat the market, so it is a CEILING, never a shippable arm.

The grid is wp055's 343-window TEST_LEN=1 (deployment-accurate). The GBM has only ever
been measured on TEST_LEN=3 before, where it gave -0.00062 at beta=0.5.
"""
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
sys.path.insert(0, str(WP / "wp053_features"))
from analysis import paired_diff                                  # noqa: E402
from gbm import walk_forward, score                               # noqa: E402

S = pickle.load((HERE / "features_v2.pkl").open("rb"))
D, OLD, NEW, MKT = S["D"], S["old"], S["new"], S["mkt"]
TUNE_LO, TUNE_HI = 70, 200          # 201..343 is the held-out evaluation range
GRID = [
    dict(max_depth=2, max_iter=150, learning_rate=0.03, l2_regularization=10.0),
    dict(max_depth=2, max_iter=400, learning_rate=0.02, l2_regularization=10.0),
    dict(max_depth=3, max_iter=250, learning_rate=0.03, l2_regularization=5.0),
]
ARMS = [("OLD", OLD), ("NEW", NEW), ("BOTH", OLD + NEW), ("+MKT", OLD + NEW + MKT)]


def main():
    d = D.reset_index(drop=True)
    wins = np.sort(d.window.unique())
    tune_w = [w for w in wins if TUNE_LO <= w <= TUNE_HI][::3]   # strided: config choice only
    eval_w = [w for w in wins if w > TUNE_HI]
    print(f"grid {len(wins)} windows | tune {TUNE_LO}-{TUNE_HI} (strided, {len(tune_w)}) "
          f"| EVAL {eval_w[0]}-{eval_w[-1]} ({len(eval_w)})")
    print("Poisson loss, log(lambda) offset, walk-forward (window w sees only w' < w)\n")

    results = {}
    for name, FE in ARMS:
        X = d[FE].to_numpy(float)
        best, best_v = None, np.inf
        for cfg in GRID:
            mc = walk_forward(cfg, tune_w, X, d)
            b, a, g, n = score(mc, d)
            v = (a - b).mean()
            if v < best_v: best_v, best = v, cfg
        print(f"[{name:4s}] {len(FE):3d} feats | locked depth={best['max_depth']} "
              f"iter={best['max_iter']} lr={best['learning_rate']} (tune dRPS {best_v:+.5f})")
        mc = walk_forward(best, eval_w, X, d)
        base, adj, g, n = score(mc, d)
        row = {}
        for beta in (0.25, 0.5, 1.0):
            _, ab, gb, _ = score(mc, d, beta=beta)
            row[beta] = paired_diff(pd.Series(ab), pd.Series(base), groups=gb) + (ab.mean(),)
        results[name] = (n, base.mean(), row)
        print(f"        n={n}  baseline RPS {base.mean():.5f}")

    print(f"\n=== EVALUATION, windows {eval_w[0]}-{eval_w[-1]}, configs locked ===")
    print(f"  {'arm':5s} {'beta':>5s} {'RPS':>9s} {'dRPS':>10s} {'CI':>22s}")
    for name, (n, b0, row) in results.items():
        for beta, (dd, lo, hi, rps) in row.items():
            tag = "  <-- CI excludes 0" if hi < 0 else ("  worse" if lo > 0 else "")
            print(f"  {name:5s} {beta:5.2f} {rps:9.5f} {dd:+10.5f} [{lo:+.5f},{hi:+.5f}]{tag}")
        print()
    # does NEW add anything OVER old? compare BOTH against OLD at the best beta
    print("  BOTH vs OLD is the question that matters: if BOTH ~= OLD the new families")
    print("  are redundant with the aggregates; if BOTH < OLD they carry new information.")
    pickle.dump(results, (HERE / "gbm_results.pkl").open("wb"))


if __name__ == "__main__":
    main()
