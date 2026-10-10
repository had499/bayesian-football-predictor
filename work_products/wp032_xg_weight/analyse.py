"""WP032 — where does the xG channel's weight curve peak?

A weight sweep is NOT the usual "does this arm beat the champion" question, so the
analysis is different in two ways:

1. MULTIPLICITY. Four arms tested against one baseline means four chances to clear
   zero. The project has been burned by this before (WP007: 3 of 26 arms exceeded
   P>0.80 where noise predicts 5.2). So the headline here is the SHAPE of the curve,
   not the best single point: a genuine optimum is a smooth curve with an interior
   peak, whereas noise gives a ragged ordering. W025 is the designed falsification --
   if both 0.25 and 1.0+ beat 0.5, the ordering is noise.

2. THE ARMS ARE NOT INDEPENDENT of WP007's result. Both grids share seasons, so this
   is a sharper measurement of the same quantity, not an independent confirmation.
   Leave-one-season-out therefore carries more weight than the headline CI.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import attach_meta, load_checkpoints, paired_diff, verdict, wide  # noqa: E402

BASE = "I_effort"
WEIGHTS = {"W025": 0.25, BASE: 0.50, "W075": 0.75, "W100": 1.00,
           "W175": 1.75, "W300": 3.00}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=str, default="")
    args = ap.parse_args()
    df = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                      .read_bytes())["df_cv"]
    a, _ = load_checkpoints(WP / "wp028_stakes/checkpoints")
    b, wb = load_checkpoints(HERE / "checkpoints")
    full = b.groupby("arm").window.nunique()
    present = (args.arms.split(",") if args.arms
               else sorted(full[full == full.max()].index, key=lambda x: WEIGHTS[x]))
    print(f"arms: {', '.join(f'{x}=w{WEIGHTS[x]} ({full[x]} windows)' for x in present)}")
    b = b[b.arm.isin(present)]
    m = attach_meta(pd.concat([a[a.arm == BASE], b], ignore_index=True), df)
    arms = [BASE] + present
    com = set.intersection(*[set(m[m.arm == x].window) for x in arms])
    m = m[m.window.isin(com)]
    R = wide(m[m.arm.isin(arms)], "rps", arms)
    g = R.index.get_level_values("window").to_numpy()
    seasons = m[m.arm == BASE].set_index(["window", "round", "home", "away"]).loc[R.index, "season"]
    print(f"paired on {len(R)} matches / {len(com)} windows\n")

    print("=== THE WEIGHT CURVE (vs the champion's 0.5) ===")
    order = sorted(arms, key=lambda x: WEIGHTS[x])
    curve = {}
    for x in order:
        if x == BASE:
            print(f"  weight {WEIGHTS[x]:4.2f}  {R[x].mean():.5f}   (champion, reference)")
            curve[WEIGHTS[x]] = 0.0
            continue
        d, lo, hi = paired_diff(R[x], R[BASE], groups=g)
        curve[WEIGHTS[x]] = d
        print(f"  weight {WEIGHTS[x]:4.2f}  {R[x].mean():.5f}   {d:+.5f} "
              f"[{lo:+.5f},{hi:+.5f}]  {verdict(lo,hi)}")
    print("\n  [ref] WP007 said w1.0 vs w0.5 = -0.00052 [-0.0019,+0.0009] on 30 windows")
    print("  [ref] whole xG channel -0.00258 | MDE +-0.00095 | gap to Pinnacle +0.00344")

    ks = sorted(curve)
    vs = [curve[k] for k in ks]
    best = ks[int(np.argmin(vs))]
    interior = best not in (ks[0], ks[-1])
    print(f"\n  best weight {best}  |  interior peak: {'YES' if interior else 'NO — at an edge'}")
    mono = all(vs[i] >= vs[i + 1] for i in range(len(vs) - 1)) or \
           all(vs[i] <= vs[i + 1] for i in range(len(vs) - 1))
    print(f"  monotone across all five points: {'YES' if mono else 'NO'}")
    if "W025" in present and curve.get(0.25, 1) < 0 and max(
            [curve[k] for k in ks if k > 0.5] or [1]) < 0:
        print("  *** FALSIFICATION TRIPPED: 0.25 AND the high weights all beat 0.5.")
        print("      That is not a curve with an optimum -- treat the ordering as noise.")

    print("\n=== leave-one-season-out (more informative than the CI here) ===")
    for x in order:
        if x == BASE:
            continue
        cells, neg = [], 0
        for s in sorted(seasons.unique()):
            k = (seasons != s).to_numpy()
            d, _, _ = paired_diff(R[x][k], R[BASE][k], groups=g[k])
            neg += d < 0
            cells.append(f"{d:+.5f}")
        print(f"  w{WEIGHTS[x]:<5.2f} BETTER in {neg}/{len(cells)} folds   {' '.join(cells)}")

    print("\n=== per-season ===")
    for x in order:
        if x == BASE:
            continue
        pos = 0
        for s in sorted(seasons.unique()):
            k = (seasons == s).to_numpy()
            d, _, _ = paired_diff(R[x][k], R[BASE][k], groups=g[k])
            pos += d < 0
        print(f"  w{WEIGHTS[x]:<5.2f} better in {pos}/{len(seasons.unique())} seasons")

    print("\n=== what the channel LEARNED (does a big weight distort it?) ===")
    for x in order:
        if x == BASE:
            continue
        w = wb[wb.arm == x]
        kap = w["param_xg_kappa"].dropna() if "param_xg_kappa" in w else pd.Series(dtype=float)
        shp = w["param_xg_shape"].dropna() if "param_xg_shape" in w else pd.Series(dtype=float)
        print(f"  w{WEIGHTS[x]:<5.2f} xg_kappa {kap.median():.3f}  xg_shape {shp.median():.2f}"
              f"  divergences {int(w.n_diverging.sum()):4d}  max rhat {w.max_rhat.max():.3f}")


if __name__ == "__main__":
    main()
