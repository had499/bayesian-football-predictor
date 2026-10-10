"""WP027 — does fixture congestion help, fitted as a covariate?

HEADLINE IS THE RESTRICTED GRID. Domestic cups exist in openfootball only from
2018-19, and the Europa League only from 2020-21. So for Understat seasons 2016,
2017 and 2025 -- 1,140 of 4,560 matches, 25% of the grid -- the covariate is built
from league + Champions League alone, i.e. close to WP009's broken league-only
measure. Pooling a well-measured covariate with a badly-measured one attenuates the
coefficient toward zero and would misattribute a null. Seasons 2018-2024 are
reported as the headline; the full grid is secondary.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import attach_meta, load_checkpoints, paired_diff, verdict, wide  # noqa: E402

BASE = "H_deep_added"
NEW = ["G_rest", "H_load"]
COMPLETE = [str(y) for y in range(2018, 2025)]   # seasons with the full calendar


def main():
    df = pickle.loads((WP / "wp021_power_confirmation/cv_shared_data_big.pkl").read_bytes())["df_cv"]
    a, _ = load_checkpoints(WP / "wp023_nutpie_baseline/checkpoints")
    b, wb = load_checkpoints(HERE / "checkpoints")
    m = attach_meta(pd.concat([a[a.arm == BASE], b], ignore_index=True), df)
    arms = [BASE] + NEW
    com = set.intersection(*[set(m[m.arm == x].window) for x in arms])
    m = m[m.window.isin(com)]
    R = wide(m[m.arm.isin(arms)], "rps", arms)
    g = R.index.get_level_values("window").to_numpy()
    seasons = m[m.arm == BASE].set_index(["window", "round", "home", "away"]).loc[R.index, "season"]
    print(f"paired on {len(R)} matches / {len(com)} windows\n")

    for lbl, mask in (("HEADLINE — complete calendar (2018-2024)", seasons.isin(COMPLETE).to_numpy()),
                      ("secondary — full grid (25% badly measured)", np.ones(len(R), bool))):
        print(f"=== {lbl}  n={mask.sum()} ===")
        print(f"  {BASE:16s} {R[BASE][mask].mean():.5f}")
        for x in NEW:
            d, lo, hi = paired_diff(R[x][mask], R[BASE][mask], groups=g[mask])
            print(f"  {x:16s} {R[x][mask].mean():.5f}   {d:+.5f} [{lo:+.5f},{hi:+.5f}]  {verdict(lo,hi)}")
        print()

    print("=== fitted coefficients (prior SD 0.1) ===")
    for arm, key in (("G_rest", "param_beta_rest"), ("H_load", "param_beta_load")):
        w = wb[wb.arm == arm]
        if key not in w.columns:
            print(f"  {arm}: {key} not recorded"); continue
        v = w[key].dropna()
        print(f"  {arm:8s} median {v.median():+.4f} = {abs(v.median())/0.1*100:3.0f}% of one prior SD, "
              f"positive in {(v>0).sum()}/{len(v)}")
    print("\n  [contrast] the Elo anchor fitted at +0.169 = 169% of its prior SD, 115/115 positive")

    print("\n=== per-season, best arm vs champion (complete-calendar seasons) ===")
    best = min(NEW, key=lambda x: R[x][seasons.isin(COMPLETE).to_numpy()].mean())
    print(f"  (best: {best})")
    pos = 0
    for s in COMPLETE:
        k = (seasons == s).to_numpy()
        if k.sum() < 50:
            continue
        d, lo, hi = paired_diff(R[best][k], R[BASE][k], groups=g[k])
        pos += d < 0
        print(f"    {s}  n={k.sum():4d}  {d:+.5f}  {verdict(lo,hi)}")
    print(f"    -> better in {pos}/{len(COMPLETE)} complete seasons")
    w = wb[wb.arm.isin(NEW)]
    print(f"\nsampler: divergences {int(w.n_diverging.sum())}  max rhat {w.max_rhat.max():.3f}")


if __name__ == "__main__":
    main()
