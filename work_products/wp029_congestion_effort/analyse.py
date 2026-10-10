"""WP029 — does the MULTIPLICATIVE form rescue congestion?

Headline is restricted to 2018-2024, where openfootball carries both domestic cups
(from 2018-19) and the Europa League (from 2020-21). For Understat seasons 2016/2017/
2025 -- 25% of the grid -- rest is built from league + Champions League only, averaging
7.27 days against 6.38 for complete seasons. Pooling a well-measured covariate with a
badly-measured one attenuates the coefficient and would misattribute the null. WP027
restricted for the same reason; this analysis does it by default rather than by memory.
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

BASE = "I_effort"          # WP028's arm IS the current champion
NEW = ["K_rest_eff", "L_load_eff"]
COMPLETE = [str(y) for y in range(2018, 2025)]


def main():
    df = pickle.loads((HERE / "cv_shared_data_both.pkl").read_bytes())["df_cv"]
    a, _ = load_checkpoints(WP / "wp028_stakes/checkpoints")
    b, wb = load_checkpoints(HERE / "checkpoints")
    m = attach_meta(pd.concat([a[a.arm == BASE], b], ignore_index=True), df)
    arms = [BASE] + NEW
    com = set.intersection(*[set(m[m.arm == x].window) for x in arms])
    m = m[m.window.isin(com)]
    R = wide(m[m.arm.isin(arms)], "rps", arms)
    g = R.index.get_level_values("window").to_numpy()
    seasons = m[m.arm == BASE].set_index(["window", "round", "home", "away"]).loc[R.index, "season"]
    print(f"paired on {len(R)} matches / {len(com)} windows\n")

    for lbl, msk in (("HEADLINE — complete calendar 2018-2024", seasons.isin(COMPLETE).to_numpy()),
                     ("secondary — full grid (25% badly measured)", np.ones(len(R), bool))):
        print(f"=== {lbl}  n={msk.sum()} ===")
        print(f"  {BASE:12s} {R[BASE][msk].mean():.5f}")
        for x in NEW:
            d, lo, hi = paired_diff(R[x][msk], R[BASE][msk], groups=g[msk])
            print(f"  {x:12s} {R[x][msk].mean():.5f}   {d:+.5f} [{lo:+.5f},{hi:+.5f}]  {verdict(lo,hi)}")
        print()

    print("=== coefficients — does the model USE them? (prior SD 0.3) ===")
    for arm, key in (("K_rest_eff", "param_rest_effort"), ("L_load_eff", "param_load_effort")):
        w = wb[wb.arm == arm]
        if key not in w.columns:
            continue
        v = w[key].dropna()
        print(f"  {arm:12s} {key.replace('param_',''):13s} median {v.median():+.4f} "
              f"= {abs(v.median())/0.3*100:4.1f}% of prior SD, sign-consistent "
              f"{max((v>0).sum(),(v<0).sum())}/{len(v)}")
        # does it decay as training history accumulates?
        early, late = v.iloc[:30].median(), v.iloc[-30:].median()
        print(f"  {'':12s} {'':13s} first 30 windows {early:+.4f} -> last 30 {late:+.4f}")
    print("\n  [stakes_effort, same multiplier, fitted at 91% of its prior SD]")
    print("  [WP027's ADDITIVE congestion fitted at ~2%]")
    w = wb[wb.arm.isin(NEW)]
    print(f"\nsampler: divergences {int(w.n_diverging.sum())}  max rhat {w.max_rhat.max():.3f}")


if __name__ == "__main__":
    main()
