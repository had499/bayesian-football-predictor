"""WP023 — verify nutpie at full scale: does it change the ANSWER across 115 windows?

Three windows said no (max |dRPS| 0.00032). This checks all 115, arm by arm,
against the stored numpyro checkpoints. Two things must hold before nutpie is
adopted:

  1. Per-window RPS agreement well inside the +-0.00095 MDE.
  2. The PAIRED arm effect (deep vs no-deep) must be the same under both samplers.
     That is the one that matters -- a systematic sampler shift cancels only when
     both arms use the same backend, and (2) is the direct test of it.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP021 = HERE.parent / "wp021_power_confirmation"
sys.path.insert(0, str(HERE.parent / "wp007_structural_fixes"))
from analysis import attach_meta, load_checkpoints, paired_diff, verdict, wide  # noqa: E402

ARMS = ["S3_xgchannel_only", "H_deep_added"]


def main():
    df_cv = pickle.loads((WP021 / "cv_shared_data_big.pkl").read_bytes())["df_cv"]
    out = {}
    for label, d in (("numpyro", WP021 / "checkpoints"), ("nutpie", HERE / "checkpoints")):
        m, wins = load_checkpoints(d)
        out[label] = (attach_meta(m, df_cv), wins)

    common = set.intersection(*[set(out[l][0][out[l][0].arm == a].window)
                                for l in out for a in ARMS])
    print(f"comparing on {len(common)} windows common to both samplers\n")

    print("=== 1. Does each arm give the same RPS? ===")
    for a in ARMS:
        w = {}
        for l in out:
            mm = out[l][0]
            mm = mm[(mm.arm == a) & mm.window.isin(common)]
            w[l] = mm.set_index(["window", "round", "home", "away"])["rps"].sort_index()
        idx = w["numpyro"].index.intersection(w["nutpie"].index)
        d = (w["nutpie"].loc[idx] - w["numpyro"].loc[idx])
        per_win = d.groupby(d.index.get_level_values("window")).mean()
        print(f"  {a:20s} mean dRPS {d.mean():+.6f}   max|per-window| {per_win.abs().max():.5f} "
              f"({per_win.abs().max()/0.00095*100:3.0f}% of MDE)")

    print("\n=== 2. Is the ARM EFFECT the same? (the one that matters) ===")
    for l in out:
        mm = out[l][0]
        mm = mm[mm.window.isin(common)]
        R = wide(mm[mm.arm.isin(ARMS)], "rps", ARMS)
        g = R.index.get_level_values("window").to_numpy()
        dd, lo, hi = paired_diff(R["H_deep_added"], R["S3_xgchannel_only"], groups=g)
        print(f"  deep vs no-deep under {l:8s}: {dd:+.5f} [{lo:+.5f}, {hi:+.5f}]  {verdict(lo, hi)}")
    print("\n  (WP021 measured -0.00083 [-0.00177, +0.00013] under numpyro.")
    print("   If nutpie reproduces that, the sampler swap is safe.)")

    print("\n=== 3. Cost ===")
    for l in out:
        wins = out[l][1]
        wins = wins[wins.window.isin(common) & wins.arm.isin(ARMS)]
        print(f"  {l:8s} {wins.seconds.sum()/3600:5.2f} CPU-hours over {len(wins)} fits  "
              f"(divergences {int(wins.n_diverging.sum())}, max rhat {wins.max_rhat.max():.3f})")
    a = out["numpyro"][1]; b = out["nutpie"][1]
    a = a[a.window.isin(common) & a.arm.isin(ARMS)]; b = b[b.window.isin(common) & b.arm.isin(ARMS)]
    if len(b):
        print(f"\n  speedup: {a.seconds.sum()/b.seconds.sum():.2f}x")


if __name__ == "__main__":
    main()
