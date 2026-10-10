"""WP013 — is the xG channel robust, and does it work on its own?"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP007 = HERE.parent / "wp007_structural_fixes"
WP008 = HERE.parent / "wp008_market_benchmark"
sys.path.insert(0, str(WP007)); sys.path.insert(0, str(WP008))
from analysis import (attach_meta, bootstrap_mean_ci, load_checkpoints,  # noqa: E402
                      paired_diff, rps_hda, verdict, wide)
from market_analysis import PIN, join_market  # noqa: E402

PIN_OPEN = ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"]
KEY = ["window", "round", "home", "away"]


def main():
    df_cv = pickle.loads((WP007 / "cv_shared_data.pkl").read_bytes())["df_cv"]
    base, _ = load_checkpoints(WP007 / "checkpoints")
    iso, wins = load_checkpoints(HERE / "checkpoints")
    m = attach_meta(pd.concat([base, iso], ignore_index=True), df_cv)

    arms = ["L0_baseline", "L5_xgchannel", "S3_xgchannel_only"]
    common = set.intersection(*[set(m[m.arm == a].window) for a in arms])
    m = m[m.window.isin(common)]
    R = wide(m[m.arm.isin(arms)], "rps", arms)
    g = R.index.get_level_values("window")
    seasons = m[m.arm == arms[0]].set_index(KEY).loc[R.index, "season"]
    print(f"paired on {len(R)} matches / {len(common)} windows\n")

    print("pooled RPS:")
    for a in arms:
        mn, lo, hi = bootstrap_mean_ci(R[a].to_numpy(), groups=g)
        print(f"  {a:20s} {mn:.5f}  [{lo:.5f}, {hi:.5f}]")

    print("\n=== 1. Does the xG channel work ALONE (vs the full L5 bundle)? ===")
    for lbl, a, b in [("xG-channel-only vs L0 (isolated effect)", "S3_xgchannel_only", "L0_baseline"),
                      ("full L5 bundle vs L0 (cumulative)", "L5_xgchannel", "L0_baseline"),
                      ("L5 vs xG-channel-only (what the other 4 rungs add)",
                       "L5_xgchannel", "S3_xgchannel_only")]:
        d, lo, hi = paired_diff(R[a], R[b], groups=g)
        print(f"  {lbl:52s} {d:+.5f}  [{lo:+.5f}, {hi:+.5f}]  {verdict(lo, hi)}")

    print("\n=== 2. Leave-one-season-out robustness ===")
    for a, b, nm in [("L5_xgchannel", "L0_baseline", "L5 vs L0"),
                     ("S3_xgchannel_only", "L0_baseline", "xG-only vs L0")]:
        print(f"  {nm}:")
        d, lo, hi = paired_diff(R[a], R[b], groups=g)
        print(f"    all seasons   n={len(R):5d}  {d:+.5f} [{lo:+.5f},{hi:+.5f}]  {verdict(lo,hi)}")
        for s in sorted(seasons.unique()):
            k = (seasons != s).to_numpy()
            dd, l, h = paired_diff(R[a][k], R[b][k], groups=g[k])
            print(f"    drop {s}     n={k.sum():5d}  {dd:+.5f} [{l:+.5f},{h:+.5f}]  {verdict(l,h)}")

    print("\n=== 3. Gap vs Pinnacle pre-close ===")
    odds = pd.read_pickle(WP008 / "odds_raw.pkl")
    j, _, _ = join_market(m[m.arm == "L0_baseline"], odds,
                          books={"pin_close": PIN, "pin_open": PIN_OPEN})
    j = j[j.pin_open_h.notna()].set_index(KEY)
    ji = j.index.intersection(R.index)
    r_open = rps_hda(j.loc[ji][["pin_open_h", "pin_open_d", "pin_open_a"]].to_numpy(), j.loc[ji]["result"])
    gg = ji.get_level_values("window").to_numpy()
    for a in arms:
        dd, lo, hi = bootstrap_mean_ci(R[a].loc[ji].to_numpy() - r_open, groups=gg)
        print(f"  {a:20s} {dd:+.5f}  [{lo:+.5f}, {hi:+.5f}]")

    print(f"\nsampler health (isolation arm): windows={wins.window.nunique()}  "
          f"divergences={int(wins.n_diverging.sum())}  max_rhat={wins.max_rhat.max():.4f}")


if __name__ == "__main__":
    main()
