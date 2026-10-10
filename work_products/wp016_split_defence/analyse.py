"""WP014 — does npxG sharpen the xG channel, and does per-season kappa help?"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP007 = HERE.parent / "wp007_structural_fixes"
WP008 = HERE.parent / "wp008_market_benchmark"
WP013 = HERE.parent / "wp013_xg_channel_robustness"
for p in (WP007, WP008):
    sys.path.insert(0, str(p))
from analysis import (attach_meta, bootstrap_mean_ci, load_checkpoints,  # noqa: E402
                      paired_diff, rps_hda, verdict, wide)
from market_analysis import PIN, join_market  # noqa: E402

PIN_OPEN = ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"]
KEY = ["window", "round", "home", "away"]
BASE = "S3_xgchannel_only"


def main():
    df_cv = pickle.loads((WP007 / "cv_shared_data.pkl").read_bytes())["df_cv"]
    frames = [load_checkpoints(d)[0] for d in (WP007 / "checkpoints", WP013 / "checkpoints", HERE / "checkpoints")]
    wins = load_checkpoints(HERE / "checkpoints")[1]
    m = attach_meta(pd.concat(frames, ignore_index=True), df_cv)

    arms = ["L0_baseline", BASE, "H_split", "H_split_loose"]
    common = set.intersection(*[set(m[m.arm == a].window) for a in arms])
    m = m[m.window.isin(common)]
    R = wide(m[m.arm.isin(arms)], "rps", arms)
    g = R.index.get_level_values("window")
    seasons = m[m.arm == arms[0]].set_index(KEY).loc[R.index, "season"]
    print(f"paired on {len(R)} matches / {len(common)} windows\n")

    print("pooled RPS:")
    for a in arms:
        mn, lo, hi = bootstrap_mean_ci(R[a].to_numpy(), groups=g)
        print(f"  {a:18s} {mn:.5f}  [{lo:.5f}, {hi:.5f}]")

    print("\n=== The comparisons that isolate each change ===")
    for lbl, a, b in [
        ("split defence, tight stopping sigma (H vs A)", "H_split", BASE),
        ("split defence, loose stopping sigma (H_loose vs A)", "H_split_loose", BASE),
        ("does the sigma prior matter (H_loose vs H)", "H_split_loose", "H_split"),
        ("[reference] xG channel itself (A vs L0)", BASE, "L0_baseline"),
    ]:
        d, lo, hi = paired_diff(R[a], R[b], groups=g)
        print(f"  {lbl:52s} {d:+.5f}  [{lo:+.5f}, {hi:+.5f}]  {verdict(lo, hi)}")

    print("\n=== Leave-one-season-out, best new arm vs the shipping config ===")
    best = min(["H_split", "H_split_loose"], key=lambda a: R[a].mean())
    print(f"  (best new arm by pooled RPS: {best})")
    d, lo, hi = paired_diff(R[best], R[BASE], groups=g)
    print(f"    all seasons  n={len(R):5d}  {d:+.5f} [{lo:+.5f},{hi:+.5f}]  {verdict(lo,hi)}")
    for s in sorted(seasons.unique()):
        k = (seasons != s).to_numpy()
        dd, l, h = paired_diff(R[best][k], R[BASE][k], groups=g[k])
        print(f"    drop {s}    n={k.sum():5d}  {dd:+.5f} [{l:+.5f},{h:+.5f}]  {verdict(l,h)}")

    print("\n=== Gap vs Pinnacle pre-close ===")
    odds = pd.read_pickle(WP008 / "odds_raw.pkl")
    j, _, _ = join_market(m[m.arm == "L0_baseline"], odds, books={"pin_close": PIN, "pin_open": PIN_OPEN})
    j = j[j.pin_open_h.notna()].set_index(KEY)
    ji = j.index.intersection(R.index)
    r_open = rps_hda(j.loc[ji][["pin_open_h", "pin_open_d", "pin_open_a"]].to_numpy(), j.loc[ji]["result"])
    gg = ji.get_level_values("window").to_numpy()
    for a in arms:
        dd, lo, hi = bootstrap_mean_ci(R[a].loc[ji].to_numpy() - r_open, groups=gg)
        print(f"  {a:18s} {dd:+.5f}  [{lo:+.5f}, {hi:+.5f}]")

    ks = [c for c in wins.columns if any(k in c for k in ("sigma_stop", "sigma_def", "sigma_att"))]
    if ks:
        print("\nlearned sigmas (per-arm medians) — did the model give stopping its own rate?")
        print(wins.groupby("arm")[ks].median().round(4).to_string())

    # Did the split actually separate the two skills, or collapse them?
    print("\n=== Did the model learn a DISTINCT shot-stopping latent? ===")
    for arm in ["H_split", "H_split_loose"]:
        sub = wins[wins.arm == arm]
        cols = [c for c in sub.columns if c in ("param_sigma_def", "param_sigma_stop")]
        if len(cols) == 2:
            sd, ss = sub["param_sigma_def"].median(), sub["param_sigma_stop"].median()
            print(f"  {arm:16s} sigma_suppress={sd:.4f}  sigma_stop={ss:.4f}  ratio={ss/sd:.2f}")
    print("  (data says suppression persists r=+0.65 vs stopping r=+0.22, so a well-behaved")
    print("   fit should want sigma_stop SMALLER relative to its own signal, not larger)")

    print(f"\nsampler: divergences={int(wins.n_diverging.sum())}  max_rhat={wins.max_rhat.max():.4f}")


if __name__ == "__main__":
    main()
