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

    arms = ["L0_baseline", BASE, "E_hfa_only", "F_level_only", "G_both"]
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
        ("time-varying home advantage (E vs A)", "E_hfa_only", BASE),
        ("hard centering + drifting level (F vs A)", "F_level_only", BASE),
        ("both together (G vs A)", "G_both", BASE),
        ("does HFA add on top of the level fix (G vs F)", "G_both", "F_level_only"),
        ("[reference] xG channel itself (A vs L0)", BASE, "L0_baseline"),
    ]:
        d, lo, hi = paired_diff(R[a], R[b], groups=g)
        print(f"  {lbl:52s} {d:+.5f}  [{lo:+.5f}, {hi:+.5f}]  {verdict(lo, hi)}")

    print("\n=== Leave-one-season-out, best new arm vs the shipping config ===")
    best = min(["E_hfa_only", "F_level_only", "G_both"], key=lambda a: R[a].mean())
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

    ks = [c for c in wins.columns if any(k in c for k in ("sigma_hfa", "sigma_level"))]
    if ks:
        print("\nlearned walk sigmas (per-arm medians) — how fast the model lets them drift:")
        print(wins.groupby("arm")[ks].median().round(4).to_string())

    # THE falsification test: does fixing HFA shrink WP009's away-favourite asymmetry?
    print("\n=== WP009 away-favourite asymmetry — does fixing HFA shrink it? ===")
    odds2 = pd.read_pickle(WP008 / "odds_raw.pkl")
    jj, _, _ = join_market(m[m.arm == "L0_baseline"], odds2, books={"pin_close": PIN, "pin_open": PIN_OPEN})
    jj = jj[jj.pin_open_h.notna()].set_index(KEY)
    k2 = jj.index.intersection(R.index)
    fav_home = (jj.loc[k2].pin_close_h >= jj.loc[k2].pin_close_a).to_numpy()
    r_mkt = rps_hda(jj.loc[k2][["pin_open_h", "pin_open_d", "pin_open_a"]].to_numpy(), jj.loc[k2]["result"])
    gg2 = k2.get_level_values("window").to_numpy()
    for a in [BASE, "E_hfa_only", "G_both"]:
        ra = R[a].loc[k2].to_numpy() - r_mkt
        dh, lh_, hh = bootstrap_mean_ci(ra[fav_home], groups=gg2[fav_home])
        da, la_, ha = bootstrap_mean_ci(ra[~fav_home], groups=gg2[~fav_home])
        print(f"  {a:18s} home-fav {dh:+.5f} [{lh_:+.5f},{hh:+.5f}] | "
              f"away-fav {da:+.5f} [{la_:+.5f},{ha:+.5f}] | asymmetry {da-dh:+.5f}")
    print(f"\nsampler: divergences={int(wins.n_diverging.sum())}  max_rhat={wins.max_rhat.max():.4f}")


if __name__ == "__main__":
    main()
