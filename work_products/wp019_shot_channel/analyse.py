"""WP019 — does observing shot COUNT alongside xG help?

The premise (measured on the shot cache): a team-season's second-half goals/match
regressed on first-half log-volume and log-quality separately gives +1.028 and
+0.554 (difference +0.474, CI [+0.132, +0.771]); total xG forces them equal, and
freeing them lifts R^2 by +0.0192 [+0.0017, +0.0503]. This asks whether that
translates into forecast accuracy.
"""
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
NEW = ["E_shots_added", "F_shots_only", "G_shots_heavy"]


def main():
    df_cv = pickle.loads((WP007 / "cv_shared_data.pkl").read_bytes())["df_cv"]
    frames = [load_checkpoints(d)[0] for d in (WP007 / "checkpoints", WP013 / "checkpoints", HERE / "checkpoints")]
    wins = load_checkpoints(HERE / "checkpoints")[1]
    m = attach_meta(pd.concat(frames, ignore_index=True), df_cv)

    arms = ["L0_baseline", BASE] + NEW
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
    comps = [
        ("shot channel ADDED to xG (E vs A)", "E_shots_added", BASE),
        ("shot channel INSTEAD of xG (F vs A)", "F_shots_only", BASE),
        ("is shot count worth anything at all (F vs L0)", "F_shots_only", "L0_baseline"),
        ("heavier shot weight (G vs E)", "G_shots_heavy", "E_shots_added"),
        ("heavier shot weight vs baseline (G vs A)", "G_shots_heavy", BASE),
        ("[reference] the xG channel itself (A vs L0)", BASE, "L0_baseline"),
    ]
    for lbl, a, b in comps:
        d, lo, hi = paired_diff(R[a], R[b], groups=g)
        print(f"  {lbl:48s} {d:+.5f}  [{lo:+.5f}, {hi:+.5f}]  {verdict(lo, hi)}")

    print("\n=== Minimum detectable effect (BACKLOG caution 3: power is set by windows) ===")
    for lbl, a, b in comps[:4]:
        d, lo, hi = paired_diff(R[a], R[b], groups=g)
        print(f"  {lbl:48s} half-width {(hi - lo) / 2:.5f} = {(hi - lo) / 2 / 0.0039 * 100:4.0f}% of the 0.0039 gap")

    print("\n=== Leave-one-season-out, best new arm vs the shipping config ===")
    best = min(NEW, key=lambda a: R[a].mean())
    print(f"  (best new arm by pooled RPS: {best})")
    d, lo, hi = paired_diff(R[best], R[BASE], groups=g)
    print(f"    all seasons  n={len(R):5d}  {d:+.5f} [{lo:+.5f},{hi:+.5f}]  {verdict(lo, hi)}")
    n_better = 0
    for s in sorted(seasons.unique()):
        k = (seasons == s).to_numpy()          # WP013 style: score each season ALONE,
        dd, l, h = paired_diff(R[best][k], R[BASE][k], groups=g[k])   # not leave-one-out
        n_better += dd < 0
        print(f"    only {s}    n={k.sum():5d}  {dd:+.5f} [{l:+.5f},{h:+.5f}]  {verdict(l, h)}")
    print(f"    -> better in {n_better}/{len(seasons.unique())} seasons "
          f"(WP013's bar for the xG channel was 6/6)")

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

    print("\n=== Learned channel scalars (the mechanism check) ===")
    # param_means is a DICT-valued column, not one column per scalar — expanding it is
    # the difference between a diagnosable arm and a blank table (BACKLOG caution 5).
    pm = pd.json_normalize(wins["param_means"].apply(lambda x: x or {}))
    pm.index = wins.index
    wins = pd.concat([wins.drop(columns=["param_means"]), pm], axis=1)
    cols = [c for c in ("shot_kappa", "shot_alpha", "xg_kappa", "xg_shape",
                        "sigma_att", "sigma_def", "rho_att", "rho_def") if c in wins.columns]
    if cols:
        print(wins.groupby("arm")[cols].median().round(4).to_string())
        print("\n  shot_kappa is shots per unit scoring rate; the data's ratio is "
              f"{12.77 / 1.36:.2f}. A learned value far from that means the channel is "
              "absorbing level rather than shape.")
    print(f"\nsampler: divergences={int(wins.n_diverging.sum())}  max_rhat={wins.max_rhat.max():.4f}")


if __name__ == "__main__":
    main()
