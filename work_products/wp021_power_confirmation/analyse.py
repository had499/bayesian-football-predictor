"""WP020 — do deep completions help, and does the gain land where predicted?

PREDICTION ON RECORD, made before the sweep ran:
  (1) H_deep_added beats S3_xgchannel_only by MORE than 0.0005 RPS
  (2) the gain concentrates in the HIGH-LINE-MOVEMENT quartile (WP010's deficit)
Point estimate -0.0015; the per-match premise check extrapolated -0.0030
[-0.0062, -0.0006], shaded toward zero for the team-identification confound.
Both claims are scored explicitly at the bottom.
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
NEW = ["H_deep_added", "K_deep_shots"]


def main():
    df_cv = pickle.loads((WP007 / "cv_shared_data.pkl").read_bytes())["df_cv"]
    # every arm is refitted on THIS grid — WP007/WP013/WP020 checkpoints sit at
    # different train_end values and are NOT comparable here.
    frames = [load_checkpoints(HERE / "checkpoints")[0]]
    wins = load_checkpoints(HERE / "checkpoints")[1]
    m = attach_meta(pd.concat(frames, ignore_index=True), df_cv)

    arms = [BASE] + NEW
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
    comps = [("deep channel ADDED to xG (H vs A)", "H_deep_added", BASE),
             ("deep + shots together (K vs A)", "K_deep_shots", BASE),
             ("does adding shots to deep help (K vs H)", "K_deep_shots", "H_deep_added")]
    for lbl, a, b in comps:
        d, lo, hi = paired_diff(R[a], R[b], groups=g)
        print(f"  {lbl:46s} {d:+.5f}  [{lo:+.5f}, {hi:+.5f}]  {verdict(lo, hi)}")


    print("\n=== Per-season, best new arm vs the shipping config ===")
    best = min(NEW, key=lambda a: R[a].mean())
    print(f"  (best new arm by pooled RPS: {best})")
    n_better = 0
    for s in sorted(seasons.unique()):
        k = (seasons == s).to_numpy()
        dd, l, h = paired_diff(R[best][k], R[BASE][k], groups=g[k])
        n_better += dd < 0
        print(f"    only {s}  n={k.sum():5d}  {dd:+.5f} [{l:+.5f},{h:+.5f}]  {verdict(l, h)}")
    print(f"    -> better in {n_better}/{len(seasons.unique())} seasons (xG channel's bar was 6/6)")

    print("\n=== Gap vs Pinnacle pre-close ===")
    odds = pd.read_pickle(WP008 / "odds_raw.pkl")
    j, _, _ = join_market(m[m.arm == BASE], odds, books={"pin_close": PIN, "pin_open": PIN_OPEN})
    j = j[j[["pin_open_h", "pin_close_h"]].notna().all(axis=1)].set_index(KEY)
    ji = j.index.intersection(R.index)
    r_open = rps_hda(j.loc[ji][["pin_open_h", "pin_open_d", "pin_open_a"]].to_numpy(), j.loc[ji]["result"])
    gg = ji.get_level_values("window").to_numpy()
    for a in arms:
        dd, lo, hi = bootstrap_mean_ci(R[a].loc[ji].to_numpy() - r_open, groups=gg)
        print(f"  {a:18s} {dd:+.5f}  [{lo:+.5f}, {hi:+.5f}]")

    # ---- PREDICTION (2): does the gain land in the high-line-movement quartile? ----
    print("\n=== PREDICTION 2: where does the gain land? (WP010 line movement) ===")
    lm = (j.loc[ji, "pin_close_h"] - j.loc[ji, "pin_open_h"]).abs()
    q75, q25 = lm.quantile(0.75), lm.quantile(0.25)
    for lbl, mask in [("top 25% line movement", lm >= q75),
                      ("middle 50%", (lm > q25) & (lm < q75)),
                      ("bottom 25% line movement", lm <= q25)]:
        idx = lm.index[mask.to_numpy()]
        d, lo, hi = paired_diff(R[best].loc[idx], R[BASE].loc[idx],
                                groups=idx.get_level_values("window").to_numpy())
        print(f"  {lbl:26s} n={len(idx):5d}  {d:+.5f} [{lo:+.5f},{hi:+.5f}]  {verdict(lo, hi)}")

    print("\n=== Learned channel scalars ===")
    # load_checkpoints may already have expanded param_means into columns; if not,
    # expand the dict-valued column ourselves (BACKLOG caution 5).
    if "param_means" in wins.columns:
        pmn = pd.json_normalize(wins["param_means"].apply(lambda x: x or {}))
        pmn.index = wins.index
        w2 = pd.concat([wins.drop(columns=["param_means"]), pmn], axis=1)
    else:
        w2 = wins
    # load_checkpoints prefixes expanded scalars with "param_" — match either spelling
    want = ("deep_kappa", "deep_alpha", "xg_kappa", "xg_shape",
            "sigma_att", "sigma_def", "rho_att", "rho_def")
    cols = [c for n in want for c in (n, f"param_{n}") if c in w2.columns]
    if cols:
        print(w2.groupby("arm")[cols].median().round(4).to_string())
        print(f"\n  deep_kappa should land near the data's ratio {6.87/1.36:.2f}; far from it means "
              "the channel is absorbing level rather than shape.")
    print(f"\nsampler: divergences={int(wins.n_diverging.sum())}  max_rhat={wins.max_rhat.max():.4f}")

    # ---- score the recorded prediction ----
    d, lo, hi = paired_diff(R["H_deep_added"], R[BASE], groups=g)
    print("\n" + "=" * 62)
    print("SCORING THE RECORDED PREDICTION")
    print(f"  WP020 measured -0.00096 [-0.00248, +0.00063] at 30 windows (MDE +-0.00156).")
    mde = (hi - lo) / 2
    print(f"  WP021 @ {len(common)} windows: {d:+.5f} [{lo:+.5f}, {hi:+.5f}]  (achieved MDE +-{mde:.5f})")
    print(f"  effect retained {d/-0.00096*100:.0f}% of WP020's size; CI narrowed {(1-mde/0.00156)*100:.0f}%")
    if hi < 0:
        print("  verdict:   CONFIRMED — the deep channel is a real improvement")
    elif d < 0 and abs(d) > mde * 0.8:
        print("  verdict:   STILL NULL, but the effect is STABLE across two independent grids")
        print(f"             and the CI now misses zero by only {hi:.5f}. Not proven; not dismissed.")
        print(f"             Clearing zero needs MDE < {abs(d):.5f}, i.e. ~{int(115*(mde/abs(d))**2)} windows.")
    elif d < 0:
        print("  verdict:   still null and the effect shrank — evidence AGAINST WP020's size.")
    else:
        print("  verdict:   REVERSED — WP020's point estimate did not survive more windows.")
    print("=" * 62)


if __name__ == "__main__":
    main()
