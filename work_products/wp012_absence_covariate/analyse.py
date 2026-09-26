"""WP012 — paired comparison: L5_xgchannel vs L5 + absence covariate."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP007 = HERE.parent / "wp007_structural_fixes"
WP008 = HERE.parent / "wp008_market_benchmark"
sys.path.insert(0, str(WP007)); sys.path.insert(0, str(WP008))
from analysis import bootstrap_mean_ci, load_checkpoints, paired_diff, rps_hda, verdict, attach_meta  # noqa: E402
from market_analysis import PIN, join_market  # noqa: E402

PIN_OPEN = ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"]
KEY = ["window", "round", "home", "away"]


def main():
    df_cv = pickle.loads((WP007 / "cv_shared_data.pkl").read_bytes())["df_cv"]
    base, _ = load_checkpoints(WP007 / "checkpoints")
    new, wins = load_checkpoints(HERE / "checkpoints")
    m = attach_meta(pd.concat([base[base.arm == "L5_xgchannel"], new], ignore_index=True), df_cv)

    arms = ["L5_xgchannel", "L7_absence"]
    common = set.intersection(*[set(m[m.arm == a].window) for a in arms])
    m = m[m.window.isin(common)]
    piv = {a: m[m.arm == a].set_index(KEY).sort_index() for a in arms}
    idx = piv[arms[0]].index.intersection(piv[arms[1]].index)
    print(f"paired on {len(idx)} matches across {len(common)} windows\n")

    R = {a: rps_hda(piv[a].loc[idx][["p_home", "p_draw", "p_away"]].to_numpy(),
                    piv[a].loc[idx]["result"]) for a in arms}
    g = idx.get_level_values("window").to_numpy()

    print("pooled RPS:")
    for a in arms:
        mn, lo, hi = bootstrap_mean_ci(R[a], groups=g)
        print(f"  {a:14s} {mn:.5f}  [{lo:.5f}, {hi:.5f}]")
    d, lo, hi = paired_diff(R["L7_absence"], R["L5_xgchannel"], groups=g)
    print(f"\nPAIRED  L7_absence - L5_xgchannel: {d:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  -> {verdict(lo, hi)}")

    # what did the model actually learn for beta_absence?
    bp = [r for r in wins.to_dict("records") if r.get("param_beta_absence") is not None]
    if bp:
        b = np.array([r["param_beta_absence"] for r in bp])
        print(f"\nlearned beta_absence: median {np.median(b):+.4f}  (5-95%: {np.percentile(b,5):+.4f} .. "
              f"{np.percentile(b,95):+.4f}) across {len(b)} windows")
        print(f"  -> a 1-SD (0.107) absence swing changes that side's rate by "
              f"{(np.exp(np.median(b)*0.107)-1)*100:+.2f}%")

    # vs the fair (pre-close) market benchmark
    odds = pd.read_pickle(WP008 / "odds_raw.pkl")
    j, _, _ = join_market(m[m.arm == "L5_xgchannel"], odds,
                          books={"pin_close": PIN, "pin_open": PIN_OPEN})
    j = j[j.pin_open_h.notna()].set_index(KEY)
    ji = j.index.intersection(idx)
    r_open = rps_hda(j.loc[ji][["pin_open_h", "pin_open_d", "pin_open_a"]].to_numpy(), j.loc[ji]["result"])
    gg = ji.get_level_values("window").to_numpy()
    print(f"\ngap vs Pinnacle pre-close (n={len(ji)}):")
    for a in arms:
        ra = rps_hda(piv[a].loc[ji][["p_home", "p_draw", "p_away"]].to_numpy(), piv[a].loc[ji]["result"])
        dd, l2, h2 = bootstrap_mean_ci(ra - r_open, groups=gg)
        print(f"  {a:14s} {dd:+.5f}  [{l2:+.5f}, {h2:+.5f}]")

    print("\nsampler health:")
    print(f"  windows={wins.window.nunique()}  divergences={int(wins.n_diverging.sum())}  "
          f"max_rhat={wins.max_rhat.max():.4f}  mean_secs={wins.seconds.mean():.0f}")


if __name__ == "__main__":
    main()
