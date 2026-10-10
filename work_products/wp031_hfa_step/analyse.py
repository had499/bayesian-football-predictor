"""WP031 — home advantage: crowd step vs mis-specified prior."""
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
CROWDLESS_FROM, CROWDLESS_TO = pd.Timestamp("2020-06-17"), pd.Timestamp("2021-05-09")


def main():
    d = pickle.loads((HERE / "cv_shared_data_crowd.pkl").read_bytes())
    df = d["df_cv"]
    a, wa = load_checkpoints(WP / "wp028_stakes/checkpoints")
    b, wb = load_checkpoints(HERE / "checkpoints")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=str, default="",
                    help="restrict to these arms; default = every arm with all 115 windows")
    args = ap.parse_args()
    full = b.groupby("arm").window.nunique()
    present = (args.arms.split(",") if args.arms
               else sorted(full[full == full.max()].index))
    b = b[b.arm.isin(present)]
    print(f"arms: {', '.join(f'{x} ({full[x]} windows)' for x in present)}")
    m = attach_meta(pd.concat([a[a.arm == BASE], b], ignore_index=True), df)
    arms = [BASE] + present
    com = set.intersection(*[set(m[m.arm == x].window) for x in arms])
    m = m[m.window.isin(com)]
    R = wide(m[m.arm.isin(arms)], "rps", arms)
    g = R.index.get_level_values("window").to_numpy()
    meta = m[m.arm == BASE].set_index(["window", "round", "home", "away"]).loc[R.index]
    seasons = meta["season"]

    dates = (df[df.is_home == 1][["round", "team", "opp_team", "datetime"]]
             .rename(columns={"team": "home", "opp_team": "away"}))
    dk = dates.set_index(["round", "home", "away"]).datetime
    idx = pd.MultiIndex.from_arrays([R.index.get_level_values("round"),
                                     R.index.get_level_values("home"),
                                     R.index.get_level_values("away")])
    dt = pd.to_datetime(pd.Series(dk.reindex(idx).to_numpy()))
    crowdless = ((dt >= CROWDLESS_FROM) & (dt <= CROWDLESS_TO)).to_numpy()

    print(f"paired on {len(R)} matches / {len(com)} windows "
          f"({crowdless.sum()} crowdless)\n")
    print(f"  {BASE:12s} {R[BASE].mean():.5f}  (the champion)")
    for x in present:
        dd, lo, hi = paired_diff(R[x], R[BASE], groups=g)
        print(f"  {x:12s} {R[x].mean():.5f}   {dd:+.5f} [{lo:+.5f},{hi:+.5f}]  {verdict(lo,hi)}")
    print("\n  [ref] xG -0.00258 | deep -0.00083 | stakes -0.00030 | MDE +-0.00095")
    print("  [ref] gap to Pinnacle OPENING line +0.00344 | WP030 intactness +0.00090\n")

    print("=== 1. crowdless vs the rest — the forward-value question ===")
    print("    A_crowd SHOULD gain only on crowdless matches. If it gains elsewhere,")
    print("    the term is picking up something other than crowds.")
    for lab, k in (("crowdless (over, no forward value)", crowdless),
                   ("crowds present (all forward value)", ~crowdless)):
        print(f"  {lab}  n={k.sum()}")
        for x in present:
            dd, lo, hi = paired_diff(R[x][k], R[BASE][k], groups=g[k])
            print(f"      {x:12s} {dd:+.5f} [{lo:+.5f},{hi:+.5f}]  {verdict(lo,hi)}")

    print("\n=== 2. leave-one-season-out ===")
    for x in present:
        cells, neg = [], 0
        for s in sorted(seasons.unique()):
            k = (seasons != s).to_numpy()
            dd, _, _ = paired_diff(R[x][k], R[BASE][k], groups=g[k])
            neg += dd < 0
            cells.append(f"{dd:+.5f}")
        print(f"  {x:12s} BETTER in {neg}/{len(cells)} folds   {' '.join(cells)}")

    print("\n=== 3. per-season ===")
    for x in present:
        cells, pos = [], 0
        for s in sorted(seasons.unique()):
            k = (seasons == s).to_numpy()
            dd, _, _ = paired_diff(R[x][k], R[BASE][k], groups=g[k])
            pos += dd < 0
            cells.append(f"{s}:{dd:+.5f}")
        print(f"  {x:12s} better in {pos}/{len(seasons.unique())}  {' '.join(cells)}")

    print("\n=== 4. what the prior was doing to home_mu ===")
    v0 = wa[wa.arm == BASE].set_index("window")["param_home_mu"].dropna()
    print(f"  champion (prior sd 0.03)  median {v0.median():.4f}"
          f"   = {(v0.median()-0.13)/0.03:+.2f} prior SDs from the 0.13 centre")
    for x in present:
        v1 = wb[wb.arm == x].set_index("window")["param_home_mu"].dropna()
        extra = ""
        if "param_crowd_hfa" in wb.columns:
            c = wb[wb.arm == x]["param_crowd_hfa"].dropna()
            if len(c):
                extra = (f"   crowd_hfa median {c.median():+.4f}"
                         f" = {abs(c.median())/0.15*100:.0f}% of prior SD,"
                         f" sign-consistent {max((c>0).sum(),(c<0).sum())}/{len(c)}")
        print(f"  {x:12s} median {v1.median():.4f}{extra}")

    w = wb[wb.arm.isin(present)]
    print(f"\n  sampler: divergences {int(w.n_diverging.sum())}  max rhat {w.max_rhat.max():.3f}")


if __name__ == "__main__":
    main()
