"""WP030 — did the training-row intactness correction help?

Baseline: WP028's I_effort, which IS the champion (L0 + xG + deep + stakes_effort).
Verified bit-identical data apart from the intact/rest/load columns.

THE MECHANISTIC SLICE, which is the real test. At prediction every test row is assumed
to be a normal XI, so the term contributes NOTHING to the test match directly. Any gain
must arrive through CLEANER LATENTS. So the gain should concentrate on matches whose
teams came through a DEPLETED recent stretch -- their attack/defence latents are the
ones the correction rescued. A gain spread evenly over matches with and without a
depleted run-up would mean something else is doing the work, and I should say so.
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

BASE = "I_effort"
NEW = ["M_intact", "N_intact_centred"]
PRIOR_SD = 0.5          # ModelConfig.intactness_sd


def _wide(df):
    """df_cv is LONG (one row per team per match); collapse to match rows."""
    d = df[df.is_home == 1][["datetime", "round", "team", "opp_team",
                             "intact_home", "intact_away"]].copy()
    return d.rename(columns={"team": "home", "opp_team": "away"})


def recent_depletion(df, lookback=6):
    """For each match, the worst intactness either side carried over its own previous
    `lookback` matches -- i.e. how corrupted its latents could have been. Prior-only."""
    long = df[["datetime", "team", "intact_home", "intact_away", "is_home"]].copy()
    long["intact"] = long.intact_home.where(long.is_home == 1, long.intact_away)
    long = long.sort_values("datetime")
    long["run"] = (long.groupby("team").intact
                   .transform(lambda s: s.shift(1).rolling(lookback, min_periods=2).min()))
    key = long.set_index(["team", "datetime"]).run
    w = _wide(df)
    h = key.reindex(pd.MultiIndex.from_arrays([w.home, w.datetime])).to_numpy()
    a = key.reindex(pd.MultiIndex.from_arrays([w.away, w.datetime])).to_numpy()
    w["run_min"] = np.nanmin(np.vstack([h, a]), axis=0)
    w["now_min"] = np.minimum(w.intact_home, w.intact_away)
    return w[["round", "home", "away", "run_min", "now_min"]]


def main():
    df = pickle.loads((HERE / "cv_shared_data_intact.pkl").read_bytes())["df_cv"]
    a, wa = load_checkpoints(WP / "wp028_stakes/checkpoints")
    b, wb = load_checkpoints(HERE / "checkpoints")
    m = attach_meta(pd.concat([a[a.arm == BASE], b], ignore_index=True), df)
    arms = [BASE] + NEW
    com = set.intersection(*[set(m[m.arm == x].window) for x in arms])
    m = m[m.window.isin(com)]
    R = wide(m[m.arm.isin(arms)], "rps", arms)
    g = R.index.get_level_values("window").to_numpy()
    meta = m[m.arm == BASE].set_index(["window", "round", "home", "away"]).loc[R.index]
    seasons = meta["season"]
    print(f"paired on {len(R)} matches / {len(com)} windows\n")
    print(f"  {BASE:12s} {R[BASE].mean():.5f}  (the champion)")
    for x in NEW:
        d, lo, hi = paired_diff(R[x], R[BASE], groups=g)
        print(f"  {x:12s} {R[x].mean():.5f}   {d:+.5f} [{lo:+.5f},{hi:+.5f}]  {verdict(lo,hi)}")
    print("\n  [ref] xG -0.00258 | deep -0.00083 | stakes_effort -0.00030 | MDE +-0.00095")
    print("  [ref] remaining gap to Pinnacle's OPENING line: +0.00344\n")

    print("=== 1. learned coefficients — THE diagnostic ===")
    print("    four additive covariates before this fitted at ~2% of prior SD.")
    for arm, key in [(a, k) for a in NEW
                     for k in ("param_intact_att", "param_intact_def")]:
        w = wb[wb.arm == arm]
        if key not in w.columns:
            print(f"  {arm} {key}: NOT RECORDED"); continue
        v = w[key].dropna()
        print(f"  {arm:18s} {key[6:]:11s} median {v.median():+.4f} = {abs(v.median())/PRIOR_SD*100:3.0f}%"
              f" of prior SD, sign-consistent {max((v>0).sum(),(v<0).sum())}/{len(v)}")

    print("\n=== 2. slices ===")
    dep = recent_depletion(df)
    idx = pd.MultiIndex.from_arrays([R.index.get_level_values("round"),
                                     R.index.get_level_values("home"),
                                     R.index.get_level_values("away")])
    dk = dep.set_index(["round", "home", "away"])
    run = dk.run_min.reindex(idx).to_numpy()
    now = dk.now_min.reindex(idx).to_numpy()

    def slc(lab, k):
        k = np.nan_to_num(k, nan=False).astype(bool)
        if k.sum() < 100:
            print(f"  {lab:34s} n={k.sum():5d}  too few"); return
        out = []
        for x in NEW:
            d, lo, hi = paired_diff(R[x][k], R[BASE][k], groups=g[k])
            out.append(f"{x[0]}: {d:+.5f} [{lo:+.5f},{hi:+.5f}] {verdict(lo,hi):<20s}")
        print(f"  {lab:34s} n={k.sum():5d}  " + "  ".join(out))

    print("  (a) MECHANISM: did the teams come through a depleted stretch? The gain,")
    print("      if the story is right, should concentrate HERE.")
    slc("depleted run-up    (<0.85)", run < 0.85)
    slc("mild               (0.85-0.95)", (run >= 0.85) & (run < 0.95))
    slc("intact run-up      (>=0.95)", run >= 0.95)
    print("\n  (b) THE SUSPECT: is the TEST match itself depleted? Prediction assumes")
    print("      intact=1, so if clean latents mean we over-rate teams that turn out")
    print("      to be weakened, the damage lands here.")
    slc("test match depleted  (<0.85)", now < 0.85)
    slc("test match mild      (0.85-0.95)", (now >= 0.85) & (now < 0.95))
    slc("test match full      (>=0.95)", now >= 0.95)

    print("\n=== 3. leave-one-season-out ===")
    for x in NEW:
        cells, neg = [], 0
        for s in sorted(seasons.unique()):
            k = (seasons != s).to_numpy()
            d, _, _ = paired_diff(R[x][k], R[BASE][k], groups=g[k])
            neg += d < 0
            cells.append(f"{d:+.5f}")
        print(f"  {x:12s} negative in {neg}/{len(cells)} folds   {' '.join(cells)}")

    print("\n=== 4. per-season ===")
    for x in NEW:
        cells, pos = [], 0
        for s in sorted(seasons.unique()):
            k = (seasons == s).to_numpy()
            d, _, _ = paired_diff(R[x][k], R[BASE][k], groups=g[k])
            pos += d < 0
            cells.append(f"{s}:{d:+.5f}")
        print(f"  {x:12s} better in {pos}/{len(seasons.unique())} seasons  {' '.join(cells)}")

    print("\n=== 5. collinearity with stakes_effort — the WP011 trap ===")
    print("    the deepest depletion is END-OF-SEASON ROTATION, which is exactly when")
    print("    stakes_effort fires too. If intactness is just stealing that variance,")
    print("    stakes_effort should shrink toward zero when intactness is added.")
    k = "param_stakes_effort"
    if k in wa.columns and k in wb.columns:
        v0 = wa[wa.arm == BASE].set_index("window")[k].dropna()
        v1 = wb[wb.arm == NEW[-1]].set_index("window")[k].dropna()
        j = v0.index.intersection(v1.index)
        v0, v1 = v0.loc[j], v1.loc[j]
        sh = (v1 - v0)
        print(f"  stakes_effort  baseline {v0.median():+.4f} -> with intactness {v1.median():+.4f}"
              f"   median shift {sh.median():+.4f}  ({sh.median()/abs(v0.median())*100:+.0f}%)")
        print(f"  |stakes_effort| shrank in {(v1.abs() < v0.abs()).sum()}/{len(j)} windows"
              f"   (>~{len(j)*0.7:.0f} would mean intactness is absorbing stakes)")

    w = wb[wb.arm.isin(NEW)]
    print(f"\n  sampler: divergences {int(w.n_diverging.sum())}  max rhat {w.max_rhat.max():.3f}"
          f"  median {w.seconds.median():.0f}s/fit")


if __name__ == "__main__":
    main()
