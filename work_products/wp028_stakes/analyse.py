"""WP028 — do the stakes arms help, and is `effort` measuring what it claims?

Two checks beyond the pooled CI:

1. LEAVE-ONE-SEASON-OUT. This is what killed the deep channel: -0.00083 looked solid
   until dropping 2016 alone halved it to -0.00044. The Elo anchor passed 10/10 folds.
   For an effect this small, fold stability matters more than the pooled interval.

2. THE CONFOUND. `effort` fits at ~-0.54, meaning a team at maximum stakes expresses
   roughly HALF its latent quality advantage -- an enormous behavioural claim, and the
   coefficient sits at 180% of its prior SD. High stakes also correlate with being
   MID-STRENGTH (title races involve two clubs; by April the top sides are often
   already safe in Europe, while relegation and European races are fought by middling
   teams). So a term suppressing expressed quality for high-stakes teams may simply be
   shrinking predictions toward even for mid-table sides -- improving RPS for reasons
   unrelated to incentive. That is the WP011 failure mode in new clothes. Test: does
   the stakes-residual relationship survive controlling for calendar position and the
   model's own implied strength gap?
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

BASE = "H_deep_added"
NEW = ["I_effort", "J_disp"]


def main():
    df = pickle.loads((WP / "wp021_power_confirmation/cv_shared_data_big.pkl").read_bytes())["df_cv"]
    a, _ = load_checkpoints(WP / "wp023_nutpie_baseline/checkpoints")
    b, wb = load_checkpoints(HERE / "checkpoints")
    m = attach_meta(pd.concat([a[a.arm == BASE], b], ignore_index=True), df)
    arms = [BASE] + NEW
    com = set.intersection(*[set(m[m.arm == x].window) for x in arms])
    m = m[m.window.isin(com)]
    R = wide(m[m.arm.isin(arms)], "rps", arms)
    g = R.index.get_level_values("window").to_numpy()
    seasons = m[m.arm == BASE].set_index(["window", "round", "home", "away"]).loc[R.index, "season"]
    print(f"paired on {len(R)} matches / {len(com)} windows\n")
    print(f"  {BASE:12s} {R[BASE].mean():.5f}")
    for x in NEW:
        d, lo, hi = paired_diff(R[x], R[BASE], groups=g)
        print(f"  {x:12s} {R[x].mean():.5f}   {d:+.5f} [{lo:+.5f},{hi:+.5f}]  {verdict(lo,hi)}")
    print(f"\n  [ref] xG -0.00258 | deep -0.00083 | Elo anchor -0.00021\n")

    print("=== 1. leave-one-season-out ===")
    for x in NEW:
        neg = 0
        cells = []
        for s in sorted(seasons.unique()):
            k = (seasons != s).to_numpy()
            d, lo, hi = paired_diff(R[x][k], R[BASE][k], groups=g[k])
            neg += d < 0
            cells.append(f"{d:+.5f}")
        print(f"  {x:12s} negative in {neg}/{len(cells)} folds   {' '.join(cells)}")
    print("    (deep HALVED on dropping 2016; the Elo anchor was negative 10/10)\n")

    print("=== per-season ===")
    for x in NEW:
        pos = 0
        for s in sorted(seasons.unique()):
            k = (seasons == s).to_numpy()
            d, _, _ = paired_diff(R[x][k], R[BASE][k], groups=g[k])
            pos += d < 0
        print(f"  {x:12s} better in {pos}/{len(seasons.unique())} seasons")

    print("\n=== learned coefficients ===")
    for arm, key, sd in (("I_effort", "param_stakes_effort", 0.3),
                         ("J_disp", "param_stakes_disp", 0.5)):
        w = wb[wb.arm == arm]
        if key not in w.columns:
            continue
        v = w[key].dropna()
        print(f"  {arm:10s} median {v.median():+.4f} = {abs(v.median())/sd*100:3.0f}% of prior SD,"
              f" sign-consistent {max((v>0).sum(),(v<0).sum())}/{len(v)}")
    w = wb[wb.arm.isin(NEW)]
    print(f"  sampler: divergences {int(w.n_diverging.sum())}  max rhat {w.max_rhat.max():.3f}")


if __name__ == "__main__":
    main()
