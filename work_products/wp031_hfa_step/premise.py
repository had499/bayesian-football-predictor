"""WP031 premise — is the champion's home advantage MIS-FIT by era?

WP015 tested a time-varying HFA as a slow RANDOM WALK and it was null. The
specification it never tested is a STEP: an abrupt level change with a KNOWN,
EXTERNAL breakpoint. The COVID finding motivates it -- home win rate went
0.477 -> 0.379 -> 0.442, which a slow walk with small sigma physically cannot
represent (it cannot jump down and back up within two seasons).

This measures the defect in the CHAMPION'S OWN out-of-sample predictions, not a raw
data pattern, because only the former is a reason to change the model.

TILT = (home goals - lambda_home) - (away goals - lambda_away), in goals per match.
Positive = the model under-rates home teams in that era.

Crowd regime is a SCHEDULING FACT, known months ahead, so it is legitimate at the
opening price:  Project Restart 2020-06-17 -> fans back ~2021-05-10.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import load_checkpoints, paired_diff  # noqa: E402

CROWDLESS_FROM = pd.Timestamp("2020-06-17")   # Project Restart
CROWDLESS_TO = pd.Timestamp("2021-05-09")     # limited fans return for final rounds


def boot_mean(x, groups, n=4000, seed=0):
    """Window-cluster bootstrap CI for a mean -- same machinery as paired_diff."""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uq}
    out = np.empty(n)
    for i in range(n):
        pick = rng.choice(uq, size=len(uq), replace=True)
        out[i] = x[np.concatenate([idx[g] for g in pick])].mean()
    return x.mean(), *np.percentile(out, [2.5, 97.5])


def main():
    df = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                      .read_bytes())["df_cv"]
    dates = (df[df.is_home == 1][["round", "team", "opp_team", "datetime", "season"]]
             .rename(columns={"team": "home", "opp_team": "away"}))
    m, _ = load_checkpoints(WP / "wp028_stakes/checkpoints")
    m = m[m.arm == "I_effort"].merge(dates, on=["round", "home", "away"], how="left")
    assert m.datetime.notna().all(), f"{m.datetime.isna().sum()} unmatched"
    m["datetime"] = pd.to_datetime(m.datetime)

    m["tilt"] = ((m.goals_home - m.lambda_home) - (m.goals_away - m.lambda_away))
    m["hw"] = (m.goals_home > m.goals_away).astype(float)
    m["hw_err"] = m.hw - m.p_home
    m["regime"] = np.where(m.datetime < CROWDLESS_FROM, "1 pre-covid",
                  np.where(m.datetime <= CROWDLESS_TO, "2 crowdless", "3 post-covid"))

    print(f"champion out-of-sample, n={len(m)}, {m.window.nunique()} windows\n")
    print("=== 1. the model's HOME TILT by crowd regime ===")
    print("    tilt > 0 means the model UNDER-rates home teams in that era\n")
    print(f"  {'regime':14s} {'n':>5s}  {'tilt (goals/match)':>26s}  {'home win rate':>14s}"
          f"  {'model P(home)':>13s}")
    for r, k in m.groupby("regime"):
        t, lo, hi = boot_mean(k.tilt.to_numpy(), k.window.to_numpy())
        print(f"  {r:14s} {len(k):5d}  {t:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"{k.hw.mean():14.3f}  {k.p_home.mean():13.3f}")

    print("\n=== 2. is the POST-covid level different from PRE? ===")
    print("    this is the part with FORWARD value -- the crowdless window is over,")
    print("    but if HFA stepped DOWN permanently, a global HFA over-rates it now.")
    pre = m[m.regime == "1 pre-covid"]
    post = m[m.regime == "3 post-covid"]
    for lab, k in (("pre-covid", pre), ("post-covid", post)):
        t, lo, hi = boot_mean(k.tilt.to_numpy(), k.window.to_numpy())
        print(f"  {lab:12s} n={len(k):5d}  tilt {t:+.4f} [{lo:+.4f},{hi:+.4f}]"
              f"   observed home win {k.hw.mean():.3f}  model {k.p_home.mean():.3f}")
    d = post.tilt.mean() - pre.tilt.mean()
    print(f"  difference (post - pre) = {d:+.4f} goals/match")

    print("\n=== 3. by season — does it look like a STEP or a DRIFT? ===")
    print("    a drift is what WP015 already tested and rejected; a step is new.")
    print(f"  {'season':8s} {'n':>5s} {'tilt':>9s} {'home win':>9s} {'model':>7s} {'err':>8s}")
    for s, k in m.groupby("season"):
        print(f"  {s:<8} {len(k):5d} {k.tilt.mean():+9.4f} {k.hw.mean():9.3f} "
              f"{k.p_home.mean():7.3f} {k.hw_err.mean():+8.3f}")

    print("\n=== 4. what the model FITTED for home_mu (prior Normal(0.13, 0.03)) ===")
    _, w = load_checkpoints(WP / "wp028_stakes/checkpoints")
    w = w[w.arm == "I_effort"]
    if "param_home_mu" in w.columns:
        v = w[["window", "param_home_mu"]].dropna().sort_values("window")
        print(f"  median {v.param_home_mu.median():.4f}, first 10 windows "
              f"{v.param_home_mu.head(10).mean():.4f}, last 10 {v.param_home_mu.tail(10).mean():.4f}")
        print("  (if the prior is pinning it, the data's view never gets expressed)")


if __name__ == "__main__":
    main()
