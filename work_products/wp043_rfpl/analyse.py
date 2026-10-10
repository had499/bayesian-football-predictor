"""WP043 — how does the model do against the RUSSIAN market?

THE BENCHMARK IS HARDER HERE. football-data's extra-league files carry CLOSING odds
only, so there are no openers to test against. On the EPL the model was ~0.003 RPS
behind Pinnacle's OPENING line and 0.0139 behind the CLOSE, and WP040 showed it added
nothing to the open. So beating or matching an RFPL CLOSING line would be a far stronger
result than anything achieved on the EPL.

Three questions:
  1. does the xG channel still work in this league? (L0 vs XG75)
  2. how far is the model from the RFPL closing line, vs the 0.0139 EPL closing gap?
  3. does the model ADD anything to the RFPL closing price? (WP040's blend test, whose
     power is established -- it detects closing-vs-opening at w=1.000)
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import load_checkpoints, paired_diff, verdict, wide  # noqa: E402

EPS = 1e-9


def shin(o):
    """de-vig 3 odds -> probabilities, proportional (overround here is ~3%, and Shin vs
    proportional differ negligibly at that level)."""
    p = 1.0 / o
    return p / p.sum(1, keepdims=True)


def rps3(p, y):
    o = np.zeros_like(p); o[np.arange(len(y)), y] = 1
    return ((p.cumsum(1) - o.cumsum(1))[:, :2] ** 2).sum(1) / 2


def logloss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1)).mean()


def blendp(pm_, pk, w):
    z = np.exp((1 - w) * np.log(pk + EPS) + w * np.log(pm_ + EPS))
    return z / z.sum(1, keepdims=True)


def main():
    shared = pickle.loads((HERE / "cv_shared_data_rfpl.pkl").read_bytes())
    df = shared["df_cv"]
    m, w = load_checkpoints(HERE / "checkpoints")
    full = m.groupby("arm").window.nunique()
    print("arms:", dict(full))
    arms = sorted(full[full == full.max()].index)
    com = set.intersection(*[set(m[m.arm == a].window) for a in arms])
    mm = m[m.window.isin(com)]

    print(f"\n=== 1. does the xG channel work in RFPL? ({len(com)} windows) ===")
    R = wide(mm[mm.arm.isin(arms)], "rps", arms)
    g = R.index.get_level_values("window").to_numpy()
    for a in arms:
        print(f"  {a:6s} RPS {R[a].mean():.5f}")
    if "L0" in arms and "XG75" in arms:
        d, lo, hi = paired_diff(R["XG75"], R["L0"], groups=g)
        print(f"  xG channel: {d:+.5f} [{lo:+.5f},{hi:+.5f}]  {verdict(lo, hi)}")
        print(f"  [EPL reference: -0.00258 [-0.00414,-0.00095], 6/6 LOSO]")

    # --- join predictions to odds ---
    o = pd.read_pickle(HERE / "odds_joined_rfpl.pkl")
    o["d"] = pd.to_datetime(o.Date).dt.normalize()
    best = "XG75" if "XG75" in arms else arms[0]
    pred = mm[mm.arm == best].copy()

    # JOIN ON THE FIXTURE, NOT THE DATE. The first version of this merged on `d` alone;
    # with ~8 RFPL matches per matchday that is a near-cartesian product, and the
    # score filter then happily pairs a 1-0 with a DIFFERENT 1-0 on the same day,
    # attaching the wrong odds. That scrambles the market's probabilities while
    # leaving the model's intact and manufactured a spectacular fake result
    # (blend weight 0.82, model "beating" the close by 0.0065).
    key = (df[df.is_home == 1][["cum_round", "team", "opp_team",
                                "team_long", "opp_team_long", "datetime"]]
           .rename(columns={"team": "home_c", "opp_team": "away_c"}))
    key["d"] = pd.to_datetime(key.datetime).dt.normalize()
    assert not key.duplicated(["cum_round", "home_c", "away_c"]).any()
    pred = pred.merge(key, left_on=["round", "home", "away"],
                      right_on=["cum_round", "home_c", "away_c"], how="inner")
    o2 = o[["d", "team_long", "opp_team_long", "PSCH", "PSCD", "PSCA",
            "MaxCH", "MaxCD", "MaxCA", "goals_home", "goals_away", "season"]].copy()
    assert not o2.duplicated(["d", "team_long", "opp_team_long"]).any(), \
        "odds key (date, home_long, away_long) must be unique"
    n_before = len(pred)
    j = pred.merge(o2, on=["d", "team_long", "opp_team_long"], how="inner",
                   suffixes=("", "_o"))
    print(f"\n  fixture-keyed join: {len(j)}/{n_before} predictions matched to odds")
    agree = (j.goals_home == j.goals_home_o) & (j.goals_away == j.goals_away_o)
    print(f"  score agreement: {agree.sum()}/{len(j)} "
          f"({'OK' if agree.all() else 'MISMATCHES — investigate'})")
    j = j[agree].copy()
    j = j.dropna(subset=["PSCH", "PSCD", "PSCA"])

    print(f"\n=== 2. model vs the RFPL CLOSING line ({len(j)} matches) ===")
    y = np.where(j.goals_home > j.goals_away, 0,
                 np.where(j.goals_home == j.goals_away, 1, 2))
    pm_ = j[["p_home", "p_draw", "p_away"]].to_numpy()
    pk = shin(j[["PSCH", "PSCD", "PSCA"]].to_numpy())
    rm, rk = rps3(pm_, y), rps3(pk, y)
    gg = j.window.to_numpy()
    d, lo, hi = paired_diff(pd.Series(rm), pd.Series(rk), groups=gg)
    print(f"  model  RPS {rm.mean():.5f}")
    print(f"  market RPS {rk.mean():.5f}   (Pinnacle CLOSING)")
    print(f"  gap    {d:+.5f} [{lo:+.5f},{hi:+.5f}]   {'MODEL AHEAD' if d < 0 else 'market ahead'}")
    print(f"  [EPL reference: +0.00315 vs the OPENING line, +0.0139 vs the CLOSE]")

    print(f"\n=== 3. does the model ADD to the RFPL closing price? ===")
    ws, lk, lb = [], [], []
    for s in sorted(j.season.unique()):
        tr, te = (j.season != s).to_numpy(), (j.season == s).to_numpy()
        if tr.sum() < 200 or te.sum() < 50:
            continue
        r = minimize_scalar(lambda ww: logloss(blendp(pm_[tr], pk[tr], ww), y[tr]),
                            bounds=(0, 1), method="bounded")
        ws.append(float(r.x))
        lk.append(logloss(pk[te], y[te]))
        lb.append(logloss(blendp(pm_[te], pk[te], float(r.x)), y[te]))
    gain = np.mean(lb) - np.mean(lk)
    print(f"  blend weight w = {np.mean(ws):.3f} [{min(ws):.3f},{max(ws):.3f}]"
          f"  over {len(ws)} seasons")
    print(f"  logloss  market {np.mean(lk):.5f} -> blend {np.mean(lb):.5f}  ({gain:+.5f})"
          f"  {'*** BLEND BEATS THE PRICE ***' if gain < 0 else ''}")
    print(f"  [EPL reference: w=0.046 vs OPEN, blend +0.00027 i.e. WORSE]")


if __name__ == "__main__":
    main()
