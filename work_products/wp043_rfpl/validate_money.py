"""WP043b — validate (or kill) the +17% ROI claim.

Four tests, cheapest first:
  1. SAME-SAMPLE calibration, binned by model prob AND by market prob. The earlier
     contradiction (model-binned showed market p 0.179 -> actual 0.108, while
     market-binned showed the market calibrated to ~1pp) compared TWO DIFFERENT SAMPLES:
     2,234 test matches vs 2,682 all-odds matches. Same sample resolves it.
  2. PER-SEASON ROI -- is it one freak season? The headline CI treats windows as the
     cluster, which does NOT protect against a single-season effect.
  3. ROI BY ODDS BAND -- mean odds taken were 4.411, so if the edge lives only in
     longshots it is fragile and probably unstakeable.
  4. ROI EXCLUDING longshots. If removing odds > 6 kills it, the result is a thin-tail
     artefact rather than a tradeable edge.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wp007_structural_fixes"))
from analysis import load_checkpoints  # noqa: E402

COLS = ["PSCH", "PSCD", "PSCA"]


def roi_ci(pnl, g, reps=4000, seed=0):
    uq = np.unique(g); idx = {w: np.where(g == w)[0] for w in uq}
    rng = np.random.default_rng(seed); out = np.empty(reps)
    for i in range(reps):
        sel = np.concatenate([idx[w] for w in rng.choice(uq, len(uq), replace=True)])
        out[i] = pnl[sel].mean()
    return pnl.mean(), *np.percentile(out, [2.5, 97.5])


def build():
    shared = pickle.loads((HERE / "cv_shared_data_rfpl.pkl").read_bytes())
    df = shared["df_cv"]
    m, _ = load_checkpoints(HERE / "checkpoints")
    pred = m[m.arm == "XG75"].copy()
    key = (df[df.is_home == 1][["cum_round", "team", "opp_team", "team_long",
                                "opp_team_long", "datetime"]]
           .rename(columns={"team": "home_c", "opp_team": "away_c"}))
    key["d"] = pd.to_datetime(key.datetime).dt.normalize()
    pred = pred.merge(key, left_on=["round", "home", "away"],
                      right_on=["cum_round", "home_c", "away_c"], how="inner")
    o = pd.read_pickle(HERE / "odds_joined_rfpl.pkl")
    o["d"] = pd.to_datetime(o.Date).dt.normalize()
    j = pred.merge(o[["d", "team_long", "opp_team_long", "goals_home", "goals_away",
                      "season", *COLS]],
                   on=["d", "team_long", "opp_team_long"], how="inner",
                   suffixes=("", "_o")).dropna(subset=COLS)
    assert ((j.goals_home == j.goals_home_o) & (j.goals_away == j.goals_away_o)).all()
    return j.reset_index(drop=True)


def main():
    j = build()
    y = np.where(j.goals_home > j.goals_away, 0,
                 np.where(j.goals_home == j.goals_away, 1, 2))
    odds = j[COLS].to_numpy()
    pm_ = j[["p_home", "p_draw", "p_away"]].to_numpy()
    pk = 1.0 / odds; pk = pk / pk.sum(1, keepdims=True)
    hit = (np.arange(3)[None, :] == y[:, None])
    win = j.window.to_numpy()
    print(f"matched sample: {len(j)} matches, {3*len(j)} legs\n")

    print("=== 1. SAME-SAMPLE calibration ===")
    for lab, src in (("binned by MODEL prob", pm_), ("binned by MARKET prob", pk)):
        f, h, mo, mk = src.ravel(), hit.ravel(), pm_.ravel(), pk.ravel()
        print(f"  {lab}")
        print(f"    {'bin':>12s} {'n':>6s} {'model':>7s} {'market':>7s} {'ACTUAL':>7s}")
        for a, b in [(0,.1),(.1,.2),(.2,.3),(.3,.4),(.4,.55),(.55,1.0)]:
            s = (f >= a) & (f < b)
            if s.sum() < 40: continue
            print(f"    {f'{a:.2f}-{b:.2f}':>12s} {s.sum():6d} {mo[s].mean():7.3f} "
                  f"{mk[s].mean():7.3f} {h[s].mean():7.3f}")

    ev = pm_ * odds - 1.0
    take = ev > 0.0

    print("\n=== 2. PER-SEASON ROI (edge>0) — one freak season? ===")
    print(f"  {'season':>8s} {'bets':>6s} {'ROI':>8s} {'cum bank':>10s}")
    bank = 0.0
    for s in sorted(j.season.unique()):
        sel = (j.season == s).to_numpy()
        i, k = np.where(take & sel[:, None])
        if len(i) < 30:
            print(f"  {s:>8s} {len(i):6d}   too few"); continue
        pnl = np.where(y[i] == k, odds[i, k] - 1.0, -1.0)
        bank += pnl.sum()
        print(f"  {s:>8s} {len(pnl):6d} {pnl.mean():+7.2%} {bank:+10.1f}")
    pos = sum(1 for s in sorted(j.season.unique())
              if ((lambda i, k: len(i) >= 30 and
                   np.where(y[i] == k, odds[i, k] - 1.0, -1.0).mean() > 0)
                  (*np.where(take & (j.season == s).to_numpy()[:, None]),)))
    print(f"  profitable in {pos}/{j.season.nunique()} seasons")

    print("\n=== 3. ROI BY ODDS BAND — where does the money come from? ===")
    i, k = np.where(take)
    od, pnl = odds[i, k], np.where(y[i] == k, odds[i, k] - 1.0, -1.0)
    g = win[i]
    print(f"  {'odds band':>14s} {'bets':>6s} {'implied p':>10s} {'actual':>8s} "
          f"{'ROI':>8s} {'CI':>20s}")
    for a, b in [(1,2),(2,3),(3,4),(4,6),(6,10),(10,100)]:
        s = (od >= a) & (od < b)
        if s.sum() < 40: continue
        won = (pnl[s] > 0)
        r, lo, hi = roi_ci(pnl[s], g[s])
        print(f"  {f'{a}-{b}':>14s} {s.sum():6d} {(1/od[s]).mean():10.3f} "
              f"{won.mean():8.3f} {r:+7.2%} [{lo:+6.2%},{hi:+6.2%}]")

    print("\n=== 4. ROI EXCLUDING longshots — is it a thin-tail artefact? ===")
    for cap in (100, 10, 6, 4, 3):
        s = od < cap
        if s.sum() < 100: continue
        r, lo, hi = roi_ci(pnl[s], g[s])
        tag = "  <-- CI excludes 0" if lo > 0 else ""
        print(f"  odds < {cap:<4d} n={s.sum():5d}  ROI {r:+7.2%} "
              f"[{lo:+6.2%},{hi:+6.2%}]{tag}")
    print("\n  if the edge survives only with the long tail in, it is not tradeable:")
    print("  those are the bets with the lowest limits and the widest true margins.")


if __name__ == "__main__":
    main()
