"""WP043 — is the RFPL edge worth money?

HARDEST POSSIBLE ENTRY. football-data's extra-league files carry CLOSING odds only, so
every bet here is struck at the CLOSE -- the sharpest price of the market's life. On the
EPL the model was 0.0139 behind the close and WP042 showed negative edge even at the
OPEN. Any positive return here is therefore a conservative read, not an optimistic one.

CLV is unavailable (it needs an opening price to compare against a close), so this falls
back on ROI with its known power limits -- MDE was 6.66% on EPL 1X2. The RPS and
log-loss evidence is the better-powered part of this work package; ROI is here to check
the sign and magnitude, and the no-edge CLV-style benchmark is not computable.

RFPL Pinnacle closing overround is 3.19% vs the EPL's 2.40%, so the margin to overcome
is 33% wider.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import load_checkpoints, wide  # noqa: E402

BOOKS = {"pinnacle_close": ["PSCH", "PSCD", "PSCA"],
         "market_max_close": ["MaxCH", "MaxCD", "MaxCA"]}


def roi_ci(pnl, stake, g, reps=4000, seed=0):
    uq = np.unique(g); idx = {w: np.where(g == w)[0] for w in uq}
    rng = np.random.default_rng(seed); out = np.empty(reps)
    for i in range(reps):
        sel = np.concatenate([idx[w] for w in rng.choice(uq, len(uq), replace=True)])
        out[i] = pnl[sel].sum() / stake[sel].sum()
    return pnl.sum() / stake.sum(), *np.percentile(out, [2.5, 97.5])


def main():
    shared = pickle.loads((HERE / "cv_shared_data_rfpl.pkl").read_bytes())
    df = shared["df_cv"]
    m, _ = load_checkpoints(HERE / "checkpoints")
    pred = m[m.arm == "XG75"].copy()
    key = (df[df.is_home == 1][["cum_round", "team", "opp_team",
                                "team_long", "opp_team_long", "datetime"]]
           .rename(columns={"team": "home_c", "opp_team": "away_c"}))
    key["d"] = pd.to_datetime(key.datetime).dt.normalize()
    pred = pred.merge(key, left_on=["round", "home", "away"],
                      right_on=["cum_round", "home_c", "away_c"], how="inner")
    o = pd.read_pickle(HERE / "odds_joined_rfpl.pkl")
    o["d"] = pd.to_datetime(o.Date).dt.normalize()
    j = pred.merge(o[["d", "team_long", "opp_team_long", "goals_home", "goals_away",
                      *sum(BOOKS.values(), [])]],
                   on=["d", "team_long", "opp_team_long"], how="inner",
                   suffixes=("", "_o"))
    assert ((j.goals_home == j.goals_home_o) & (j.goals_away == j.goals_away_o)).all()
    y = np.where(j.goals_home > j.goals_away, 0,
                 np.where(j.goals_home == j.goals_away, 1, 2))
    pm_ = j[["p_home", "p_draw", "p_away"]].to_numpy()
    win = j.window.to_numpy()

    print(f"\nRFPL, betting at the CLOSING price, flat stakes  (n={len(j)} matches)")
    print(f"  {'book':18s} {'edge':>5s} {'bets':>6s} {'%played':>8s} {'ROI':>8s} "
          f"{'CI (window-clustered)':>24s}")
    for book, cols in BOOKS.items():
        d = j.dropna(subset=cols)
        if not len(d):
            continue
        odds = d[cols].to_numpy()
        pmd = d[["p_home", "p_draw", "p_away"]].to_numpy()
        yd = y[d.index.map(lambda i: j.index.get_loc(i))] if False else np.where(
            d.goals_home > d.goals_away, 0,
            np.where(d.goals_home == d.goals_away, 1, 2))
        g = d.window.to_numpy()
        ov = (1 / odds).sum(1).mean() - 1
        print(f"  {book:18s} (overround {ov:.2%})")
        for edge in (0.0, 0.02, 0.05, 0.10):
            ev = pmd * odds - 1.0
            take = ev > edge
            if take.sum() < 50:
                print(f"  {'':18s} {edge:5.0%} {int(take.sum()):6d}    too few"); continue
            i, k = np.where(take)
            won = (yd[i] == k)
            pnl = np.where(won, odds[i, k] - 1.0, -1.0)
            r, lo, hi = roi_ci(pnl, np.ones(len(pnl)), g[i])
            flag = "  <-- CI excludes 0" if lo > 0 else ""
            print(f"  {'':18s} {edge:5.0%} {len(pnl):6d} {len(pnl)/(3*len(d)):7.1%} "
                  f"{r:+7.2%} [{lo:+6.2%},{hi:+6.2%}]{flag}")
        print()
    print("  NOTE: these are bets at the CLOSE, the hardest price available. No opening")
    print("  odds exist for this league, so CLV cannot be computed and ROI's weak power")
    print("  (MDE ~5-7%) applies. The RPS/log-loss result is the better evidence.")


if __name__ == "__main__":
    main()
