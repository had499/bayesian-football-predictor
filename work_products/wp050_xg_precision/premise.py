"""WP050 premise — is a match's xG MORE RELIABLE when it comes from more shots?

THE GAP. The xG channel is `xG ~ Gamma(mean=kappa*lambda, shape)` with `shape` a single
fitted constant (~3.2, CV ~56%). Every match's xG is therefore treated as equally
informative about the latent rate. But xG accumulated over 20 shots is a much more
precise measurement than xG from 4 shots, and the channel cannot say so.

If precision scales with shots, `shape_i = shape0 * shots_i / mean(shots)` would let the
channel lean on the matches that actually carry information. That is a LIKELIHOOD change
-- the only category that has ever improved this model -- aimed at its one established
mechanism (the xG channel, -0.00258).

TEST 1 (direct): bin matches by shot count and measure how tightly that match's xG
tracks the team's contemporaneous scoring rate. If high-shot xG is tighter, precision
really does scale.

TEST 2 (WP038 framework, with its anchor): does a SHOT-WEIGHTED rolling xG predict
next-match goals better than an unweighted one? The known-failure anchor is raw shot
count, which added +0.00362 here and FAILED as a channel in WP019; `+ goals` is the
positive control.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp008_market_benchmark"))
from market_analysis import CODE_TO_FD  # noqa: E402

K = 10


def main():
    df = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                      .read_bytes())["df_cv"]
    fd = pd.read_pickle(WP / "wp044_league_scan" / "odds_all_leagues.pkl")["EPL"].copy()
    fd["Date"] = pd.to_datetime(fd.Date, dayfirst=True, errors="coerce").dt.normalize()

    mt = df[df.is_home == 1][["datetime", "team", "opp_team", "goals_home",
                              "goals_away", "season"]].copy()
    mt["d"] = pd.to_datetime(mt.datetime).dt.normalize()
    mt["h_fd"] = mt.team.map(CODE_TO_FD); mt["a_fd"] = mt.opp_team.map(CODE_TO_FD)
    j = mt.merge(fd[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS"]],
                 left_on=["d", "h_fd", "a_fd"],
                 right_on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    j = j[(j.goals_home == j.FTHG) & (j.goals_away == j.FTAG)].copy()
    xg = df[["datetime", "team", "xG", "goals"]].copy()
    xg["d"] = pd.to_datetime(xg.datetime).dt.normalize()
    recs = []
    for _, r in j.iterrows():
        recs.append(dict(team=r.team, d=r.d, season=r.season, shots=r.HS))
        recs.append(dict(team=r.opp_team, d=r.d, season=r.season, shots=r.AS))
    s = (pd.DataFrame(recs).merge(xg.drop(columns="datetime"), on=["team", "d"],
                                  how="inner").sort_values("d"))
    print(f"panel: {len(s)} team-matches, shots mean {s.shots.mean():.1f}\n")

    print("=== 1. does xG track the team's rate more tightly when shots are high? ===")
    print("    reference rate = the team's leak-free rolling xG over its PREVIOUS 10")
    print("    matches. A match's xG should scatter LESS around that when it comes from")
    print("    more shots, if precision scales with shot count.\n")
    s["ref"] = (s.groupby("team").xG
                 .transform(lambda x: x.shift(1).rolling(K, min_periods=5).mean()))
    d = s.dropna(subset=["ref"]).copy()
    d["dev"] = (d.xG - d.ref) / d.ref            # relative deviation, scale-free
    print(f"  {'shots in match':18s} {'n':>5s} {'mean xG':>8s} {'CV of xG/ref':>14s}")
    d["sb"] = pd.cut(d.shots, [0, 7, 10, 13, 17, 60],
                     labels=["<=7", "8-10", "11-13", "14-17", "18+"])
    for b, k in d.groupby("sb", observed=True):
        print(f"  {str(b):18s} {len(k):5d} {k.xG.mean():8.3f} {k.dev.std():14.4f}")
    print("\n    if CV FALLS as shots rise, precision scales and the channel should know.")

    print("\n=== 2. does SHOT-WEIGHTED rolling xG predict next-match goals better? ===")
    for c in ("xG", "goals", "shots"):
        s[f"r_{c}"] = (s.groupby("team")[c]
                        .transform(lambda x: x.shift(1).rolling(K, min_periods=5).mean()))
    # shot-weighted rolling xG: sum(xG_i * shots_i) / sum(shots_i) over the window
    s["xs"] = s.xG * s.shots
    s["r_xs"] = (s.groupby("team").xs
                  .transform(lambda x: x.shift(1).rolling(K, min_periods=5).sum()))
    s["r_sh_sum"] = (s.groupby("team").shots
                      .transform(lambda x: x.shift(1).rolling(K, min_periods=5).sum()))
    s["r_xg_w"] = s.r_xs / s.r_sh_sum
    d2 = s.dropna(subset=["r_xG", "r_goals", "r_shots", "r_xg_w"]).copy()

    def loso(cols):
        errs, base = [], []
        for ss in sorted(d2.season.unique()):
            tr, te = d2[d2.season != ss], d2[d2.season == ss]
            X = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in cols])
            b, *_ = np.linalg.lstsq(X, tr.goals.to_numpy(), rcond=None)
            Xe = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in cols])
            errs.append(((te.goals.to_numpy() - Xe @ b) ** 2).sum())
            base.append(((te.goals.to_numpy() - tr.goals.mean()) ** 2).sum())
        return 1 - sum(errs) / sum(base)

    solo = loso(["r_xG"])
    print(f"  {'specification':38s} {'OOS R^2':>10s} {'vs xG alone':>13s}")
    print(f"  {'unweighted rolling xG (incumbent)':38s} {solo:+10.5f} {'—':>13s}")
    for lab, cols in (("+ goals        [CONTROL, real]", ["r_xG", "r_goals"]),
                      ("+ shot count   [ANCHOR: failed]", ["r_xG", "r_shots"]),
                      ("SHOT-WEIGHTED xG (replaces it)", ["r_xg_w"]),
                      ("both weighted & unweighted", ["r_xG", "r_xg_w"])):
        r = loso(cols)
        print(f"  {lab:38s} {r:+10.5f} {r - solo:+13.5f}")
    print("\n  the anchor is the bar: raw shot count scored +0.00362 and FAILED as a")
    print("  channel. Shot-weighting must beat the incumbent to be worth building.")


if __name__ == "__main__":
    main()
