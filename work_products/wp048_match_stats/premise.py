"""WP048 premise — do football-data's match stats carry anything xG does not?

WHAT IS NEW. WP038 closed the Understat SHOT-LEVEL feed, but football-data's main-league
files carry per-match counts we have never touched, all at 100% coverage and already on
disk from WP044's scan:

    HST/AST  shots ON TARGET      -- WP019 tested TOTAL shots (null); on-target is a
                                    higher-quality subset and a different signal
    HC/AC    corners              -- a territory proxy, the same family as `deep`, which
                                    WORKED (-0.00083)
    HF/AF    fouls
    HTHG/HTAG half-time goals     -- a first-half-only observation

THE CALIBRATION ANCHOR IS BUILT IN. HS/AS (total shots) is included deliberately: WP019
BUILT that as a measurement channel and it was NULL, actively cancelling the deep channel
(+0.00082 worse). So whatever OOS R^2 total shots scores HERE is the level at which a
feature is KNOWN to fail as a channel. Anything not clearly above it is presumed dead.
On WP038's Understat version of this test that level was +0.00484.

The `full + goals` row is the second control: goals are known to add real information
beyond xG (the model uses both), so a positive there proves the test has power.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
for p in (WP / "wp007_structural_fixes", WP / "wp008_market_benchmark"):
    sys.path.insert(0, str(p))
from market_analysis import CODE_TO_FD  # noqa: E402

K = 10


def main():
    df = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                      .read_bytes())["df_cv"]
    fd = pd.read_pickle(WP / "wp044_league_scan" / "odds_all_leagues.pkl")["EPL"].copy()
    fd["Date"] = pd.to_datetime(fd.Date, dayfirst=True, errors="coerce").dt.normalize()

    # match-level panel from our side, mapped to football-data names
    mt = df[df.is_home == 1][["datetime", "team", "opp_team", "goals_home",
                              "goals_away", "season"]].copy()
    mt["d"] = pd.to_datetime(mt.datetime).dt.normalize()
    mt["h_fd"] = mt.team.map(CODE_TO_FD); mt["a_fd"] = mt.opp_team.map(CODE_TO_FD)
    cols = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HTHG", "HTAG"]
    j = mt.merge(fd[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", *cols]],
                 left_on=["d", "h_fd", "a_fd"],
                 right_on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    ok = (j.goals_home == j.FTHG) & (j.goals_away == j.FTAG)
    print(f"join: {len(j)}/{len(mt)} matched, score agreement {ok.sum()}/{len(j)}")
    j = j[ok].copy()

    # -> long per-team panel, with the team's OWN stats and its own xG
    xg = df[["datetime", "team", "xG", "goals"]].copy()
    xg["d"] = pd.to_datetime(xg.datetime).dt.normalize()
    recs = []
    for _, r in j.iterrows():
        recs.append(dict(team=r.team, d=r.d, season=r.season, shots=r.HS, sot=r.HST,
                         corners=r.HC, fouls=r.HF, h1_goals=r.HTHG))
        recs.append(dict(team=r.opp_team, d=r.d, season=r.season, shots=r.AS, sot=r.AST,
                         corners=r.AC, fouls=r.AF, h1_goals=r.HTAG))
    s = pd.DataFrame(recs).merge(xg.drop(columns="datetime"), on=["team", "d"], how="inner")
    assert len(s) > 0.9 * 2 * len(j), f"xG join dropped too much: {len(s)}"
    s = s.sort_values("d")
    print(f"per-team panel: {len(s)} rows\n")

    feats = ["xG", "goals", "shots", "sot", "corners", "fouls", "h1_goals"]
    for c in feats:
        s[f"r_{c}"] = (s.groupby("team")[c]
                        .transform(lambda x: x.shift(1).rolling(K, min_periods=5).mean()))
    d = s.dropna(subset=[f"r_{c}" for c in feats]).copy()
    print(f"usable after rolling (K={K}): {len(d)} rows, "
          f"{d.season.nunique()} seasons\n")

    def loso(use):
        errs, base = [], []
        for ss in sorted(d.season.unique()):
            tr, te = d[d.season != ss], d[d.season == ss]
            X = np.column_stack([np.ones(len(tr))] + [tr[f"r_{c}"].to_numpy() for c in use])
            b, *_ = np.linalg.lstsq(X, tr.goals.to_numpy(), rcond=None)
            Xe = np.column_stack([np.ones(len(te))] + [te[f"r_{c}"].to_numpy() for c in use])
            errs.append(((te.goals.to_numpy() - Xe @ b) ** 2).sum())
            base.append(((te.goals.to_numpy() - tr.goals.mean()) ** 2).sum())
        return 1 - sum(errs) / sum(base)

    solo = loso(["xG"])
    print("=== incremental OOS R^2 for next-match goals, on top of team xG ===")
    print(f"  {'specification':34s} {'OOS R^2':>10s} {'vs xG alone':>13s}")
    print(f"  {'xG only (incumbent channel)':34s} {solo:+10.5f} {'—':>13s}")
    rows = [("+ goals          [CONTROL, real]", ["xG", "goals"]),
            ("+ TOTAL shots    [ANCHOR: failed]", ["xG", "shots"]),
            ("+ shots on target", ["xG", "sot"]),
            ("+ corners", ["xG", "corners"]),
            ("+ fouls", ["xG", "fouls"]),
            ("+ half-time goals", ["xG", "h1_goals"]),
            ("+ sot & corners", ["xG", "sot", "corners"]),
            ("everything", feats)]
    res = {}
    for lab, use in rows:
        r = loso(use)
        res[lab] = r - solo
        print(f"  {lab:34s} {r:+10.5f} {r - solo:+13.5f}")

    anchor = res["+ TOTAL shots    [ANCHOR: failed]"]
    ctrl = res["+ goals          [CONTROL, real]"]
    print(f"\n  CONTROL (+goals) = {ctrl:+.5f}  -> the test {'HAS' if ctrl > 0 else 'LACKS'} power")
    print(f"  ANCHOR (+total shots) = {anchor:+.5f}  -> WP019 built this and it FAILED")
    print(f"  so the bar to clear is ~{anchor:+.5f}; anything below it is presumed dead.")
    live = [k for k, v in res.items() if v > anchor and "ANCHOR" not in k and "everything" not in k]
    print(f"\n  clears the bar: {live if live else 'NOTHING'}")




def vs_champion():
    """THE DECISIVE TEST. Corners clears the known-failure anchor by only 14%
    (+0.00414 vs +0.00362), and corners and `deep` are BOTH territory proxies. WP021
    measured shots and deep actively CANCELLING each other (+0.00082 worse together),
    so the question is not whether corners beats xG -- it is whether corners adds
    anything on top of xG AND DEEP, which is what the champion already has.
    """
    import pickle
    import numpy as np
    import pandas as pd

    df = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                      .read_bytes())["df_cv"]
    fd = pd.read_pickle(WP / "wp044_league_scan" / "odds_all_leagues.pkl")["EPL"].copy()
    fd["Date"] = pd.to_datetime(fd.Date, dayfirst=True, errors="coerce").dt.normalize()
    mt = df[df.is_home == 1][["datetime", "team", "opp_team", "goals_home", "goals_away",
                              "season", "deep_home", "deep_away"]].copy()
    mt["d"] = pd.to_datetime(mt.datetime).dt.normalize()
    mt["h_fd"] = mt.team.map(CODE_TO_FD); mt["a_fd"] = mt.opp_team.map(CODE_TO_FD)
    j = mt.merge(fd[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
                     "HS", "AS", "HST", "AST", "HC", "AC"]],
                 left_on=["d", "h_fd", "a_fd"],
                 right_on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    j = j[(j.goals_home == j.FTHG) & (j.goals_away == j.FTAG)].copy()
    xg = df[["datetime", "team", "xG", "goals"]].copy()
    xg["d"] = pd.to_datetime(xg.datetime).dt.normalize()
    recs = []
    for _, r in j.iterrows():
        recs.append(dict(team=r.team, d=r.d, season=r.season, corners=r.HC,
                         sot=r.HST, deep=r.deep_home))
        recs.append(dict(team=r.opp_team, d=r.d, season=r.season, corners=r.AC,
                         sot=r.AST, deep=r.deep_away))
    s = (pd.DataFrame(recs).merge(xg.drop(columns="datetime"),
                                  on=["team", "d"], how="inner").sort_values("d"))
    feats = ["xG", "deep", "corners", "sot", "goals"]
    for c in feats:
        s[f"r_{c}"] = (s.groupby("team")[c]
                        .transform(lambda x: x.shift(1).rolling(K, min_periods=5).mean()))
    d = s.dropna(subset=[f"r_{c}" for c in feats]).copy()

    def loso(use):
        errs, base = [], []
        for ss in sorted(d.season.unique()):
            tr, te = d[d.season != ss], d[d.season == ss]
            X = np.column_stack([np.ones(len(tr))] + [tr[f"r_{c}"].to_numpy() for c in use])
            b, *_ = np.linalg.lstsq(X, tr.goals.to_numpy(), rcond=None)
            Xe = np.column_stack([np.ones(len(te))] + [te[f"r_{c}"].to_numpy() for c in use])
            errs.append(((te.goals.to_numpy() - Xe @ b) ** 2).sum())
            base.append(((te.goals.to_numpy() - tr.goals.mean()) ** 2).sum())
        return 1 - sum(errs) / sum(base)

    print(f"\n\n=== DECISIVE: does corners add on top of the CHAMPION'S xG + deep? "
          f"(n={len(d)}) ===")
    base = loso(["xG", "deep"])
    print(f"  {'specification':36s} {'OOS R^2':>10s} {'vs xG+deep':>12s}")
    print(f"  {'xG + deep (the champion)':36s} {base:+10.5f} {'—':>12s}")
    for lab, use in (("+ goals    [CONTROL, real]", ["xG", "deep", "goals"]),
                     ("+ corners", ["xG", "deep", "corners"]),
                     ("+ shots on target", ["xG", "deep", "sot"]),
                     ("+ corners & sot", ["xG", "deep", "corners", "sot"])):
        r = loso(use)
        print(f"  {lab:36s} {r:+10.5f} {r - base:+12.5f}")
    print(f"\n  xG alone, for reference: {loso(['xG']):+.5f}  "
          f"-> deep is worth {base - loso(['xG']):+.5f} on this test")
    print("  if corners adds ~nothing once deep is present, it is the WP021 shots-vs-deep")
    print("  cancellation again: two measures of the same territory signal.")


if __name__ == "__main__":
    main()
    vs_champion()
