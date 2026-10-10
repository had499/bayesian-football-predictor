"""WP027 step 2 — does TRUE congestion predict what the model gets wrong?

WP009 measured league-only rest days and found nothing. That measure is wrong for
any club playing midweek in a cup or in Europe -- a Thursday Europa League trip
followed by a Sunday league game looks identically rested to a clear week. This
rebuilds rest days from the COMPLETE fixture list and re-tests.

The decisive comparison is rest_all vs rest_league. If the complete measure predicts
residuals where the league-only one does not, WP009's null was a measurement artefact.

Cup coverage starts 2018-19, so the headline test uses Understat seasons 2018+ where
the calendar is complete. Earlier seasons have European fixtures only and are reported
separately as a partial measure.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from aliases import to_code  # noqa: E402

FULL_CUP_FROM = "2018"   # Understat season label; cups exist in openfootball from 2018-19


def build():
    df = pickle.loads((WP / "wp021_power_confirmation/cv_shared_data_big.pkl").read_bytes())["df_cv"]
    fx = pickle.loads((HERE / "fixtures_raw.pkl").read_bytes())
    fx["code"] = fx.team.map(to_code)
    fx = fx[fx.code.notna()][["code", "date", "comp"]].rename(columns={"code": "team"})

    # league fixtures from the model's own data, both perspectives
    lg = pd.concat([
        df[["team", "datetime"]].rename(columns={"datetime": "date"}),
        df[["opp_team", "datetime"]].rename(columns={"opp_team": "team", "datetime": "date"}),
    ], ignore_index=True)
    lg["date"] = pd.to_datetime(lg.date).dt.normalize()
    lg["comp"] = "League"
    fx["date"] = pd.to_datetime(fx.date).dt.normalize()

    cal_all = pd.concat([lg, fx], ignore_index=True).drop_duplicates(["team", "date"]).sort_values("date")
    cal_lg = lg.drop_duplicates(["team", "date"]).sort_values("date")
    return df, cal_all, cal_lg, fx


def feats(cal, team, when):
    """days since the team's previous fixture, and load in the prior 14 days"""
    s = cal[(cal.team == team) & (cal.date < when)]
    if not len(s):
        return np.nan, 0
    prev = s.date.iloc[-1]
    return (when - prev).days, int((s.date >= when - pd.Timedelta("14D")).sum())


def main():
    df, cal_all, cal_lg, fx = build()
    rows = []
    for f in sorted((WP / "wp023_nutpie_baseline/checkpoints/H_deep_added").glob("w*.pkl")):
        rows.extend(pickle.loads(f.read_bytes())["cv_match_predictions"])
    R = pd.DataFrame(rows)
    meta = df[df.is_home == 1].drop_duplicates("cum_round").set_index("cum_round")[["datetime", "season"]]
    R["dt"] = pd.to_datetime(R["round"].map(meta.datetime)).dt.normalize()
    R["season"] = R["round"].map(meta.season)
    R = R.dropna(subset=["dt"])

    idx_all = {t: g.date.to_numpy() for t, g in cal_all.groupby("team")}
    idx_lg = {t: g.date.to_numpy() for t, g in cal_lg.groupby("team")}

    def quick(idx, team, when):
        a = idx.get(team)
        if a is None:
            return np.nan, 0
        p = a[a < np.datetime64(when)]
        if not len(p):
            return np.nan, 0
        return (when - pd.Timestamp(p[-1])).days, int((p >= np.datetime64(when - pd.Timedelta("14D"))).sum())

    for side in ("home", "away"):
        R[f"rest_all_{side}"], R[f"load_{side}"] = zip(*[quick(idx_all, t, w) for t, w in zip(R[side], R.dt)])
        R[f"rest_lg_{side}"], _ = zip(*[quick(idx_lg, t, w) for t, w in zip(R[side], R.dt)])
    R["res_margin"] = (R.goals_home - R.lambda_home) - (R.goals_away - R.lambda_away)
    for k in ("rest_all", "rest_lg", "load"):
        R[f"d_{k}"] = R[f"{k}_home"] - R[f"{k}_away"]
    R = R.dropna(subset=["d_rest_all", "d_rest_lg"])
    print(f"{len(R)} matches with a complete fixture calendar\n")
    print("How often does the COMPLETE measure disagree with the league-only one?")
    dis = (R.rest_all_home != R.rest_lg_home) | (R.rest_all_away != R.rest_lg_away)
    print(f"  rest days differ for at least one side: {dis.mean()*100:.1f}% of matches")
    print(f"  median rest: all-comps {np.median(R.rest_all_home):.0f}d vs "
          f"league-only {np.median(R.rest_lg_home):.0f}d")
    print(f"  short rest (<=3d) under all-comps {int((R.rest_all_home<=3).sum())} vs "
          f"league-only {int((R.rest_lg_home<=3).sum())} home sides\n")

    teams = R.home.values
    uq = np.unique(teams)
    ix = {t: np.where(teams == t)[0] for t in uq}
    rng = np.random.default_rng(0)

    def ci(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        r = np.corrcoef(x, y)[0, 1]
        bs = []
        for _ in range(3000):
            i = np.concatenate([ix[t] for t in rng.choice(uq, len(uq), replace=True)])
            i = i[i < len(x)]
            bs.append(np.corrcoef(x[i], y[i])[0, 1])
        return r, *np.percentile(bs, [2.5, 97.5])

    for lbl, sub in (("ALL SEASONS (European only before 2018)", R),
                     (f"FULL CUP COVERAGE ({FULL_CUP_FROM}+)", R[R.season >= FULL_CUP_FROM])):
        print(f"=== {lbl}  n={len(sub)} ===")
        y = sub.res_margin.to_numpy()
        for c, name in (("d_rest_lg", "league-only rest diff (WP009's measure)"),
                        ("d_rest_all", "ALL-COMPETITIONS rest diff"),
                        ("d_load", "14-day fixture load diff")):
            r, lo, hi = ci(sub[c].to_numpy(float), y)
            print(f"  {name:42s} r={r:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                  f"{'REAL' if lo>0 or hi<0 else 'null'}")
        print()


if __name__ == "__main__":
    main()
