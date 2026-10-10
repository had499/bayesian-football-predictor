"""WP027 step 3 — attach congestion covariates to the CV dataframe.

LEAK SAFETY. Rest is computed from fixtures STRICTLY BEFORE the match being
predicted, so it is legitimate pre-match information. Fixture schedules are known
weeks ahead, so this is knowable at prediction time in reality too -- unlike, say,
realised xG, which is only an observation for training rows.

STANDARDISATION. Rest and load are z-scored against the training distribution so the
fitted coefficient reads as "log-goals per SD", comparable to the other covariates.
Rest is also CLIPPED at 14 days: beyond a fortnight the difference between 20 and 40
days' rest is an international break, not recovery, and leaving it unclipped lets
summer gaps dominate the scale.

    python work_products/wp027_congestion/prepare_data.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(HERE))
from aliases import to_code  # noqa: E402

OUT = HERE / "cv_shared_data_congestion.pkl"
REST_CAP = 14


def main():
    shared = pickle.loads((WP / "wp021_power_confirmation/cv_shared_data_big.pkl").read_bytes())
    df = shared["df_cv"].copy()
    fx = pickle.loads((HERE / "fixtures_raw.pkl").read_bytes())
    fx["code"] = fx.team.map(to_code)
    fx = fx[fx.code.notna()][["code", "date"]].rename(columns={"code": "team"})
    fx["date"] = pd.to_datetime(fx.date).dt.normalize()

    df["d"] = pd.to_datetime(df.datetime).dt.normalize()
    lg = pd.concat([df[["team", "d"]],
                    df[["opp_team", "d"]].rename(columns={"opp_team": "team"})], ignore_index=True)
    lg = lg.rename(columns={"d": "date"})
    cal = (pd.concat([lg, fx], ignore_index=True)
             .drop_duplicates(["team", "date"]).sort_values("date"))
    idx = {t: g.date.to_numpy() for t, g in cal.groupby("team")}
    print(f"calendar: {len(cal)} team-fixtures for {len(idx)} clubs "
          f"({len(lg)} league + {len(fx)} cup/European)")

    def pair(team, when):
        a = idx.get(team)
        if a is None:
            return np.nan, 0
        w = np.datetime64(when)
        p = a[a < w]
        if not len(p):
            return np.nan, 0
        rest = (when - pd.Timestamp(p[-1])).days
        load = int((p >= w - np.timedelta64(14, "D")).sum())
        return min(rest, REST_CAP), load

    for who, col in (("team", "self"), ("opp_team", "opp")):
        vals = [pair(t, w) for t, w in zip(df[who], df.d)]
        df[f"rest_{col}"] = [v[0] for v in vals]
        df[f"load_{col}"] = [v[1] for v in vals]

    n_missing = df[["rest_self", "rest_opp"]].isna().any(axis=1).sum()
    if n_missing:
        print(f"note {n_missing} rows have no prior fixture (season openers) -> set to the median")
    for c in ("rest_self", "rest_opp"):
        df[c] = df[c].fillna(df[c].median())

    # z-score, then map to the home-perspective names prepare_model_data reads
    for base in ("rest", "load"):
        v = pd.concat([df[f"{base}_self"], df[f"{base}_opp"]])
        mu, sd = v.mean(), v.std()
        for col, side in ((f"{base}_self", "home"), (f"{base}_opp", "away")):
            df[f"{base}_{side}"] = (df[col] - mu) / sd
        print(f"  {base}: mean {mu:.2f} sd {sd:.2f}")

    h = df[df.is_home == 1]
    print(f"\nhome rest z mean {h.rest_home.mean():+.3f}  away {h.rest_away.mean():+.3f}")
    print(f"corr(rest_home, load_home) = {h.rest_home.corr(h.load_home):+.3f} "
          f"-> tested SEPARATELY (BACKLOG caution 1)")
    shared["df_cv"] = df.drop(columns=["d", "rest_self", "rest_opp", "load_self", "load_opp"])
    OUT.write_bytes(pickle.dumps(shared))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
