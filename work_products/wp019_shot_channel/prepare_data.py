"""Join per-team shot COUNTS onto the CV dataframe so the model can observe them.

WP019's premise (measured in WP018's follow-up): predicting second-half goals from
first-half data, the coefficient on log(shot volume) is +1.028 and on log(shot
quality) +0.554 — a difference of +0.474, CI [+0.132, +0.771]. Total xG forces
those to be EQUAL because it only sees their product, and freeing them raises R^2
from 0.4674 to 0.4866 (gain CI [+0.0017, +0.0503]).

Volume is also the more repeatable skill (first->second half r = +0.770 vs +0.567
for quality), so a channel that observes shot count gives the attack latent a
lower-noise indicator than total xG alone.

    python work_products/wp019_shot_channel/prepare_data.py
"""
import pickle
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WP007 = HERE.parent / "wp007_structural_fixes"
SHOTS = HERE.parent / "wp014_npxg_decomposition" / "shots"
OUT = HERE / "cv_shared_data_shots.pkl"


def team_match_shots():
    rows = []
    for f in sorted(SHOTS.glob("*.pkl")):
        for rec in pickle.loads(f.read_bytes()).values():
            m = rec["meta"]
            date = pd.Timestamp(m["datetime"]).normalize()
            for side, team in (("h", m["home_short"]), ("a", m["away_short"])):
                rows.append({"team": team, "date": date, "n_shots": len(rec[side])})
    return pd.DataFrame(rows)


def main():
    shared = pickle.loads((WP007 / "cv_shared_data.pkl").read_bytes())
    df = shared["df_cv"].copy()
    sh = team_match_shots()
    assert not sh.duplicated(subset=["team", "date"]).any(), "(team, date) must be unique"

    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    n0 = len(df)
    df = df.merge(sh.rename(columns={"n_shots": "shots_self"}), on=["team", "date"], how="left")
    df = df.merge(sh.rename(columns={"team": "opp_team", "n_shots": "shots_opp"}),
                  on=["opp_team", "date"], how="left")
    assert len(df) == n0, f"join changed row count {n0} -> {len(df)}"

    miss = df[["shots_self", "shots_opp"]].isna().any(axis=1)
    if miss.any():
        print(f"WARNING {miss.sum()} rows lack shot counts — filled with the league mean")
    df[["shots_self", "shots_opp"]] = df[["shots_self", "shots_opp"]].fillna(sh.n_shots.mean())
    # df_obs keeps is_home==1 rows, where self=home and opp=away
    df["shots_home"] = df.shots_self
    df["shots_away"] = df.shots_opp

    print(f"shots/match: mean {sh.n_shots.mean():.2f}  SD {sh.n_shots.std():.2f}  "
          f"var/mean {sh.n_shots.var()/sh.n_shots.mean():.2f}  (>1 => needs NegBinomial, not Poisson)")
    shared["df_cv"] = df.drop(columns=["date", "shots_self", "shots_opp"])
    OUT.write_bytes(pickle.dumps(shared))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
