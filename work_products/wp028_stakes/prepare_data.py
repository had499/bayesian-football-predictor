"""WP028 step 2 — attach end-of-season stakes to the CV dataframe.

JOIN KEY IS (team, DATE), not (team, round). `cum_round` is an ISO-week bucket, so a
team playing twice in one week yields TWO stakes rows -- keying on round silently fans
the join out (1,170 duplicate keys here; WP009 hit the same trap and its join went
1205 -> 2285 rows). And each match's date must come from (round, home, away), NOT from
the round's first date: taking the round's date kept only the Saturday-3pm subset, 38%
of matches and a biased one, since TV picks are the bigger clubs.

    python work_products/wp028_stakes/prepare_data.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(HERE))
OUT = HERE / "cv_shared_data_stakes.pkl"
COLS = ["stakes_relegation", "stakes_europe", "stakes_title"]


def main():
    shared = pickle.loads((WP / "wp021_power_confirmation/cv_shared_data_big.pkl").read_bytes())
    df = shared["df_cv"].copy()
    S = pd.read_pickle(HERE / "stakes_table.pkl")
    assert not S.duplicated(["date", "team"]).any(), "(date, team) must be unique"
    S["total"] = S[COLS].sum(axis=1)

    df["d"] = pd.to_datetime(df.datetime).dt.normalize()
    key = S.set_index(["date", "team"])["total"]
    n0 = len(df)
    df["stakes_self"] = [key.get((d, t), np.nan) for d, t in zip(df.d, df.team)]
    df["stakes_opp"] = [key.get((d, t), np.nan) for d, t in zip(df.d, df.opp_team)]
    assert len(df) == n0
    miss = df[["stakes_self", "stakes_opp"]].isna().any(axis=1).sum()
    print(f"{n0} rows; {miss} missing stakes -> filled with 0 (no race to be in)")
    df[["stakes_self", "stakes_opp"]] = df[["stakes_self", "stakes_opp"]].fillna(0.0)

    # home-perspective names prepare_model_data reads. Raw scale (0..~1) is kept
    # rather than z-scored so the effort term reads as 1 + g*stakes, i.e. g is the
    # multiplier at maximal stakes and stakes=0 means "behaves normally".
    df["stakes_home"] = df.stakes_self
    df["stakes_away"] = df.stakes_opp
    df["stakes_match"] = np.minimum(df.stakes_self, df.stakes_opp)

    h = df[df.is_home == 1]
    print(f"stakes_home mean {h.stakes_home.mean():.4f} max {h.stakes_home.max():.4f}")
    print(f"stakes_match mean {h.stakes_match.mean():.4f}  "
          f"dead rubbers (<0.02): {(h.stakes_match < 0.02).mean()*100:.1f}% of matches")
    shared["df_cv"] = df.drop(columns=["d", "stakes_self", "stakes_opp"])
    OUT.write_bytes(pickle.dumps(shared))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
