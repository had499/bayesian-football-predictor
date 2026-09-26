"""Join WP011's leak-safe squad features onto the CV dataframe so the model
can train on them.

WP011 only *correlated* absence against finished predictions. That misses the
channel where the feature changes how the latent team strengths are inferred
during training, so this work product puts it in the model properly. The
feature travels with `df_cv` (rather than being loaded separately inside
prepare_model_data) so nothing downstream has to know where it came from.

    python work_products/wp012_absence_covariate/prepare_data.py
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WP007 = HERE.parent / "wp007_structural_fixes"
WP011 = HERE.parent / "wp011_squad_instability"
sys.path.insert(0, str(WP011))
from features import load_rosters, team_match_features  # noqa: E402

OUT = HERE / "cv_shared_data_absence.pkl"
FEATURE = "absent_drop"


def main():
    shared = pickle.loads((WP007 / "cv_shared_data.pkl").read_bytes())
    df = shared["df_cv"].copy()

    feats = team_match_features(load_rosters())
    feats["date"] = feats.datetime.dt.normalize()
    lookup = feats[["team", "date", FEATURE]].dropna()
    assert not lookup.duplicated(subset=["team", "date"]).any(), "(team, date) must be unique"

    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    n0 = len(df)
    # own side's absence, and the opponent's
    df = df.merge(lookup.rename(columns={"team": "team", FEATURE: "absent_self"}),
                  on=["team", "date"], how="left")
    df = df.merge(lookup.rename(columns={"team": "opp_team", FEATURE: "absent_opp"}),
                  on=["opp_team", "date"], how="left")
    assert len(df) == n0, f"join changed row count {n0} -> {len(df)}"

    # Rows before the roster data starts (pre-2019) have no feature — zero them.
    # Zero is the right neutral value: the model term is beta*absence, so 0 means
    # "no absence effect", and the intercept absorbs the overall mean anyway.
    cov = df[["absent_self", "absent_opp"]].notna().all(axis=1)
    print(f"{cov.sum()}/{len(df)} rows have squad features "
          f"({df.loc[cov, 'season'].min()}..{df.loc[cov, 'season'].max()}); rest zero-filled")
    df[["absent_self", "absent_opp"]] = df[["absent_self", "absent_opp"]].fillna(0.0)

    shared["df_cv"] = df.drop(columns="date")
    OUT.write_bytes(pickle.dumps(shared))
    print(f"wrote {OUT}  ({len(df)} rows, {len(shared['windows'])} windows)")


if __name__ == "__main__":
    main()
