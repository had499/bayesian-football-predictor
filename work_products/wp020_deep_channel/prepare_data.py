"""WP020 — join per-team DEEP COMPLETIONS onto the CV dataframe.

Understat `deep`: passes completed within ~20m of the opposing goal, excluding
crosses. The reason this is the one channel left worth testing after WP019: it is
NOT a function of the shot list. Every shot-feed decomposition WP019 tried
correlated >=0.93 with shot count or total xG, because they are all sums over the
same 116k shots. Deep completions measure possession worked into dangerous areas
that did NOT become a shot — structurally invisible to a goals+xG model.

    python work_products/wp020_deep_channel/prepare_data.py
"""
import pickle
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WP007 = HERE.parent / "wp007_structural_fixes"
OUT = HERE / "cv_shared_data_deep.pkl"


def main():
    shared = pickle.loads((WP007 / "cv_shared_data.pkl").read_bytes())
    df = shared["df_cv"].copy()
    ctx = pickle.loads((HERE / "team_context.pkl").read_bytes())
    s2t = pickle.loads((HERE / "short_to_title.pkl").read_bytes())

    ctx = ctx.assign(d=ctx.dt.dt.normalize())[["team", "d", "deep"]]
    assert not ctx.duplicated(subset=["team", "d"]).any(), "(team, date) must be unique"

    df["d"] = pd.to_datetime(df["datetime"]).dt.normalize()
    # df_cv carries Understat SHORT codes; team_context carries full titles
    df["_t"] = df["team"].map(lambda x: s2t.get(x, x))
    df["_o"] = df["opp_team"].map(lambda x: s2t.get(x, x))
    n0 = len(df)
    df = df.merge(ctx.rename(columns={"team": "_t", "deep": "deep_self"}), on=["_t", "d"], how="left")
    df = df.merge(ctx.rename(columns={"team": "_o", "deep": "deep_opp"}), on=["_o", "d"], how="left")
    assert len(df) == n0, f"join changed row count {n0} -> {len(df)}"

    miss = df[["deep_self", "deep_opp"]].isna().any(axis=1)
    if miss.any():
        print(f"WARNING {miss.sum()} of {n0} rows lack deep counts — filled with the league mean")
    df[["deep_self", "deep_opp"]] = df[["deep_self", "deep_opp"]].fillna(ctx.deep.mean())
    # df_obs keeps is_home==1 rows, where self=home and opp=away
    df["deep_home"], df["deep_away"] = df.deep_self, df.deep_opp

    h = df[df.is_home == 1]
    print(f"deep/match: mean {ctx.deep.mean():.2f}  SD {ctx.deep.std():.2f}  "
          f"var/mean {ctx.deep.var()/ctx.deep.mean():.2f}  (>1 => NegBinomial, not Poisson)")
    print(f"home {h.deep_home.mean():.2f} vs away {h.deep_away.mean():.2f}  "
          f"(home advantage should show here too)")
    shared["df_cv"] = df.drop(columns=["d", "_t", "_o", "deep_self", "deep_opp"])
    OUT.write_bytes(pickle.dumps(shared))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
