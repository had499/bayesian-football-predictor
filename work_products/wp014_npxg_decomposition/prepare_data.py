"""Compute non-penalty xG per team-match from the shot cache and join it onto df_cv.

npxG = sum of shot xG excluding `situation == "Penalty"`. Penalties are high-xG
(~0.76 each) and poorly repeatable — winning them has weak season-to-season
persistence — so they inject noise into the xG measurement channel roughly in
proportion to their share of xG while carrying little signal about team strength.
npxG is the standard metric in football analytics for exactly this reason.

    python work_products/wp014_npxg_decomposition/prepare_data.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP007 = HERE.parent / "wp007_structural_fixes"
CACHE = HERE / "shots"
OUT_NPXG = HERE / "cv_shared_data_npxg.pkl"      # channel observes shot-derived NON-PENALTY xG
OUT_SHOT = HERE / "cv_shared_data_shotxg.pkl"     # channel observes shot-derived TOTAL xG (source control)


def team_match_npxg():
    """-> DataFrame(team, date, npxg, xg_total, pen_xg) — one row per team-match."""
    rows = []
    for f in sorted(CACHE.glob("*.pkl")):
        for mid, rec in pickle.loads(f.read_bytes()).items():
            meta = rec["meta"]
            date = pd.Timestamp(meta["datetime"]).normalize()
            for side, team in (("h", meta["home_short"]), ("a", meta["away_short"])):
                shots = rec[side]
                tot = sum(s["xG"] for s in shots)
                pen = sum(s["xG"] for s in shots if s["situation"] == "Penalty")
                rows.append({"team": team, "date": date, "npxg": tot - pen,
                             "xg_total": tot, "pen_xg": pen})
    return pd.DataFrame(rows)


def main():
    shared = pickle.loads((WP007 / "cv_shared_data.pkl").read_bytes())
    df = shared["df_cv"].copy()
    npx = team_match_npxg()
    assert not npx.duplicated(subset=["team", "date"]).any(), "(team, date) must be unique"

    print(f"shot cache: {len(npx)} team-matches, {npx.date.min().date()}..{npx.date.max().date()}")
    print(f"penalty share of total xG: {npx.pen_xg.sum()/npx.xg_total.sum():.1%}")
    print(f"matches with a penalty:    {(npx.pen_xg > 0).mean():.1%}")

    # IMPORTANT: Understat's match-level xG and its shot feed disagree — corr 0.997
    # but the shot-sum averages 1.448 vs the match aggregate's 1.425, with a tail of
    # ~1.0 discrepancies (not own goals; only ~32/season). Almost certainly the two
    # were computed at different xG-model versions. So comparing shot-derived npxG
    # against the match-level xG the model currently uses would confound "penalties
    # removed" with "different data source". Both variants below are therefore
    # derived from the SAME shot feed, and a total-xG arm acts as the source control:
    #   npxG vs shot-total  -> the penalty effect, source held constant
    #   shot-total vs match -> whether the source change matters at all
    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    n0 = len(df)
    for col, src in (("npxg", "npxg"), ("shotxg", "xg_total")):
        df = df.merge(npx.rename(columns={src: f"{col}_self"})[["team", "date", f"{col}_self"]],
                      on=["team", "date"], how="left")
        df = df.merge(npx.rename(columns={"team": "opp_team", src: f"{col}_opp"})[["opp_team", "date", f"{col}_opp"]],
                      on=["opp_team", "date"], how="left")
    assert len(df) == n0, f"join changed row count {n0} -> {len(df)}"

    miss = df[["npxg_self", "shotxg_self"]].isna().any(axis=1)
    if miss.any():
        print(f"WARNING {miss.sum()} rows lack shot data — falling back to match-level xG there")

    # invariant that actually holds: within the shot feed, npxG is a subset of total
    ok = df[["npxg_self", "shotxg_self"]].notna().all(axis=1)
    assert (df.loc[ok, "npxg_self"] <= df.loc[ok, "shotxg_self"] + 1e-6).all(), "npxG exceeds shot-total"

    print(f"mean match-level xG {df.xG_home.mean():.4f} | shot-total {df.shotxg_self.mean():.4f} "
          f"| npxG {df.npxg_self.mean():.4f}  ({1-df.npxg_self.mean()/df.shotxg_self.mean():.1%} is penalties)")

    for out, col in ((OUT_NPXG, "npxg"), (OUT_SHOT, "shotxg")):
        d2 = df.copy()
        # prepare_model_data looks for npxg_home/away; point them at whichever variant
        d2["npxg_home"] = d2[f"{col}_self"].fillna(d2.xG_home)
        d2["npxg_away"] = d2[f"{col}_opp"].fillna(d2.xG_away)
        shared["df_cv"] = d2.drop(columns=["date", "npxg_self", "npxg_opp", "shotxg_self", "shotxg_opp"])
        out.write_bytes(pickle.dumps(shared))
        print(f"wrote {out.name}: channel observes {col}")


if __name__ == "__main__":
    main()
