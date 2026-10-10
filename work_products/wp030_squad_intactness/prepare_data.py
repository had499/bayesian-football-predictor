"""WP030 step 2 — squad intactness per team-match, attached to the CV data.

THE MECHANISM. `intact` says what share of a team's normal starting strength actually
took the field. It enters theta as b*(intact - 1), which is ZERO at full strength:

  * TRAINING rows  -> the observed value from that match's real lineup. A depleted
                      team's poor result is explained by the depletion rather than
                      dragging its attack latent down.
  * PREDICTED row  -> assume a normal XI, intact = 1, the term vanishes. NO lineup
                      information about the match being priced is used, so the model
                      stays usable against OPENING prices.

This is the `returning players` mechanism the BACKLOG names: "a star back from three
months out has decayed out of expected minutes, so the model still prices the team as
weakened." Discounting the depleted stretch is how you stop carrying that penalty.

WHY IT IS NOT WP012 AGAIN, despite the same algebra:
  * weighted by MINUTES, not xG+xA -- which scores keepers at 0.003 against 0.193 for
    outfielders, so WP011/WP012 only ever tested ATTACKING absence (WP016's point).
  * it EXPLAINS past lineups (observed, exact) instead of FORECASTING today's (hard;
    WP012's threshold "fired 4 matches late").

HALF-LIFE 15, not the 3 that maximised the downstream correlation. At 3 only 31% of a
player's record survives five missed matches, so the measure forgets the very absences
it exists to detect. 15 retains 79% and costs little signal (r +0.117 vs +0.139).

    python work_products/wp030_squad_intactness/prepare_data.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(HERE))
from importance import build_minutes, ewma_importance, expand_squad  # noqa: E402

OUT = HERE / "cv_shared_data_intact.pkl"
HALF_LIFE = 15


def intactness(half_life=HALF_LIFE, norm_half_life=15):
    """intact = this match's importance-weighted XI, relative to that team's OWN norm.

    BUG FIXED: dividing by the summed importance of the whole SQUAD gave a mean of
    0.63 with 83% of matches "depleted". Sum of importance over a squad is ~11 by
    construction (11 players x 90 minutes each match), but importance is spread over
    ~20 rotating players, so the ~11 who actually play can never reach that total. It
    was measuring squad DEPTH, not depletion -- the same error WP011 made when
    `absent_share` turned out to be "measuring squad dispersion not absence".

    The fix is self-normalising: compare each match's weighted XI against an EWMA of
    the team's own PRIOR matches (leak-free). 1.0 then means "as intact as we usually
    are", which is the quantity the mechanism actually needs.
    """
    d = expand_squad(build_minutes()).reset_index(drop=True)
    d["imp"] = ewma_importance(d, half_life)
    g = d.dropna(subset=["imp"]).copy()
    g["w"] = g.imp * (g.minutes / 90.0)
    raw = g.groupby(["team", "date"]).w.sum().rename("xi").reset_index().sort_values("date")

    lam = 0.5 ** (1.0 / norm_half_life)
    out = []
    for team, tg in raw.groupby("team", sort=False):
        tg = tg.sort_values("date").copy()
        num = den = 0.0
        norms = []
        for v in tg.xi.to_numpy():
            norms.append((num / den) if den > 0 else np.nan)   # BEFORE this match
            num = num * lam + v
            den = den * lam + 1.0
        tg["norm"] = norms
        out.append(tg)
    o = pd.concat(out, ignore_index=True)
    o["intact"] = o.xi / o["norm"]
    o = o.dropna(subset=["intact"])
    # a wild ratio means the norm is barely established (first matches of a spell)
    o["intact"] = o.intact.clip(0.5, 1.5)
    assert not o.duplicated(["team", "date"]).any()
    return o[["team", "date", "intact"]]


def main():
    shared = pickle.loads((WP / "wp029_congestion_effort/cv_shared_data_both.pkl").read_bytes())
    df = shared["df_cv"].copy()
    it = intactness()
    print(f"intactness: {len(it)} team-matches, mean {it.intact.mean():.4f}, "
          f"p5 {it.intact.quantile(.05):.4f}, p95 {it.intact.quantile(.95):.4f}")

    df["d"] = pd.to_datetime(df.datetime).dt.normalize()
    key = it.set_index(["team", "date"])["intact"]
    n0 = len(df)
    df["intact_self"] = [key.get((t, d), np.nan) for t, d in zip(df.team, df.d)]
    df["intact_opp"] = [key.get((t, d), np.nan) for t, d in zip(df.opp_team, df.d)]
    assert len(df) == n0
    miss = df[["intact_self", "intact_opp"]].isna().any(axis=1).sum()
    print(f"{miss} of {n0} rows lack roster data -> set to 1.0 (full strength), so the "
          f"term contributes nothing rather than inventing a depletion")
    df[["intact_self", "intact_opp"]] = df[["intact_self", "intact_opp"]].fillna(1.0)
    df["intact_home"] = df.intact_self
    df["intact_away"] = df.intact_opp

    h = df[df.is_home == 1]
    print(f"\nhome intact mean {h.intact_home.mean():.4f}  away {h.intact_away.mean():.4f}")
    print(f"share of team-matches below 0.9 intact: "
          f"{(pd.concat([h.intact_home, h.intact_away]) < 0.9).mean()*100:.1f}%")
    print(f"corr(intact_home, stakes_home) = {h.intact_home.corr(h.stakes_home):+.3f}")
    print(f"corr(intact_home, rest_home)   = {h.intact_home.corr(h.rest_home):+.3f}")
    shared["df_cv"] = df.drop(columns=["d", "intact_self", "intact_opp"])
    OUT.write_bytes(pickle.dumps(shared))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
