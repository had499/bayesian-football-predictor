"""WP039 — build the CV data file with market-implied goal rates per side.

Pinnacle CLOSING 1X2 -> Shin de-vig -> invert a Poisson scoreline model -> (lh, la).
Two free probabilities, two rates: exactly identified.

LEAK SAFETY. These are per-match OBSERVATIONS, exactly like xG_obs_*. The channel is a
pm.Potential that fires on training rows only; predicting round R therefore reads prices
from rounds < R and never the priced match's own line. `market_obs_w` is 0 where no price
exists so the channel simply does not fire rather than needing imputation.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp008_market_benchmark"))
from market_analysis import CODE_TO_FD, add_devig_probs  # noqa: E402

SRC = WP / "wp030_squad_intactness" / "cv_shared_data_intact.pkl"
OUT = HERE / "cv_shared_data_market.pkl"
PIN_CLOSE = ["pinnacle_1x2_home_close", "pinnacle_1x2_draw_close", "pinnacle_1x2_away_close"]
MAXG = 10


def poisson_probs(lh, la):
    g = np.outer(poisson.pmf(np.arange(MAXG + 1), lh),
                 poisson.pmf(np.arange(MAXG + 1), la))
    return np.array([np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()])


def invert(p):
    r = least_squares(lambda x: poisson_probs(np.exp(x[0]), np.exp(x[1])) - p,
                      x0=np.log([1.5, 1.2]), method="lm")
    return np.exp(r.x[0]), np.exp(r.x[1])


def main():
    d = pickle.loads(SRC.read_bytes())
    df = d["df_cv"].copy()
    odds = pd.read_pickle(WP / "wp008_market_benchmark" / "odds_raw.pkl")

    mt = df[df.is_home == 1][["datetime", "team", "opp_team",
                              "goals_home", "goals_away"]].copy()
    mt["date"] = pd.to_datetime(mt.datetime).dt.normalize()
    mt["home_fd"] = mt.team.map(CODE_TO_FD)
    mt["away_fd"] = mt.opp_team.map(CODE_TO_FD)
    assert mt.home_fd.notna().all() and mt.away_fd.notna().all(), "CODE_TO_FD gap"

    o = odds.copy()
    o["date"] = pd.to_datetime(o.date).dt.normalize()
    j = mt.merge(o[["date", "home_team", "away_team", "fthg", "ftag"] + PIN_CLOSE],
                 left_on=["date", "home_fd", "away_fd"],
                 right_on=["date", "home_team", "away_team"], how="left")
    # a score disagreement means the join found the WRONG fixture (rescheduled etc) --
    # drop the price rather than attach it to the wrong match
    bad = j.fthg.notna() & ((j.goals_home != j.fthg) | (j.goals_away != j.ftag))
    print(f"matches {len(mt)}, priced {j[PIN_CLOSE].notna().all(1).sum()}, "
          f"score-mismatched {bad.sum()}")
    j.loc[bad, PIN_CLOSE] = np.nan
    j = add_devig_probs(j, PIN_CLOSE, "pinc")

    P = j[["pinc_shin_h", "pinc_shin_d", "pinc_shin_a"]].to_numpy()
    good = np.isfinite(P).all(1) & (P.sum(1) > 0.99) & (P.sum(1) < 1.01)
    lh = np.zeros(len(j)); la = np.zeros(len(j)); w = np.zeros(len(j))
    for i in np.where(good)[0]:
        lh[i], la[i] = invert(P[i])
        w[i] = 1.0
    # guard against a pathological solve -- a rate outside this band is not football
    sane = (lh > 0.2) & (lh < 5.0) & (la > 0.2) & (la < 5.0)
    w = w * sane
    lh = np.where(w > 0, lh, 1.0)      # value is irrelevant where weight is 0, but
    la = np.where(w > 0, la, 1.0)      # must stay finite for the Gamma logp
    print(f"inverted {int(w.sum())}/{len(j)} ({w.mean():.1%})")
    print(f"  implied mean rates: home {lh[w>0].mean():.3f}, away {la[w>0].mean():.3f}")
    print(f"  model's own means:  home 1.571, away 1.264  "
          f"(Poisson-independence inversion compresses both to fit the draw price)")

    mt2 = j[["team", "opp_team", "date"]].copy()
    mt2["market_lh"], mt2["market_la"], mt2["market_obs_w"] = lh, la, w

    # attach to BOTH perspectives of each match in the long panel, keyed on (team,date)
    df["date"] = pd.to_datetime(df.datetime).dt.normalize()
    key = mt2.set_index(["team", "date"])[["market_lh", "market_la", "market_obs_w"]]
    idx = pd.MultiIndex.from_arrays([df.team.where(df.is_home == 1, df.opp_team),
                                     df.date])
    got = key.reindex(idx)
    for c in ("market_lh", "market_la", "market_obs_w"):
        df[c] = got[c].to_numpy()
    df[["market_lh", "market_la"]] = df[["market_lh", "market_la"]].fillna(1.0)
    df["market_obs_w"] = df.market_obs_w.fillna(0.0)
    df = df.drop(columns="date")
    assert df.market_obs_w.notna().all()
    print(f"\npanel rows with a usable price: {df.market_obs_w.mean():.1%}")

    d["df_cv"] = df
    OUT.write_bytes(pickle.dumps(d))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
