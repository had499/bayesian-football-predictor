"""WP052b — the same bound, with a feature set that is not misspecified.

premise.py had two real flaws, both found by inspection rather than by the result:

  1. ONE window length (rolling 10). No short-vs-long contrast, when the DIFFERENCE
     between recent and season-long form is precisely what "form" means.
  2. NO DEFENSIVE records at all. It gave the learner the opponent's ATTACKING form
     (goals/xG scored) but never what anyone CONCEDES -- and to predict how many goals
     a team scores, the relevant opponent feature is its defensive record. That is a
     misspecification, not a missing nicety.

Also added: volatility (rolling sd), a short-minus-long trend term, and home/away-split
form. If the earlier "our features are noise" verdict was an artefact of a thin feature
set, a properly specified one should break it.

Same permutation null (40 shuffles), same market CEILING, same leave-one-season-out.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import attach_meta, load_checkpoints  # noqa: E402

WINDOWS = (3, 10, 25)
BASE = ("gf", "ga", "xgf", "xga", "deepf", "deepa", "shf", "sha")


def build():
    d = pickle.loads((WP / "wp039_market_channel/cv_shared_data_market.pkl")
                     .read_bytes())["df_cv"]
    m, _ = load_checkpoints(WP / "wp032_xg_weight/checkpoints")
    m = attach_meta(m[m.arm == "W075"], d)
    h = d[d.is_home == 1]
    # BOTH directions for every quantity: scored (f) AND conceded (a)
    long = pd.concat([
        pd.DataFrame(dict(datetime=h.datetime, t=h.team, home=1.0,
                          gf=h.goals_home, ga=h.goals_away,
                          xgf=h.xG_home, xga=h.xG_away,
                          deepf=h.deep_home, deepa=h.deep_away,
                          shf=h.shots_home, sha=h.shots_away)),
        pd.DataFrame(dict(datetime=h.datetime, t=h.opp_team, home=0.0,
                          gf=h.goals_away, ga=h.goals_home,
                          xgf=h.xG_away, xga=h.xG_home,
                          deepf=h.deep_away, deepa=h.deep_home,
                          shf=h.shots_away, sha=h.shots_home)),
    ]).sort_values("datetime")
    long["d"] = pd.to_datetime(long.datetime).dt.normalize()

    g = long.groupby("t")
    cols = []
    for c in BASE:
        for K in WINDOWS:
            long[f"{c}_m{K}"] = g[c].transform(
                lambda x, K=K: x.shift(1).rolling(K, min_periods=max(3, K // 3)).mean())
            cols.append(f"{c}_m{K}")
        # volatility and a short-minus-long TREND term
        long[f"{c}_sd10"] = g[c].transform(
            lambda x: x.shift(1).rolling(10, min_periods=5).std())
        long[f"{c}_trend"] = long[f"{c}_m3"] - long[f"{c}_m25"]
        cols += [f"{c}_sd10", f"{c}_trend"]
    # home/away-split form for the two headline quantities
    for c in ("gf", "xgf"):
        for hv, tag in ((1.0, "H"), (0.0, "A")):
            s = long[c].where(long.home == hv)
            long[f"{c}_{tag}8"] = s.groupby(long.t).transform(
                lambda x: x.shift(1).rolling(8, min_periods=3).mean())
            cols.append(f"{c}_{tag}8")
    form = long.set_index(["t", "d"])[cols]

    mt = h[["datetime", "team", "opp_team", "stakes_home", "stakes_away",
            "rest_home", "rest_away", "load_home", "load_away",
            "market_lh", "market_la", "market_obs_w"]].copy()
    mt["d"] = pd.to_datetime(mt.datetime).dt.normalize()
    cov = mt.set_index(["team", "opp_team", "d"])

    m["d"] = pd.to_datetime(m.date).dt.normalize()
    rows = []
    for _, r in m.iterrows():
        try:
            c = cov.loc[(r.home, r.away, r.d)]
        except KeyError:
            continue
        for side in ("h", "a"):
            me, opp = (r.home, r.away) if side == "h" else (r.away, r.home)
            gl = r.goals_home if side == "h" else r.goals_away
            lam = r.lambda_home if side == "h" else r.lambda_away
            try:
                fm, fo = form.loc[(me, r.d)], form.loc[(opp, r.d)]
            except KeyError:
                continue
            rec = dict(window=r.window, season=r.season, resid=gl - lam, lam=lam,
                       is_home=1.0 if side == "h" else 0.0,
                       stakes_own=c.stakes_home if side == "h" else c.stakes_away,
                       stakes_opp=c.stakes_away if side == "h" else c.stakes_home,
                       rest_own=c.rest_home if side == "h" else c.rest_away,
                       rest_opp=c.rest_away if side == "h" else c.rest_home,
                       load_own=c.load_home if side == "h" else c.load_away,
                       load_opp=c.load_away if side == "h" else c.load_home,
                       mkt_own=(c.market_lh if side == "h" else c.market_la)
                               if c.market_obs_w > 0 else np.nan,
                       mkt_opp=(c.market_la if side == "h" else c.market_lh)
                               if c.market_obs_w > 0 else np.nan)
            for k in cols:
                rec[f"own_{k}"] = fm[k]
                rec[f"opp_{k}"] = fo[k]
            rows.append(rec)
    D = pd.DataFrame(rows)
    feats = [c for c in D.columns
             if c not in ("window", "season", "resid", "mkt_own", "mkt_opp")]
    return D, feats


def main():
    D, OURS = build()
    CEIL = OURS + ["mkt_own", "mkt_opp"]
    A = D.dropna(subset=OURS).copy()
    B = D.dropna(subset=CEIL).copy()
    print(f"features: {len(OURS)} (was 16)  |  windows {WINDOWS}  |  "
          f"quantities {len(BASE)} incl. CONCEDED")
    print(f"rows: ours {len(A)}, ceiling {len(B)}\n")

    def loso(d, cols, model, y=None):
        y = d.resid.to_numpy() if y is None else y
        P, Aa, errs, base = [], [], [], []
        for s in sorted(d.season.unique()):
            tr = (d.season != s).to_numpy(); te = ~tr
            if tr.sum() < 500 or te.sum() < 100:
                continue
            mdl = model(); mdl.fit(d.loc[tr, cols].to_numpy(), y[tr])
            p = mdl.predict(d.loc[te, cols].to_numpy())
            P.append(p); Aa.append(y[te])
            errs.append(((y[te] - p) ** 2).sum())
            base.append(((y[te] - y[tr].mean()) ** 2).sum())
        P, Aa = np.concatenate(P), np.concatenate(Aa)
        return 1 - sum(errs) / sum(base), np.corrcoef(P, Aa)[0, 1]

    rg = lambda: RidgeCV(alphas=np.logspace(-2, 4, 25))
    gb = lambda: HistGradientBoostingRegressor(max_depth=2, max_iter=150,
                                               learning_rate=0.03, l2_regularization=10.0,
                                               early_stopping=False, random_state=0)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(40):
        An = A.copy(); An["resid"] = rng.permutation(A.resid.to_numpy())
        null.append(loso(An, OURS, rg)[0])
    null = np.array(null); lo, hi = np.percentile(null, [2.5, 97.5])
    print("=== permutation null (40 shuffles, ridge) ===")
    print(f"  mean {null.mean():+.5f}, 95% range [{lo:+.5f},{hi:+.5f}]\n")
    print(f"  {'specification':34s} {'learner':9s} {'OOS R^2':>10s} {'corr':>8s}  verdict")
    for lab, dd, cols in (("OUR features (rich)", A, OURS),
                          ("+ MARKET price [CEILING]", B, CEIL)):
        for ln, mk in (("ridge", rg), ("boosting", gb)):
            r, cc = loso(dd, cols, mk)
            v = "INSIDE null" if lo <= r <= hi else ("OUTSIDE null" if r > hi else "below null")
            print(f"  {lab:34s} {ln:9s} {r:+10.5f} {cc:+8.4f}  {v}")
    print(f"\n  p-value for OUR features (ridge): "
          f"{(null >= loso(A, OURS, rg)[0]).mean():.3f}")


if __name__ == "__main__":
    main()
