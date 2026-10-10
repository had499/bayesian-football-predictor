"""WP052 — can ANY combination of our features predict the champion's residual?

THE POINT. ~30 arms each tried to name one missing information source; almost all null.
The market channel then took 32% of the gap in one shot, because it imports an AGGREGATE
of everything rather than guessing which piece is missing. The natural question is
whether a combination of OUR OWN features could do the same.

Rather than build a composite and test it, this BOUNDS what every feature we have could
ever contribute: boost the champion. Predict its out-of-sample RESIDUAL (goals - lambda)
from our features with a flexible learner, leave-one-season-out. If the residual is
predictable, there is exploitable structure and a composite channel is worth building --
AND it would mean the one-at-a-time nulls were an aggregation failure, not an information
failure. If it is not predictable, no combination of these features helps, and both the
one-at-a-time and the bundle-plus-ablate strategies are closed at once.

FEATURES MUST BE OPEN-PRICE KNOWABLE, or the answer does not bear on our actual problem:
  * rolling (prior-only) xG, goals, deep, shots -- for the team AND its opponent
  * stakes (league table), rest / load (fixture calendar)
  * EXCLUDED: `intact` -- WP030 established it needs the match's own team sheet, which
    is unavailable at the opening price.
  * EXCLUDED from run A: the market price for THIS match. We already know the market
    beats us (WP040), so including it guarantees success and teaches nothing. It is run
    separately as a CEILING -- what a perfect external signal achieves against what our
    own features achieve.

A NOTE ON THE METRIC. Poisson noise dominates a single match, so even real structure
gives a tiny R^2. The benchmark is therefore not "is R^2 big" but "does it beat zero
out-of-sample, and how does it compare to the market ceiling".
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

K = 10


def build():
    d = pickle.loads((WP / "wp039_market_channel/cv_shared_data_market.pkl")
                     .read_bytes())["df_cv"]
    m, _ = load_checkpoints(WP / "wp032_xg_weight/checkpoints")
    m = attach_meta(m[m.arm == "W075"], d)

    # --- per-team prior-only rolling form ---
    long = pd.concat([
        d[d.is_home == 1].assign(t=lambda x: x.team, gf=lambda x: x.goals_home,
                                 xgf=lambda x: x.xG_home, deepf=lambda x: x.deep_home,
                                 shf=lambda x: x.shots_home),
        d[d.is_home == 1].assign(t=lambda x: x.opp_team, gf=lambda x: x.goals_away,
                                 xgf=lambda x: x.xG_away, deepf=lambda x: x.deep_away,
                                 shf=lambda x: x.shots_away),
    ])[["datetime", "t", "gf", "xgf", "deepf", "shf"]].sort_values("datetime")
    long["d"] = pd.to_datetime(long.datetime).dt.normalize()
    for c in ("gf", "xgf", "deepf", "shf"):
        long[f"r_{c}"] = (long.groupby("t")[c]
                          .transform(lambda x: x.shift(1).rolling(K, min_periods=5).mean()))
    form = long.set_index(["t", "d"])[[f"r_{c}" for c in ("gf", "xgf", "deepf", "shf")]]

    # --- pre-match covariates, keyed on the match ---
    mt = d[d.is_home == 1][["datetime", "team", "opp_team", "stakes_home", "stakes_away",
                            "rest_home", "rest_away", "load_home", "load_away",
                            "market_lh", "market_la", "market_obs_w"]].copy()
    mt["d"] = pd.to_datetime(mt.datetime).dt.normalize()
    cov = mt.set_index(["team", "opp_team", "d"])

    rows = []
    m["d"] = pd.to_datetime(m.date).dt.normalize()
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
                f_me, f_op = form.loc[(me, r.d)], form.loc[(opp, r.d)]
            except KeyError:
                continue
            rows.append(dict(
                window=r.window, season=r.season, resid=gl - lam, lam=lam,
                is_home=1.0 if side == "h" else 0.0,
                own_g=f_me.r_gf, own_xg=f_me.r_xgf, own_deep=f_me.r_deepf, own_sh=f_me.r_shf,
                opp_g=f_op.r_gf, opp_xg=f_op.r_xgf, opp_deep=f_op.r_deepf, opp_sh=f_op.r_shf,
                stakes_own=c.stakes_home if side == "h" else c.stakes_away,
                stakes_opp=c.stakes_away if side == "h" else c.stakes_home,
                rest_own=c.rest_home if side == "h" else c.rest_away,
                rest_opp=c.rest_away if side == "h" else c.rest_home,
                load_own=c.load_home if side == "h" else c.load_away,
                load_opp=c.load_away if side == "h" else c.load_home,
                mkt_own=(c.market_lh if side == "h" else c.market_la) if c.market_obs_w > 0 else np.nan,
                mkt_opp=(c.market_la if side == "h" else c.market_lh) if c.market_obs_w > 0 else np.nan,
            ))
    return pd.DataFrame(rows)


OURS = ["is_home", "own_g", "own_xg", "own_deep", "own_sh",
        "opp_g", "opp_xg", "opp_deep", "opp_sh",
        "stakes_own", "stakes_opp", "rest_own", "rest_opp", "load_own", "load_opp", "lam"]
CEIL = OURS + ["mkt_own", "mkt_opp"]


def loso_r2(d, cols, model):
    errs, base = [], []
    for s in sorted(d.season.unique()):
        tr, te = d[d.season != s], d[d.season == s]
        if len(tr) < 500 or len(te) < 100:
            continue
        mdl = model()
        mdl.fit(tr[cols].to_numpy(), tr.resid.to_numpy())
        p = mdl.predict(te[cols].to_numpy())
        errs.append(((te.resid.to_numpy() - p) ** 2).sum())
        base.append(((te.resid.to_numpy() - tr.resid.mean()) ** 2).sum())
    return 1 - sum(errs) / sum(base)


def main():
    D = build()
    print(f"rows {len(D)}, seasons {D.season.nunique()}, "
          f"residual sd {D.resid.std():.3f}\n")
    A = D.dropna(subset=OURS).copy()
    B = D.dropna(subset=CEIL).copy()
    print(f"run A (our features only):  n={len(A)}")
    print(f"run B (+ market ceiling):   n={len(B)}\n")

    gb = lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                               learning_rate=0.05, l2_regularization=1.0,
                                               random_state=0)
    rg = lambda: RidgeCV(alphas=np.logspace(-2, 3, 20))

    print("=== can the champion's RESIDUAL be predicted out-of-sample? ===")
    print("    OOS R^2 > 0 means exploitable structure remains in these features.\n")
    print(f"  {'features':34s} {'learner':10s} {'OOS R^2':>10s}")
    for lab, dd, cols in (("OUR features only", A, OURS),
                          ("OUR features only", A, OURS)):
        pass
    for learner_name, learner in (("ridge", rg), ("boosting", gb)):
        r = loso_r2(A, OURS, learner)
        print(f"  {'OUR features only':34s} {learner_name:10s} {r:+10.5f}")
    for learner_name, learner in (("ridge", rg), ("boosting", gb)):
        r = loso_r2(B, CEIL, learner)
        print(f"  {'+ MARKET price [CEILING]':34s} {learner_name:10s} {r:+10.5f}")
    # a pure-noise control: shuffled target
    An = A.copy()
    rng = np.random.default_rng(0)
    An["resid"] = rng.permutation(An.resid.to_numpy())
    print(f"  {'SHUFFLED target [control]':34s} {'boosting':10s} "
          f"{loso_r2(An, OURS, gb):+10.5f}")
    print("\n  the shuffled control shows what zero signal looks like through this")
    print("  pipeline; if OUR features land there and the market ceiling does not,")
    print("  the information we are missing is simply not in anything we have built.")




def rigorous():
    """Strengthened version. The first pass reported POINT R^2 only, with ONE shuffle as
    the control and UNTUNED boosting whose R^2 was negative on both real and shuffled
    data -- i.e. comparing two overfit models. Fixed here:

      * a PERMUTATION NULL: 40 shuffles give a distribution of OOS R^2 under zero
        signal, so "our features land on the control" becomes a quantified claim.
      * REGULARISATION SWEEP for boosting, so the answer is not an artefact of one
        hyperparameter choice.
      * a SECOND METRIC: out-of-sample correlation between predicted and actual
        residual, which does not depend on the R^2 denominator.
    """
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV

    D = build()
    A = D.dropna(subset=OURS).copy()
    B = D.dropna(subset=CEIL).copy()

    def loso_both(d, cols, model, target=None):
        """returns (pooled OOS R^2, OOS corr(pred, actual))"""
        y = d.resid.to_numpy() if target is None else target
        preds, acts, errs, base = [], [], [], []
        for s in sorted(d.season.unique()):
            tr = (d.season != s).to_numpy(); te = ~tr
            if tr.sum() < 500 or te.sum() < 100:
                continue
            mdl = model()
            mdl.fit(d.loc[tr, cols].to_numpy(), y[tr])
            p = mdl.predict(d.loc[te, cols].to_numpy())
            preds.append(p); acts.append(y[te])
            errs.append(((y[te] - p) ** 2).sum())
            base.append(((y[te] - y[tr].mean()) ** 2).sum())
        P, Aa = np.concatenate(preds), np.concatenate(acts)
        return 1 - sum(errs) / sum(base), np.corrcoef(P, Aa)[0, 1]

    rg = lambda: RidgeCV(alphas=np.logspace(-2, 4, 25))
    print("\n\n=== STRENGTHENED: permutation null, 40 shuffles ===")
    rng = np.random.default_rng(0)
    null = []
    for i in range(40):
        An = A.copy()
        An["resid"] = rng.permutation(A.resid.to_numpy())
        null.append(loso_both(An, OURS, rg)[0])
    null = np.array(null)
    r_ours, c_ours = loso_both(A, OURS, rg)
    r_ceil, c_ceil = loso_both(B, CEIL, rg)
    lo, hi = np.percentile(null, [2.5, 97.5])
    print(f"  permutation null (ridge): mean {null.mean():+.5f}, "
          f"95% range [{lo:+.5f},{hi:+.5f}]")
    print(f"  OUR features             : R^2 {r_ours:+.5f}  corr {c_ours:+.4f}   "
          f"-> {'INSIDE the null' if lo <= r_ours <= hi else 'OUTSIDE the null'}")
    print(f"  + MARKET price [CEILING] : R^2 {r_ceil:+.5f}  corr {c_ceil:+.4f}   "
          f"-> {'INSIDE the null' if lo <= r_ceil <= hi else 'OUTSIDE the null'}")
    print(f"  permutation p-value for OUR features: "
          f"{(null >= r_ours).mean():.3f}  (fraction of shuffles scoring at least as well)")

    print("\n=== boosting REGULARISATION sweep (was one untuned setting) ===")
    print(f"  {'depth':>6s} {'iters':>6s} {'lr':>6s} {'l2':>6s} {'ours R^2':>10s} {'ceiling R^2':>12s}")
    for depth, iters, lr, l2 in ((2, 100, 0.03, 10.0), (3, 200, 0.05, 1.0),
                                 (2, 400, 0.02, 30.0), (4, 300, 0.05, 0.1)):
        mk = lambda d=depth, i=iters, l=lr, r=l2: HistGradientBoostingRegressor(
            max_depth=d, max_iter=i, learning_rate=l, l2_regularization=r,
            early_stopping=False, random_state=0)
        ro, _ = loso_both(A, OURS, mk)
        rc, _ = loso_both(B, CEIL, mk)
        print(f"  {depth:6d} {iters:6d} {lr:6.2f} {l2:6.1f} {ro:+10.5f} {rc:+12.5f}")
    print("\n  if OUR features stay inside the permutation null at EVERY regularisation")
    print("  while the ceiling stays outside it, the conclusion is not an artefact.")


if __name__ == "__main__":
    main()
    rigorous()
