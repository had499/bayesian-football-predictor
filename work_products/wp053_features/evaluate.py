"""WP053 — can the comprehensive feature set predict the champion's residual?

253 features (WP053/build.py) vs the 96 of WP052b and the misspecified 16 of WP052.
Same protocol: leave-one-season-out, a 40-shuffle PERMUTATION NULL through the identical
pipeline, both a linear and a non-linear learner, and the market price held out as a
separate CEILING rather than mixed in.

Missing values are imputed with the TRAINING-FOLD median (leak-safe) instead of dropping
rows -- the 25-match trend features only reach 88% coverage, and an intersection drop
would quietly bias the sample toward mid-season.
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).resolve().parent
S = pickle.load((HERE / "features.pkl").open("rb"))
D, FE = S["D"], S["feats"]
CEIL = FE + ["mkt_own", "mkt_opp"]


def loso(cols, model, y=None, seed=0):
    y = D.resid.to_numpy() if y is None else y
    P, A, errs, base = [], [], [], []
    for s in sorted(D.season.unique()):
        tr = (D.season != s).to_numpy(); te = ~tr
        if tr.sum() < 500 or te.sum() < 100:
            continue
        Xtr, Xte = D.loc[tr, cols].to_numpy(float), D.loc[te, cols].to_numpy(float)
        med = np.nanmedian(Xtr, axis=0)                      # TRAIN-fold medians only
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        Xte = np.where(np.isfinite(Xte), Xte, med)
        mdl = model(); mdl.fit(Xtr, y[tr])
        p = mdl.predict(Xte)
        P.append(p); A.append(y[te])
        errs.append(((y[te] - p) ** 2).sum())
        base.append(((y[te] - y[tr].mean()) ** 2).sum())
    P, A = np.concatenate(P), np.concatenate(A)
    return 1 - sum(errs) / sum(base), np.corrcoef(P, A)[0, 1]


def main():
    print(f"{len(FE)} features, {len(D)} rows, {D.season.nunique()} seasons, "
          f"residual sd {D.resid.std():.3f}\n")
    rg = lambda: RidgeCV(alphas=np.logspace(-1, 5, 25))
    gb = lambda: HistGradientBoostingRegressor(max_depth=2, max_iter=150,
                                               learning_rate=0.03, l2_regularization=10.0,
                                               early_stopping=False, random_state=0)
    rng = np.random.default_rng(0)
    y0 = D.resid.to_numpy()
    null = np.array([loso(FE, rg, y=rng.permutation(y0))[0] for _ in range(40)])
    lo, hi = np.percentile(null, [2.5, 97.5])
    print(f"=== permutation null (40 shuffles, ridge, {len(FE)} features) ===")
    print(f"  mean {null.mean():+.5f}, 95% range [{lo:+.5f},{hi:+.5f}]\n")
    print(f"  {'specification':32s} {'learner':9s} {'OOS R^2':>10s} {'corr':>8s}  verdict")
    res = {}
    for lab, cols in (("OUR features (253)", FE), ("+ MARKET [CEILING]", CEIL)):
        for ln, mk in (("ridge", rg), ("boosting", gb)):
            r, c = loso(cols, mk)
            res[(lab, ln)] = r
            v = ("OUTSIDE null" if r > hi else "INSIDE null" if r >= lo else "below null")
            print(f"  {lab:32s} {ln:9s} {r:+10.5f} {c:+8.4f}  {v}")
    print(f"\n  p-value for OUR features (ridge): "
          f"{(null >= res[('OUR features (253)','ridge')]).mean():.3f}")
    print(f"\n  progression of the SAME test as the feature set improved:")
    print(f"    16 features (misspecified, no defence)  -0.00106   INSIDE null")
    print(f"    96 features (defence + 3 windows)       +0.00233   OUTSIDE null")
    print(f"   {len(FE)} features (this)                      "
          f"{res[('OUR features (253)','ridge')]:+.5f}   "
          f"{'OUTSIDE null' if res[('OUR features (253)','ridge')] > hi else 'INSIDE null'}")
    best_own = max(res[("OUR features (253)", l)] for l in ("ridge", "boosting"))
    best_ceil = max(res[("+ MARKET [CEILING]", l)] for l in ("ridge", "boosting"))
    print(f"\n  our features reach {best_own/best_ceil:.0%} of the market ceiling")
    print(f"  (the market's +0.0091 on the 96-feature run corresponded to -0.00083 RPS,")
    print(f"   so a composite worth {best_own:.5f} scales to very roughly "
          f"{-0.00083*best_own/0.00914:.5f} RPS -- an ESTIMATE, not a measurement)")


if __name__ == "__main__":
    main()
