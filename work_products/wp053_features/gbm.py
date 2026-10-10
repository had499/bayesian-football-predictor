"""WP053b — a PROPERLY SPECIFIED, PROPERLY TUNED GBM on the champion.

Two things were wrong with the first GBM pass:

  1. WRONG OBJECTIVE. It regressed the additive residual (goals - lambda) with SQUARED
     loss. Goals are counts and the model is multiplicative, so the correct formulation
     is POISSON with log(lambda) as an OFFSET -- i.e. learn a multiplicative correction
     m_i with lambda_adj = lambda * m. sklearn has no init_score, but the standard offset
     trick is exact: fit Poisson loss on target `goals/lambda` with sample_weight
     `lambda`, which has the same optimum as offset Poisson.
  2. UNDER-CAPACITY AND UNTUNED. depth 2, 150 trees, lr 0.03, l2 10 -- fixed by hand for
     253 features, and never searched.

HONEST TUNING. Hyperparameters are searched on windows 26-70 and then LOCKED before
touching windows 71-115, so the configuration choice cannot leak into the reported
result. Both halves use WALK-FORWARD training (window w sees only windows < w), because
WP053 measured leave-one-season-out leaking the future for a second-stage learner and
halving the apparent signal (+0.060 -> +0.030).

(LightGBM would be preferable but needs libomp, and there is no brew on this machine.)
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import paired_diff  # noqa: E402

S = pickle.load((HERE / "features.pkl").open("rb"))
D, FE = S["D"], S["feats"]
MAXG = 12
TUNE_END = 70          # windows 26..70 tune; 71..115 evaluate, config locked first


def rps_from_lam(lh, la, y):
    k = np.arange(MAXG + 1)
    ph = poisson.pmf(k[None, :], lh[:, None]); pa = poisson.pmf(k[None, :], la[:, None])
    g = ph[:, :, None] * pa[:, None, :]
    iu = np.triu_indices(MAXG + 1, 1); il = np.tril_indices(MAXG + 1, -1)
    p = np.stack([g[:, il[0], il[1]].sum(1), np.einsum("ikk->i", g),
                  g[:, iu[0], iu[1]].sum(1)], 1)
    p = p / p.sum(1, keepdims=True)
    o = np.zeros_like(p); o[np.arange(len(y)), y] = 1
    return ((p.cumsum(1) - o.cumsum(1))[:, :2] ** 2).sum(1) / 2


def walk_forward(cfg, wins, X, d, start_hist=25):
    """multiplicative correction m for each row, trained only on EARLIER windows."""
    out = np.full(len(d), np.nan)
    lam = d.lam.to_numpy()
    goals = (d.resid.to_numpy() + lam)
    y = np.clip(goals / lam, 0, 20)                 # offset trick target
    w = lam                                          # offset trick weight
    for w_ in wins:
        tr = (d.window < w_).to_numpy() & (d.window >= 1).to_numpy()
        te = (d.window == w_).to_numpy()
        if tr.sum() < 1500 or te.sum() == 0:
            continue
        Xtr, Xte = X[tr], X[te]
        med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        Xte = np.where(np.isfinite(Xte), Xte, med)
        mdl = HistGradientBoostingRegressor(loss="poisson", random_state=0,
                                            early_stopping=False, **cfg)
        mdl.fit(Xtr, y[tr], sample_weight=w[tr])
        out[te] = mdl.predict(Xte)
    return out


def score(m_corr, d, beta=1.0):
    """pair the two sides of each match and compute RPS with the correction applied."""
    d = d.reset_index(drop=True).copy()
    d["m"] = m_corr
    H = d[d.is_home == 1].reset_index(drop=True)
    A = d[d.is_home == 0].reset_index(drop=True)
    ok = np.isfinite(H.m) & np.isfinite(A.m)
    H, A = H[ok].reset_index(drop=True), A[ok].reset_index(drop=True)
    gh = (H.resid + H.lam).round().astype(int).to_numpy()
    ga = (A.resid + A.lam).round().astype(int).to_numpy()
    y = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))
    base = rps_from_lam(H.lam.to_numpy(), A.lam.to_numpy(), y)
    lh = H.lam.to_numpy() * np.power(np.clip(H.m.to_numpy(), .3, 3.0), beta)
    la = A.lam.to_numpy() * np.power(np.clip(A.m.to_numpy(), .3, 3.0), beta)
    adj = rps_from_lam(lh, la, y)
    return base, adj, H.window.to_numpy(), len(H)


def main():
    d = D.reset_index(drop=True)
    X = d[FE].to_numpy(float)
    wins = np.sort(d.window.unique())
    tune_w = [w for w in wins if 26 <= w <= TUNE_END]
    eval_w = [w for w in wins if w > TUNE_END]
    print(f"{len(FE)} features | tuning windows {tune_w[0]}-{tune_w[-1]}, "
          f"evaluation windows {eval_w[0]}-{eval_w[-1]}")
    print("Poisson loss with log(lambda) offset (target goals/lambda, weight lambda)\n")

    GRID = [
        dict(max_depth=2, max_iter=150, learning_rate=0.03, l2_regularization=10.0),
        dict(max_depth=3, max_iter=400, learning_rate=0.03, l2_regularization=1.0),
        dict(max_depth=None, max_leaf_nodes=31, max_iter=500, learning_rate=0.05,
             l2_regularization=1.0, min_samples_leaf=40),
        dict(max_depth=6, max_iter=800, learning_rate=0.02, l2_regularization=0.1,
             min_samples_leaf=20),
        dict(max_depth=4, max_iter=300, learning_rate=0.05, l2_regularization=5.0,
             max_features=0.5),
    ]
    print("=== tuning on windows 26-70 (dRPS, lower is better) ===")
    best, best_d = None, 1e9
    for i, cfg in enumerate(GRID):
        mc = walk_forward(cfg, tune_w, X, d)
        base, adj, g, n = score(mc, d)
        dd, lo, hi = paired_diff(pd.Series(adj), pd.Series(base), groups=g)
        print(f"  cfg{i}: {str(cfg)[:62]:62s} dRPS {dd:+.5f}  n={n}")
        if dd < best_d:
            best_d, best = dd, cfg
    print(f"\n  LOCKED config: {best}")
    print(f"  (tuning dRPS {best_d:+.5f} — in-sample for the CONFIG CHOICE, so not reportable)")

    print("\n=== EVALUATION on windows 71-115, config locked ===")
    mc = walk_forward(best, eval_w, X, d)
    base, adj, g, n = score(mc, d)
    print(f"  n={n} matches")
    print(f"  {'beta':>6s} {'RPS':>9s} {'dRPS':>10s} {'CI':>22s}")
    for beta in (0.0, 0.25, 0.5, 1.0):
        _, adjb, gb, _ = score(mc, d, beta=beta)
        dd, lo, hi = paired_diff(pd.Series(adjb), pd.Series(base), groups=gb)
        tag = "  <-- control" if beta == 0 else ("  <-- CI excludes 0" if hi < 0 else "")
        print(f"  {beta:6.2f} {adjb.mean():9.5f} {dd:+10.5f} [{lo:+.5f},{hi:+.5f}]{tag}")
    print("\n  this is the honest number: GBM with the right objective, tuned on a")
    print("  disjoint window range, trained walk-forward, evaluated on held-out windows.")




def by_history():
    """Does the correction improve as TRAINING HISTORY grows?

    The tuning half (windows 26-70) gave dRPS +0.00120 for the locked config while the
    evaluation half (71-115) gave -0.00035 to -0.00100. A sign flip between halves needs
    explaining. The benign story is training-set size: window 26 trains on ~800 rows,
    window 71 on ~4,500. If that is right, dRPS should fall MONOTONICALLY as history
    accumulates. If it is noise, the blocks will jump around.
    """
    import numpy as np
    import pandas as pd
    d = D.reset_index(drop=True)
    X = d[FE].to_numpy(float)
    cfg = dict(max_depth=2, max_iter=150, learning_rate=0.03, l2_regularization=10.0)
    wins = [w for w in np.sort(d.window.unique()) if w >= 26]
    mc = walk_forward(cfg, wins, X, d)
    d2 = d.copy(); d2["m"] = mc
    print("\n\n=== does the correction improve as training history grows? ===")
    print(f"  {'window block':16s} {'train rows':>11s} {'n':>6s} "
          f"{'dRPS b=0.5':>12s} {'CI':>22s}")
    for lo_w, hi_w in ((26, 45), (46, 65), (66, 85), (86, 115)):
        sub = d2[(d2.window >= lo_w) & (d2.window <= hi_w)]
        if not len(sub):
            continue
        base, adj, g, n = score(sub.m.to_numpy(), sub, beta=0.5)
        dd, l2, h2 = paired_diff(pd.Series(adj), pd.Series(base), groups=g)
        ntr = int((d2.window < lo_w).sum())
        print(f"  {f'{lo_w}-{hi_w}':16s} {ntr:11d} {n:6d} {dd:+12.5f} [{l2:+.5f},{h2:+.5f}]")
    print("\n  monotone improvement with history => the composite is data-limited and the")
    print("  sign flip is benign. Non-monotone => the result is unstable and not usable.")


if __name__ == "__main__":
    by_history()
