"""WP053 — does the composite actually improve RPS? Measured, not extrapolated.

The residual is predictably non-zero (p=0.000 against a 40-shuffle null), but the
magnitude estimates span 2.5x for our features and 9x for the market ceiling depending on
learner and feature count, so scaling OOS R^2 into an RPS figure is not trustworthy.

This measures it directly: train the composite WALK-FORWARD (each CV window's composite
sees only matches from strictly earlier windows, matching the model's own protocol, NOT
the leave-one-season-out used for the bound), apply it as a multiplicative correction to
the champion's lambdas, and recompute RPS on the same 3,780 matches.

    lambda_adj = lambda * exp(beta * composite / lambda)

beta is a global shrinkage on the correction, swept rather than fitted, because a fitted
beta on the same data it corrects would be circular. beta = 0 reproduces the champion
exactly, which is the control.

WP036 used exactly this technique to kill the lambda-shrinkage idea in minutes; if a
post-hoc correction cannot find RPS here, an in-model version will not either.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
WP = HERE.parent
import sys
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
from analysis import paired_diff  # noqa: E402

S = pickle.load((HERE / "features.pkl").open("rb"))
D, FE = S["D"], S["feats"]
MAXG = 12


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


def main():
    d = D.copy().reset_index(drop=True)
    X = d[FE].to_numpy(float)
    wins = np.sort(d.window.unique())
    # WALK-FORWARD composite: window w trained only on windows < w
    pred = np.full(len(d), np.nan)
    for i, w in enumerate(wins):
        if i < 25:            # need a usable training base
            continue
        tr = (d.window < w).to_numpy(); te = (d.window == w).to_numpy()
        Xtr, Xte = X[tr], X[te]
        med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        Xte = np.where(np.isfinite(Xte), Xte, med)
        mdl = HistGradientBoostingRegressor(max_depth=2, max_iter=150, learning_rate=0.03,
                                            l2_regularization=10.0, early_stopping=False,
                                            random_state=0)
        mdl.fit(Xtr, d.resid.to_numpy()[tr])
        pred[te] = mdl.predict(Xte)
    ok = np.isfinite(pred)
    print(f"walk-forward composite available for {ok.sum()}/{len(d)} rows "
          f"({d.window[ok].nunique()} windows)")
    corr = np.corrcoef(pred[ok], d.resid.to_numpy()[ok])[0, 1]
    print(f"walk-forward corr(composite, residual) = {corr:+.4f}\n")

    # rebuild match rows: each match has a home side (is_home=1) and an away side
    d["pred"] = pred
    hh = d[(d.is_home == 1) & ok].set_index(["window", "lam", "lam_opp"], drop=False)
    # pair the two sides by (window, lam, lam_opp) <-> (window, lam_opp, lam)
    H = d[(d.is_home == 1)].reset_index(drop=True)
    A = d[(d.is_home == 0)].reset_index(drop=True)
    assert len(H) == len(A), "sides not paired"
    m = pd.DataFrame(dict(window=H.window, lh=H.lam, la=A.lam,
                          ph=H.pred, pa=A.pred,
                          gh=(H.resid + H.lam).round().astype(int),
                          ga=(A.resid + A.lam).round().astype(int)))
    m = m[np.isfinite(m.ph) & np.isfinite(m.pa)].copy()
    y = np.where(m.gh > m.ga, 0, np.where(m.gh == m.ga, 1, 2))
    base = rps_from_lam(m.lh.to_numpy(), m.la.to_numpy(), y)
    print(f"control RPS (beta=0) {base.mean():.5f} on {len(m)} matches\n")
    print(f"  {'beta':>6s} {'RPS':>9s} {'dRPS':>10s} {'CI':>22s}")
    for beta in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        lh = m.lh.to_numpy() * np.exp(beta * m.ph.to_numpy() / m.lh.to_numpy())
        la = m.la.to_numpy() * np.exp(beta * m.pa.to_numpy() / m.la.to_numpy())
        r = rps_from_lam(lh, la, y)
        dd, lo, hi = paired_diff(pd.Series(r), pd.Series(base), groups=m.window.to_numpy())
        tag = "  <-- control" if beta == 0 else ("  <-- CI excludes 0" if hi < 0 else "")
        print(f"  {beta:6.2f} {r.mean():9.5f} {dd:+10.5f} [{lo:+.5f},{hi:+.5f}]{tag}")
    print("\n  if no beta improves RPS, the predictable residual is not RPS-exploitable")
    print("  and an in-model version will not help either (WP036's lesson).")


if __name__ == "__main__":
    main()
