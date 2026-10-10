"""WP041 — is there MONEY in the over/under 2.5 market, not just information?

The blend test found the champion earns w = 0.203-0.244 on totals, 4x its 1X2 weight
(0.046-0.098), yet the blend does not beat the price. This asks the money question
directly, which is better powered here than on 1X2 for a structural reason: totals are
near even-money, so per-bet variance is ~1.0 rather than the ~1.5-2.0 of 3-way longshots,
and every match is bettable rather than a filtered subset.

Reported per book and edge threshold, with window-clustered CIs (matches inside a CV
window share one model fit). `market_max` is kept for reference only -- WP024 measured it
as 32% arbitrage, i.e. not simultaneously available.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
sys.path.insert(0, str(WP / "wp008_market_benchmark"))
from totals_test import BOOKS, load, devig2  # noqa: E402

BOOKS_T = {k: v for k, v in BOOKS.items() if k != "pinnacle_close"}


def roi_ci(pnl, stake, g, reps=4000, seed=0):
    uq = np.unique(g)
    idx = {w: np.where(g == w)[0] for w in uq}
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for i in range(reps):
        sel = np.concatenate([idx[w] for w in rng.choice(uq, len(uq), replace=True)])
        out[i] = pnl[sel].sum() / stake[sel].sum()
    return pnl.sum() / stake.sum(), *np.percentile(out, [2.5, 97.5])


def simulate(d, co, cu, edge):
    """Bet the side the model prefers, when model EV exceeds the threshold."""
    p = d.p_over_model.to_numpy()
    oo, ou = d[co].to_numpy(), d[cu].to_numpy()
    y = d.y_over.to_numpy()
    g = d.window.to_numpy()
    ev_o, ev_u = p * oo - 1.0, (1 - p) * ou - 1.0
    rows = []
    for side, ev, odds, won in (("over", ev_o, oo, y == 1), ("under", ev_u, ou, y == 0)):
        take = ev > edge
        if not take.any():
            continue
        pnl = np.where(won[take], odds[take] - 1.0, -1.0)
        rows.append(pd.DataFrame(dict(window=g[take], stake=1.0, pnl=pnl,
                                      side=side, odds=odds[take])))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    j = load(WP / "wp032_xg_weight/checkpoints", "W075")
    print("\nOVER/UNDER 2.5, flat stakes, champion model")
    print("(totals are near even-money, so this is better powered than the 1X2 ROI)\n")
    print(f"  {'book':12s} {'edge':>5s} {'bets':>6s} {'%played':>8s} {'ROI':>8s} "
          f"{'CI (window-clustered)':>24s}")
    for book, (co, cu) in BOOKS_T.items():
        d = j.dropna(subset=[co, cu, "p_over_model"]).copy()
        for edge in (0.0, 0.02, 0.05):
            b = simulate(d, co, cu, edge)
            if not len(b):
                print(f"  {book:12s} {edge:5.0%} {0:6d}       --       --"); continue
            r, lo, hi = roi_ci(b.pnl.to_numpy(), b.stake.to_numpy(), b.window.to_numpy())
            flag = "  <-- CI excludes 0" if lo > 0 else ""
            print(f"  {book:12s} {edge:5.0%} {len(b):6d} {len(b)/(2*len(d)):7.1%} "
                  f"{r:+7.2%} [{lo:+6.2%},{hi:+6.2%}]{flag}")
        print()

    print("=== power: how small an ROI could this detect? ===")
    d = j.dropna(subset=["pinnacle_over25", "pinnacle_under25", "p_over_model"]).copy()
    b = simulate(d, "pinnacle_over25", "pinnacle_under25", 0.0)
    r, lo, hi = roi_ci(b.pnl.to_numpy(), b.stake.to_numpy(), b.window.to_numpy())
    print(f"  pinnacle, all bets: n={len(b)}, CI half-width {(hi-lo)/2:.2%}")
    print(f"  => MDE ~{(hi-lo)/2:.2%} vs professional edges of 1-3%")
    print("  (1X2 MDE was 6.66%; totals is better but may still not resolve 1-3%)")

    print("\n=== where the model disagrees most with the price ===")
    d["dis"] = (d.p_over_model - devig2(d.pinnacle_over25.to_numpy(),
                                        d.pinnacle_under25.to_numpy())).abs()
    d["q"] = pd.qcut(d.dis, 4, labels=["Q1 agree", "Q2", "Q3", "Q4 disagree"])
    for q, k in d.groupby("q", observed=True):
        b = simulate(k, "pinnacle_over25", "pinnacle_under25", 0.0)
        if len(b) < 100:
            continue
        r, lo, hi = roi_ci(b.pnl.to_numpy(), b.stake.to_numpy(), b.window.to_numpy())
        print(f"  {str(q):14s} n={len(b):5d}  mean|disagree| {k.dis.mean():.3f}  "
              f"ROI {r:+7.2%} [{lo:+6.2%},{hi:+6.2%}]")
    print("\n  WP003/WP008 found a disagreement-quantile pattern on 1X2; if a totals edge")
    print("  exists it should concentrate in Q4, where the model's view differs most.")


if __name__ == "__main__":
    main()
