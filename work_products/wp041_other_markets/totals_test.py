"""WP041 — does the model have anything on the OVER/UNDER 2.5 market?

WHY TOTALS. WP040 showed a well-powered NO on 1X2 against the opening price. But every
test in this project has scored only the model's 1X2 COLLAPSE, which discards the
total-goals information in its scoreline grid. And the xG channel -- the one established
win (-0.00258, and WP032 showed its weight was under-set) -- informs goal RATES directly,
which is precisely what a totals market prices. So this is where the model's strongest
component is most directly expressed and least tested.

SAMPLE. Totals odds exist only from season 2019 (verified), so ~2,476 matches over 7
seasons rather than 3,780 over 10.

METHOD. Identical to WP040, whose power is established (it detects closing-vs-opening at
w=1.000 / -0.00313):
  * model P(over 2.5) from the Poisson scoreline grid on the posterior-mean rates. WP036
    verified this plug-in reconstruction reproduces the harness's 1X2 RPS to -0.00000.
  * two-way de-vig by proportional normalisation (for a 2-outcome market Shin and
    proportional agree closely, and the overround is small).
  * blend p_market^(1-w) * p_model^w, w fitted by log-loss LEAVE-ONE-SEASON-OUT.
  * w = 0 means the model adds nothing the totals price lacks.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
WP = HERE.parent
sys.path.insert(0, str(WP / "wp007_structural_fixes"))
sys.path.insert(0, str(WP / "wp008_market_benchmark"))
from analysis import attach_meta, load_checkpoints  # noqa: E402
from market_analysis import CODE_TO_FD  # noqa: E402

EPS = 1e-9
MAXG = 15
BOOKS = {"pinnacle": ("pinnacle_over25", "pinnacle_under25"),
         "pinnacle_close": ("pinnacle_over25_close", "pinnacle_under25_close"),
         "market_avg": ("market_avg_over25", "market_avg_under25"),
         "market_max": ("market_max_over25", "market_max_under25")}


def p_over(lh, la, line=2.5):
    """P(total goals > line) under independent Poisson on the posterior-mean rates."""
    k = np.arange(MAXG + 1)
    ph = poisson.pmf(k[None, :], lh[:, None])
    pa = poisson.pmf(k[None, :], la[:, None])
    tot = np.zeros((len(lh), 2 * MAXG + 1))
    for i in range(MAXG + 1):
        tot[:, i:i + MAXG + 1] += ph[:, [i]] * pa
    thr = int(np.floor(line)) + 1
    return tot[:, thr:].sum(1)


def blend(pm_, pk, w):
    """two-outcome geometric blend, renormalised"""
    a = np.exp((1 - w) * np.log(pk + EPS) + w * np.log(pm_ + EPS))
    b = np.exp((1 - w) * np.log(1 - pk + EPS) + w * np.log(1 - pm_ + EPS))
    return a / (a + b)


def logloss(p, y):
    return -np.mean(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1)))


def brier(p, y):
    return np.mean((p - y) ** 2)


def fit_w(pm_, pk, y):
    r = minimize_scalar(lambda w: logloss(blend(pm_, pk, w), y),
                        bounds=(0.0, 1.0), method="bounded")
    return float(r.x)


def load(arm_dir, arm):
    df_cv = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                         .read_bytes())["df_cv"]
    m, _ = load_checkpoints(arm_dir)
    m = attach_meta(m[m.arm == arm], df_cv)
    m["home_fd"] = m.home.map(CODE_TO_FD)
    m["away_fd"] = m.away.map(CODE_TO_FD)
    o = pd.read_pickle(HERE / "odds_other.pkl")
    o["date"] = pd.to_datetime(o.date).dt.normalize()
    m["date"] = pd.to_datetime(m.date).dt.normalize()
    j = m.merge(o, left_on=["date", "home_fd", "away_fd"],
                right_on=["date", "home_team", "away_team"], how="inner")
    ok = (j.goals_home == j.fthg) & (j.goals_away == j.ftag)
    j = j[ok].copy()
    j["p_over_model"] = p_over(j.lambda_home.to_numpy(), j.lambda_away.to_numpy())
    j["y_over"] = ((j.goals_home + j.goals_away) > 2.5).astype(int)
    return j


def devig2(o_over, o_under):
    po, pu = 1 / o_over, 1 / o_under
    return po / (po + pu)


def evaluate(j, book, label):
    co, cu = BOOKS[book]
    d = j.dropna(subset=[co, cu, "p_over_model", "season"]).copy()
    if len(d) < 300:
        print(f"  {label:24s} {book:15s} n={len(d)} — too few"); return None
    pk = devig2(d[co].to_numpy(), d[cu].to_numpy())
    pm_ = d.p_over_model.to_numpy()
    y = d.y_over.to_numpy()
    ws, lm, lk, lb, bm, bk = [], [], [], [], [], []
    for s in sorted(d.season.unique()):
        tr, te = (d.season != s).to_numpy(), (d.season == s).to_numpy()
        if tr.sum() < 200 or te.sum() < 50:
            continue
        w = fit_w(pm_[tr], pk[tr], y[tr])
        ws.append(w)
        lk.append(logloss(pk[te], y[te])); lm.append(logloss(pm_[te], y[te]))
        lb.append(logloss(blend(pm_[te], pk[te], w), y[te]))
        bk.append(brier(pk[te], y[te])); bm.append(brier(pm_[te], y[te]))
    gain = np.mean(lb) - np.mean(lk)
    print(f"  {label:24s} {book:15s} n={len(d):5d}  w={np.mean(ws):.3f} "
          f"[{min(ws):.3f},{max(ws):.3f}]")
    print(f"  {'':24s} {'':15s} logloss  market {np.mean(lk):.5f}  model "
          f"{np.mean(lm):.5f}  blend {np.mean(lb):.5f}  ({gain:+.5f})"
          f"  {'BLEND WINS' if gain < 0 else ''}")
    print(f"  {'':24s} {'':15s} brier    market {np.mean(bk):.5f}  model {np.mean(bm):.5f}"
          f"   over-rate {y.mean():.3f}")
    return np.mean(ws), gain


def main():
    arms = [("champion", WP / "wp032_xg_weight/checkpoints", "W075"),
            ("+ market channel", WP / "wp039_market_channel/checkpoints", "T100")]
    print("\nOVER/UNDER 2.5 — does the model add to the TOTALS price?")
    print("(w = weight the model earns in a blend; blend gain < 0 means it beats the price)\n")
    out = {}
    for label, ck, arm in arms:
        j = load(ck, arm)
        print(f"  [{label}] joined {len(j)} matches, "
              f"seasons {sorted(j.season.unique())}")
        for book in BOOKS:
            r = evaluate(j, book, label)
            if r:
                out[(label, book)] = r
        print()
    print("=" * 78)
    for book in BOOKS:
        ks = [(l, book) for l, _, _ in arms if (l, book) in out]
        if len(ks) == 2:
            a, b = out[ks[0]], out[ks[1]]
            print(f"  {book:15s} w {a[0]:.3f} -> {b[0]:.3f}   blend gain "
                  f"{a[1]:+.5f} -> {b[1]:+.5f}")


if __name__ == "__main__":
    main()
