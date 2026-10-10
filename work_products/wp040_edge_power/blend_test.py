"""WP040 — does the model add information the MARKET PRICE does not already contain?

WHY NOT JUST FIX WP024's ROI TEST. Its MDE was 6.66% against professional edges of
1-3%, so it could not have detected a real edge either way. That is NOT fixable by a
better estimator: ROI = sum(pnl)/sum(stake) over ~1,000 flat-stake bets at average odds
~3, and the variance is dominated by the binary win/lose outcome, which carries no
information about whether an edge exists. Resolving 1-3% needs ~10-50k bets, i.e. a
decade of multi-league data. The sample is the binding constraint, not the method.

THE POWERFUL ALTERNATIVE. "Is there an edge" is the same question as "do the model's
DISAGREEMENTS with the price carry information". That is testable on EVERY match rather
than on a filtered subset of bets, and the estimator is far more efficient because it
never discretises into win/lose.

    p_blend = normalise( p_market^(1-w) * p_model^w )

Fit w by out-of-sample log-loss, leave-one-season-out. w = 0 means the model adds
nothing the price lacks; w > 0 means it does, and how much.

THE QUESTION THIS SETTLES, which is the one that matters for WP039. The market channel
(-0.00085 RPS) pulls the model's predictions TOWARD market consensus. That raises
accuracy while shrinking the disagreements profit depends on -- a model reproducing the
closing line exactly would have superb RPS and ZERO edge. So:

  * if the market channel's blend weight HOLDS or RISES, it genuinely improved the
    team-strength estimates and is worth having for betting as well as for RPS.
  * if its blend weight FALLS while its RPS improves, it is distilling the market rather
    than adding to it -- better on the scorecard, worse at beating the price.

Comparison is against the champion on the SAME matches, both at OPENING prices, since
those are what a bettor can actually act on.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
WP = HERE.parent
for p in (WP / "wp007_structural_fixes", WP / "wp008_market_benchmark"):
    sys.path.insert(0, str(p))
from analysis import attach_meta, load_checkpoints  # noqa: E402
from market_analysis import join_market  # noqa: E402

# OPENING prices only -- what a bettor can act on. pinnacle = sharp and sustainable,
# bet365 = soft but gets you restricted, market_avg = consensus reference.
ODDS = {
    "pinnacle": ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"],
    "bet365":   ["bet365_1x2_home", "bet365_1x2_draw", "bet365_1x2_away"],
    "market_avg": ["market_avg_1x2_home", "market_avg_1x2_draw", "market_avg_1x2_away"],
}
EPS = 1e-9


def load_arm(ckpt_dir, arm, df_cv):
    m, _ = load_checkpoints(ckpt_dir)
    m = attach_meta(m[m.arm == arm], df_cv)
    j, unmatched, mismatched = join_market(m, pd.read_pickle(
        WP / "wp008_market_benchmark" / "odds_raw.pkl"), books=ODDS)
    probs = m.set_index(["window", "round", "home", "away"])[["p_home", "p_draw", "p_away"]]
    j = j.set_index(["window", "round", "home", "away"]).join(probs, how="inner")
    return j.reset_index(), len(unmatched), len(mismatched)


def blend(pm_, pk, w):
    """geometric blend, renormalised: p ~ pk^(1-w) * pm^w"""
    z = np.exp((1 - w) * np.log(pk + EPS) + w * np.log(pm_ + EPS))
    return z / z.sum(1, keepdims=True)


def logloss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1)).mean()


def rps(p, y):
    o = np.zeros_like(p); o[np.arange(len(y)), y] = 1
    return ((p.cumsum(1) - o.cumsum(1))[:, :2] ** 2).sum(1).mean() / 2


def fit_w(pm_, pk, y):
    r = minimize_scalar(lambda w: logloss(blend(pm_, pk, w), y),
                        bounds=(0.0, 1.0), method="bounded")
    return float(r.x)


def evaluate(j, book, label):
    """leave-one-season-out: fit w on 9 seasons, score on the 10th."""
    cols = [f"{book}_shin_h", f"{book}_shin_d", f"{book}_shin_a"]
    d = j.dropna(subset=cols + ["p_home", "p_draw", "p_away", "season", "result"]).copy()
    y = d.result.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    pm_ = d[["p_home", "p_draw", "p_away"]].to_numpy()
    pk = d[cols].to_numpy()
    pk = pk / pk.sum(1, keepdims=True)

    ws, ll_mkt, ll_blend, ll_model, rps_mkt, rps_blend = [], [], [], [], [], []
    for s in sorted(d.season.unique()):
        tr, te = (d.season != s).to_numpy(), (d.season == s).to_numpy()
        w = fit_w(pm_[tr], pk[tr], y[tr])
        ws.append(w)
        ll_mkt.append(logloss(pk[te], y[te]))
        ll_model.append(logloss(pm_[te], y[te]))
        ll_blend.append(logloss(blend(pm_[te], pk[te], w), y[te]))
        rps_mkt.append(rps(pk[te], y[te]))
        rps_blend.append(rps(blend(pm_[te], pk[te], w), y[te]))
    ws = np.array(ws)
    print(f"  {label:26s} {book:11s} n={len(d):5d}  w={ws.mean():.3f} "
          f"[{ws.min():.3f},{ws.max():.3f}]  w>0 in {int((ws > 0.005).sum())}/{len(ws)}")
    print(f"  {'':26s} {'':11s} logloss  market {np.mean(ll_mkt):.5f} -> blend "
          f"{np.mean(ll_blend):.5f}  ({np.mean(ll_blend)-np.mean(ll_mkt):+.5f})"
          f"   model alone {np.mean(ll_model):.5f}")
    print(f"  {'':26s} {'':11s} RPS      market {np.mean(rps_mkt):.5f} -> blend "
          f"{np.mean(rps_blend):.5f}  ({np.mean(rps_blend)-np.mean(rps_mkt):+.5f})")
    return ws.mean(), np.mean(ll_blend) - np.mean(ll_mkt)


def main():
    df_cv = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                         .read_bytes())["df_cv"]
    arms = [("champion (no market channel)", WP / "wp032_xg_weight/checkpoints", "W075"),
            ("+ market channel (T100)", WP / "wp039_market_channel/checkpoints", "T100")]
    print("\nOPENING prices. w = the weight the MODEL earns in a blend with the price.")
    print("w near 0 => the model adds nothing the market has not already priced.\n")
    res = {}
    for label, ck, arm in arms:
        j, un, mm = load_arm(ck, arm, df_cv)
        if un or mm:
            print(f"  (dropped {un} unmatched, {mm} score-mismatched)")
        for book in ODDS:
            res[(label, book)] = evaluate(j, book, label)
        print()

    print("=" * 78)
    print("DOES THE MARKET CHANNEL HELP OR HURT AT BEATING THE PRICE?")
    print("=" * 78)
    for book in ODDS:
        a = res[(arms[0][0], book)]
        b = res[(arms[1][0], book)]
        print(f"  {book:11s} blend weight {a[0]:.3f} -> {b[0]:.3f} ({b[0]-a[0]:+.3f})"
              f"   logloss gain over market {a[1]:+.5f} -> {b[1]:+.5f}")
    print("\n  weight FALLING while RPS improves = distilling the market, not adding to it.")




def power_control():
    """POSITIVE CONTROL. The blend test says neither model adds information to the
    opening price out-of-sample. That is only meaningful if the test CAN detect a real
    improvement, so give it one it must find: Pinnacle's CLOSING price, blended into its
    own OPENING price. WP010 measured closing to be strictly sharper, so a test with
    power must return a large w and an OOS logloss GAIN.

    Also reported: how big an edge the test can resolve, by blending a synthetically
    sharpened version of the market into the opening price.
    """
    odds = pd.read_pickle(WP / "wp008_market_benchmark" / "odds_raw.pkl")
    df_cv = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                         .read_bytes())["df_cv"]
    books = {"open": ODDS["pinnacle"],
             "close": ["pinnacle_1x2_home_close", "pinnacle_1x2_draw_close",
                       "pinnacle_1x2_away_close"]}
    m, _ = load_checkpoints(WP / "wp032_xg_weight/checkpoints")
    m = attach_meta(m[m.arm == "W075"], df_cv)
    j, _, _ = join_market(m, odds, books=books)
    oc = [f"open_shin_{x}" for x in "hda"]
    cc = [f"close_shin_{x}" for x in "hda"]
    d = j.dropna(subset=oc + cc + ["season", "result"]).copy()
    y = d.result.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    po = d[oc].to_numpy(); po = po / po.sum(1, keepdims=True)
    pc = d[cc].to_numpy(); pc = pc / pc.sum(1, keepdims=True)

    print("\n" + "=" * 78)
    print("POSITIVE CONTROL — can this test detect a KNOWN real improvement?")
    print("=" * 78)
    ws, base, bl = [], [], []
    for s in sorted(d.season.unique()):
        tr, te = (d.season != s).to_numpy(), (d.season == s).to_numpy()
        w = fit_w(pc[tr], po[tr], y[tr])
        ws.append(w)
        base.append(logloss(po[te], y[te]))
        bl.append(logloss(blend(pc[te], po[te], w), y[te]))
    print(f"  CLOSING price blended into OPENING, n={len(d)}")
    print(f"    w = {np.mean(ws):.3f} [{min(ws):.3f},{max(ws):.3f}]")
    print(f"    logloss  opening {np.mean(base):.5f} -> blend {np.mean(bl):.5f}"
          f"  ({np.mean(bl)-np.mean(base):+.5f})")
    detected = np.mean(bl) < np.mean(base)
    print(f"    => the test {'DOES' if detected else 'does NOT'} detect it"
          f"{'' if detected else ' — the null above is then uninterpretable'}")

    print("\n  NOTE: the rows below are a FLAWED control and prove nothing. Sharpening")
    print("  an already-calibrated forecast DEGRADES it (see the 'alone' column: all")
    print("  positive, i.e. worse), so 'NOT detected' is the correct answer rather than")
    print("  a power failure. Kept for the record; the closing-price arm above is the")
    print("  control that actually works.")
    for k in (1.15, 1.08, 1.04, 1.02):
        ps = po ** k
        ps = ps / ps.sum(1, keepdims=True)
        ws2, b2, l2 = [], [], []
        for s in sorted(d.season.unique()):
            tr, te = (d.season != s).to_numpy(), (d.season == s).to_numpy()
            w = fit_w(ps[tr], po[tr], y[tr])
            ws2.append(w); b2.append(logloss(po[te], y[te]))
            l2.append(logloss(blend(ps[te], po[te], w), y[te]))
        gain_alone = logloss(ps, y) - logloss(po, y)
        print(f"    k={k:.2f}  (alone {gain_alone:+.5f} logloss)  w={np.mean(ws2):.3f}"
              f"  blend {np.mean(l2)-np.mean(b2):+.5f}"
              f"  {'detected' if np.mean(l2) < np.mean(b2) else 'NOT detected'}")


if __name__ == "__main__":
    main()
    power_control()
