"""WP042 — does the model beat the CLOSING LINE? The high-power money test.

WHY THIS IS THE RIGHT TEST, and why WP024/WP041 could not answer the money question.
ROI scores a bet against the BINARY outcome, whose variance (~1.0 per bet at even money,
~1.5-2.0 at 3-way odds) swamps any 1-3% edge: MDE came out 6.66% on 1X2 and 5.02% on
totals. Closing-line value scores the same bet against the CLOSING PROBABILITY instead --
a far lower-variance estimate of the true probability -- which cuts the standard error by
roughly an order of magnitude on the same sample.

    CLV = mean over taken bets of ( p_close_devig * odds_taken_at_open - 1 )

Positive CLV means we are systematically getting prices better than where the market
ends up. It is the professional standard for detecting edge on a limited sample, because
if you consistently beat the closing line you will profit in the long run.

WHAT IT ASSUMES, stated plainly: that the closing price is approximately efficient. If
the close carries a systematic bias, CLV inherits it. Pinnacle's close is about as close
to efficient as football betting gets (WP040 measured it blending into the open at
w = 1.000), so this is a reasonable assumption here -- but it is an assumption, not a
realised P&L.

SECOND TEST: does the model's disagreement with the open PREDICT the line's movement?
    (p_close - p_open) ~ a + b * (p_model - p_open)
b > 0 means our disagreements point the way the market is about to go.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP = HERE.parent
for p in (WP / "wp007_structural_fixes", WP / "wp008_market_benchmark"):
    sys.path.insert(0, str(p))
from analysis import attach_meta, load_checkpoints  # noqa: E402
from market_analysis import join_market  # noqa: E402

BOOKS = {
    "pin_open":  ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"],
    "pin_close": ["pinnacle_1x2_home_close", "pinnacle_1x2_draw_close",
                  "pinnacle_1x2_away_close"],
    "b365_open": ["bet365_1x2_home", "bet365_1x2_draw", "bet365_1x2_away"],
}
OUTCOMES = ["H", "D", "A"]


def boot(x, g, reps=4000, seed=0):
    uq = np.unique(g)
    idx = {w: np.where(g == w)[0] for w in uq}
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for i in range(reps):
        sel = np.concatenate([idx[w] for w in rng.choice(uq, len(uq), replace=True)])
        out[i] = x[sel].mean()
    return x.mean(), *np.percentile(out, [2.5, 97.5])


def load(ck, arm):
    df_cv = pickle.loads((WP / "wp030_squad_intactness/cv_shared_data_intact.pkl")
                         .read_bytes())["df_cv"]
    m, _ = load_checkpoints(ck)
    m = attach_meta(m[m.arm == arm], df_cv)
    j, _, _ = join_market(m, pd.read_pickle(
        WP / "wp008_market_benchmark" / "odds_raw.pkl"), books=BOOKS)
    probs = m.set_index(["window", "round", "home", "away"])[["p_home", "p_draw", "p_away"]]
    return j.set_index(["window", "round", "home", "away"]).join(probs, how="inner").reset_index()


def clv_table(j, label, open_book, edge_grid=(0.0, 0.02, 0.05)):
    oc = [f"{open_book}_shin_{x}" for x in "hda"]
    cc = [f"pin_close_shin_{x}" for x in "hda"]
    ocols = BOOKS[open_book]
    d = j.dropna(subset=oc + cc + ocols + ["p_home", "p_draw", "p_away"]).copy()
    po = d[oc].to_numpy(); po = po / po.sum(1, keepdims=True)
    pc = d[cc].to_numpy(); pc = pc / pc.sum(1, keepdims=True)
    pm_ = d[["p_home", "p_draw", "p_away"]].to_numpy()
    odds = d[ocols].to_numpy()
    win = d.window.to_numpy()
    res = d.result.to_numpy()

    # THE NO-EDGE BENCHMARK, without which these numbers are uninterpretable.
    # CLV here uses the OFFERED odds, which carry the book's margin, so a bettor with
    # ZERO skill scores roughly minus the overround -- not zero. Two references:
    #   all-legs : stake every outcome of every match equally
    #   price-wt : stake in proportion to the opening price's own de-vigged probs
    # Our selections' CLV must beat THESE, not beat zero.
    clv_all = (pc * odds - 1.0).ravel()
    bench_all, ba_lo, ba_hi = boot(clv_all, np.repeat(win, 3))
    wt = po / po.sum(1, keepdims=True)
    clv_wt = (wt * (pc * odds - 1.0)).sum(1)
    bench_wt, bw_lo, bw_hi = boot(clv_wt, win)
    print(f"\n  {label}  betting at {open_book}, graded vs PINNACLE CLOSE  (n={len(d)})")
    print(f"    NO-EDGE BENCHMARK  all-legs {bench_all:+.2%} [{ba_lo:+.2%},{ba_hi:+.2%}]"
          f"   price-weighted {bench_wt:+.2%} [{bw_lo:+.2%},{bw_hi:+.2%}]")
    print(f"    (that is the book's margin; our selections must beat this, not beat 0)")
    print(f"    {'edge':>5s} {'bets':>6s} {'CLV':>9s} {'CI':>20s} "
          f"{'ROI':>9s} {'CI (for contrast)':>22s}")
    for edge in edge_grid:
        ev = pm_ * odds - 1.0
        take = ev > edge
        if take.sum() < 50:
            print(f"    {edge:5.0%} {int(take.sum()):6d}   too few"); continue
        i, k = np.where(take)
        clv = pc[i, k] * odds[i, k] - 1.0
        won = (res[i] == np.array(OUTCOMES)[k])
        pnl = np.where(won, odds[i, k] - 1.0, -1.0)
        c, clo, chi = boot(clv, win[i])
        r, rlo, rhi = boot(pnl, win[i])
        excess = c - bench_wt
        flag = ("  <-- BEATS the no-edge benchmark" if clo > bench_wt
                else "  <-- WORSE than no-edge" if chi < bench_wt else "")
        print(f"    {edge:5.0%} {len(clv):6d} {c:+8.2%} [{clo:+6.2%},{chi:+6.2%}] "
              f"{r:+8.2%} [{rlo:+6.2%},{rhi:+6.2%}]  excess {excess:+.2%}{flag}")
    # how much power did CLV buy us?
    ev = pm_ * odds - 1.0
    i, k = np.where(ev > 0.0)
    clv = pc[i, k] * odds[i, k] - 1.0
    won = (res[i] == np.array(OUTCOMES)[k])
    pnl = np.where(won, odds[i, k] - 1.0, -1.0)
    _, a, b = boot(clv, win[i]); _, c2, d2 = boot(pnl, win[i])
    print(f"    MDE: CLV +-{(b-a)/2:.2%}  vs  ROI +-{(d2-c2)/2:.2%}"
          f"   ({(d2-c2)/(b-a):.0f}x more powerful)")


def movement_test(j, label):
    """Do the model's disagreements with the OPEN predict where the line MOVES?"""
    oc = [f"pin_open_shin_{x}" for x in "hda"]
    cc = [f"pin_close_shin_{x}" for x in "hda"]
    d = j.dropna(subset=oc + cc + ["p_home", "p_draw", "p_away"]).copy()
    po = d[oc].to_numpy(); po = po / po.sum(1, keepdims=True)
    pc = d[cc].to_numpy(); pc = pc / pc.sum(1, keepdims=True)
    pm_ = d[["p_home", "p_draw", "p_away"]].to_numpy()
    x = (pm_ - po).ravel()
    y = (pc - po).ravel()
    g = np.repeat(d.window.to_numpy(), 3)
    uq = np.unique(g)
    idx = {w: np.where(g == w)[0] for w in uq}
    rng = np.random.default_rng(0)
    bs = np.empty(2000)
    for i in range(2000):
        sel = np.concatenate([idx[w] for w in rng.choice(uq, len(uq), replace=True)])
        bs[i] = np.polyfit(x[sel], y[sel], 1)[0]
    b = np.polyfit(x, y, 1)[0]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\n  {label}  d(close-open) / d(model-open) = {b:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print(f"    corr = {np.corrcoef(x, y)[0,1]:+.4f}   n={len(x)} outcome-legs")
    return b, lo


def main():
    print("=" * 78)
    print("CLOSING-LINE VALUE — the high-power money test (EPL 1X2)")
    print("=" * 78)
    for label, ck, arm in (("champion", WP / "wp032_xg_weight/checkpoints", "W075"),
                           ("+ market channel", WP / "wp039_market_channel/checkpoints", "T100")):
        j = load(ck, arm)
        for ob in ("pin_open", "b365_open"):
            clv_table(j, label, ob)
        movement_test(j, label)
    print("\n  POSITIVE CLV = systematically getting better prices than the market's own")
    print("  close, which is the professional proxy for long-run profit. A positive")
    print("  MOVEMENT slope means our disagreements point where the line is going.")


if __name__ == "__main__":
    main()
