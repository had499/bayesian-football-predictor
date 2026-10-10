"""WP024 — would betting this model actually make money?

A different question from RPS. A model can be WORSE than Pinnacle on accuracy and
still profit, because profit comes from finding prices that are wrong at the book
you can actually bet, not from beating the sharpest line in the world.

Uses the champion (L0 + xG channel + deep channel) on WP021's 115-window grid:
3,780 walk-forward matches, every prediction genuinely out-of-sample.

PRICES. Four series, deliberately spanning optimistic to realistic:

  market_max   best price across all books      WHAT WAS ASKED FOR -- and the most
                                                optimistic. Mean overround 1.0049 and
                                                **32% of matches show overround < 1.00,
                                                i.e. pure arbitrage**. That cannot be a
                                                simultaneously available price; it is a
                                                max taken over books AND over time. Treat
                                                its results as an upper bound that nobody
                                                could actually trade.
  bet365       a soft book, openers             realistic to bet, but beating it
                                                persistently gets accounts restricted
  pinnacle     sharp book, openers              realistic AND sustainable -- Pinnacle
                                                welcomes winners -- but hardest to beat
  market_avg   the consensus                    a reference, not tradeable

All at the OPENING price, as requested. WP010 measured the model at +0.0039 RPS behind
Pinnacle's pre-close line (vs +0.0139 behind the close), so openers are where it is
closest to competitive.

STAKING. Flat stakes (1 unit) is the headline, because Kelly's returns are dominated by
a few large bets and its variance makes a 3,780-match sample look better or worse than
it is. Fractional Kelly is reported alongside.

WHAT THIS DOES NOT MODEL: stake limits at opening prices (often very low), account
restriction, price movement between the recorded open and when you could act, and the
fact that a 32%-arbitrage max series is not obtainable. Every one of those pushes real
returns BELOW what is reported here.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP007 = HERE.parent / "wp007_structural_fixes"
WP008 = HERE.parent / "wp008_market_benchmark"
WP021 = HERE.parent / "wp021_power_confirmation"
WP023 = HERE.parent / "wp023_nutpie_baseline"
for p in (WP007, WP008):
    sys.path.insert(0, str(p))
from analysis import attach_meta, load_checkpoints  # noqa: E402
from market_analysis import join_market  # noqa: E402

ODDS = {
    "market_max": ["market_max_1x2_home", "market_max_1x2_draw", "market_max_1x2_away"],
    "bet365":     ["bet365_1x2_home", "bet365_1x2_draw", "bet365_1x2_away"],
    "pinnacle":   ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"],
    "market_avg": ["market_avg_1x2_home", "market_avg_1x2_draw", "market_avg_1x2_away"],
}
OUTCOMES = ["H", "D", "A"]


def load():
    df_cv = pickle.loads((WP021 / "cv_shared_data_big.pkl").read_bytes())["df_cv"]
    m, _ = load_checkpoints(WP023 / "checkpoints")
    m = attach_meta(m[m.arm == "H_deep_added"], df_cv)
    j, unmatched, mismatched = join_market(m, pd.read_pickle(WP008 / "odds_raw.pkl"), books=ODDS)
    if len(unmatched) or len(mismatched):
        print(f"note: dropped {len(unmatched)} unmatched, {len(mismatched)} score-mismatched rows")
    probs = m.set_index(["window", "round", "home", "away"])[["p_home", "p_draw", "p_away"]]
    j = j.set_index(["window", "round", "home", "away"]).join(probs, how="inner")
    return j.reset_index()


def simulate(j, book, edge, kelly_frac=0.0):
    """One unit staked per qualifying bet (or fractional Kelly). Returns per-bet rows."""
    cols = ODDS[book]
    out = []
    for oc, pcol, ocol in zip(OUTCOMES, ["p_home", "p_draw", "p_away"], cols):
        d = j[[ "window", "result", pcol, ocol]].dropna()
        p, o = d[pcol].to_numpy(), d[ocol].to_numpy()
        ev = p * o - 1.0
        take = ev > edge
        if not take.any():
            continue
        won = (d.result.to_numpy()[take] == oc)
        odds, pp = o[take], p[take]
        if kelly_frac > 0:
            b = odds - 1.0
            stake = np.clip(kelly_frac * ((pp * odds - 1.0) / b), 0, 0.05)
        else:
            stake = np.ones(take.sum())
        pnl = np.where(won, stake * (odds - 1.0), -stake)
        out.append(pd.DataFrame(dict(window=d.window.to_numpy()[take], stake=stake,
                                     pnl=pnl, ev=ev[take], odds=odds, outcome=oc)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def roi_ci(bets, reps=4000, seed=0):
    """Window-clustered bootstrap: matches inside a CV window share one model fit."""
    if not len(bets):
        return np.nan, np.nan, np.nan
    g = bets.window.to_numpy()
    uq = np.unique(g)
    idx = {w: np.where(g == w)[0] for w in uq}
    rng = np.random.default_rng(seed)
    pnl, stake = bets.pnl.to_numpy(), bets.stake.to_numpy()
    out = np.empty(reps)
    for i in range(reps):
        sel = np.concatenate([idx[w] for w in rng.choice(uq, len(uq), replace=True)])
        out[i] = pnl[sel].sum() / stake[sel].sum()
    return pnl.sum() / stake.sum(), *np.percentile(out, [2.5, 97.5])


def main():
    j = load()
    print(f"\n{len(j)} matches with model predictions joined to odds\n")
    print("=" * 78)
    print("FLAT STAKES, opening prices, bet whenever model EV exceeds the threshold")
    print("=" * 78)
    print(f"{'book':>11s} {'edge':>6s} {'bets':>6s} {'%played':>8s} {'ROI':>8s} {'CI (window-clustered)':>26s}")
    for book in ODDS:
        for edge in (0.0, 0.02, 0.05, 0.10):
            b = simulate(j, book, edge)
            if not len(b):
                print(f"{book:>11s} {edge:6.0%} {0:6d}      --       --")
                continue
            r, lo, hi = roi_ci(b)
            flag = "  <-- CI excludes 0" if lo > 0 else ""
            print(f"{book:>11s} {edge:6.0%} {len(b):6d} {len(b)/(3*len(j)):7.1%} "
                  f"{r:+7.2%} [{lo:+7.2%}, {hi:+7.2%}]{flag}")
        print()

    print("=" * 78)
    print("QUARTER-KELLY (capped at 5% of bank), 2% edge threshold")
    print("=" * 78)
    for book in ODDS:
        b = simulate(j, book, 0.02, kelly_frac=0.25)
        if not len(b):
            continue
        r, lo, hi = roi_ci(b)
        print(f"{book:>11s} bets {len(b):5d}  turnover {b.stake.sum():8.1f}u  "
              f"ROI {r:+7.2%} [{lo:+7.2%}, {hi:+7.2%}]  profit {b.pnl.sum():+8.1f}u")


if __name__ == "__main__":
    main()
