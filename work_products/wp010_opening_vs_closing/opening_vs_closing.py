"""WP010 — benchmark WP007's models against Pinnacle's PRE-CLOSING line, not
just the closing line, and size how much of the remaining gap is late
information the model structurally cannot have.

Run:  python work_products/wp010_opening_vs_closing/opening_vs_closing.py

Reuses WP008's odds cache and join (one odds pipeline, not two) — the only
new thing here is the book set (pre-closing columns alongside closing) and
the comparisons below.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WP010 = Path(__file__).resolve().parent
WP008 = WP010.parent / "wp008_market_benchmark"
WP007 = WP010.parent / "wp007_structural_fixes"
sys.path.insert(0, str(WP008))
sys.path.insert(0, str(WP007))
from market_analysis import B365, PIN, join_market, load_wp007_matches  # noqa: E402
from analysis import bootstrap_mean_ci, rps_hda, verdict  # noqa: E402

# Same books, pre-closing snapshot (football-data.co.uk's `C` infix marks
# closing; these are the columns without it). The mirror's SPEC doesn't say
# *when* the non-`C` snapshot is taken, so `describe_snapshots` below checks
# empirically that it behaves like an earlier, less-informed price rather
# than trusting the label.
PIN_OPEN = ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"]
B365_OPEN = ["bet365_1x2_home", "bet365_1x2_draw", "bet365_1x2_away"]
BOOKS = {"pin_close": PIN, "pin_open": PIN_OPEN, "b365_close": B365, "b365_open": B365_OPEN}

ARMS = ["L0_baseline", "L5_xgchannel"]
KEY = ["window", "round", "home", "away"]


def build():
    """Per-match frame with both Pinnacle snapshots and both arms' RPS."""
    M = load_wp007_matches(ARMS)
    odds = pd.read_pickle(WP008 / "odds_raw.pkl")
    joined, unmatched, mismatched = join_market(M[M.arm == ARMS[0]], odds, books=BOOKS)
    if len(unmatched) or len(mismatched):
        print(f"note: {len(unmatched)} unmatched, {len(mismatched)} score-mismatched rows dropped")

    # apples-to-apples: only matches carrying BOTH Pinnacle snapshots
    both = joined[joined[["pin_close_h", "pin_open_h"]].notna().all(axis=1)].copy()
    both["r_open"] = rps_hda(both[["pin_open_h", "pin_open_d", "pin_open_a"]].to_numpy(), both.result)
    both["r_close"] = rps_hda(both[["pin_close_h", "pin_close_d", "pin_close_a"]].to_numpy(), both.result)
    for arm in ARMS:
        Ma = M[M.arm == arm].set_index(KEY).loc[both.set_index(KEY).index]
        both[f"r_{arm}"] = rps_hda(Ma[["p_home", "p_draw", "p_away"]].to_numpy(), Ma.result).astype(float)
    # how far the price travelled between the two snapshots
    both["line_move"] = (both.pin_close_h - both.pin_open_h).abs()
    # upset defined off the CLOSING favourite (the sharper of the two)
    fav_home = both.pin_close_h >= both.pin_close_a
    both["upset"] = np.where(fav_home, both.result != "H", both.result != "A")
    return both, joined


def _cmp(a, b, groups, label):
    d, lo, hi = bootstrap_mean_ci(np.asarray(a) - np.asarray(b), groups=groups)
    return {"comparison": label, "diff": round(d, 5), "CI (cluster)": f"[{lo:+.5f}, {hi:+.5f}]",
            "verdict": verdict(lo, hi)}


def describe_snapshots(joined):
    """Empirical check that the non-`C` columns really are an earlier, less
    informed price: an earlier line should carry a FATTER margin (books tighten
    as they take sharp money) and score WORSE."""
    rows = []
    for b in ["pin_open", "pin_close", "b365_open", "b365_close"]:
        sub = joined[joined[f"{b}_h"].notna()]
        r = rps_hda(sub[[f"{b}_h", f"{b}_d", f"{b}_a"]].to_numpy(), sub.result)
        rows.append({"book": b, "n": len(sub), "overround": f"{sub[f'overround_{b}'].mean():.2%}",
                     "pooled_rps": round(r.mean(), 5)})
    return pd.DataFrame(rows)


def main():
    both, joined = build()
    g = both.window.to_numpy()
    print(f"n = {len(both)} matches carrying both Pinnacle snapshots\n")

    print("=== 1. Are the non-`C` columns really an earlier, less-informed price? ===")
    print(describe_snapshots(joined).to_string(index=False))

    print("\n=== 2. The late-information premium (paired, same matches) ===")
    print(pd.DataFrame([_cmp(both.r_close, both.r_open, g, "Pinnacle closing - opening")]).to_string(index=False))
    premium = (both.r_open - both.r_close).mean()
    print(f"    => closing is sharper than opening by {premium:.5f} RPS")

    print("\n=== 3. Model gap vs each snapshot ===")
    rows = []
    for arm in ARMS:
        for snap, col in [("opening", "r_open"), ("closing", "r_close")]:
            rows.append({"arm": arm, **_cmp(both[f"r_{arm}"], both[col], g, f"vs {snap}")})
    gaps = pd.DataFrame(rows)
    print(gaps.to_string(index=False))

    print("\n    share of each arm's closing-gap that is late information:")
    for arm in ARMS:
        vs_close = gaps[(gaps.arm == arm) & (gaps.comparison == "vs closing")]["diff"].iloc[0]
        print(f"      {arm:14s} {premium/vs_close:.0%}  (premium {premium:.5f} / closing-gap {vs_close:.5f})")

    print("\n=== 4. Where the market's open->close sharpening actually happens ===")
    rows = [{"subset": lbl, **_cmp(s.r_close, s.r_open, s.window.to_numpy(), "closing - opening"), "n": len(s)}
            for lbl, s in [("upset", both[both.upset]), ("no upset", both[~both.upset])]]
    print(pd.DataFrame(rows)[["subset", "n", "diff", "CI (cluster)"]].to_string(index=False))

    print("\n=== 5. Model vs opening, split by how far the line moved ===")
    q = both.line_move
    print(f"    |move| in P(home): mean {q.mean():.4f}  median {q.median():.4f}  90th pct {q.quantile(0.9):.4f}")
    rows = []
    for lbl, s in [("top 25% line movement", both[q >= q.quantile(0.75)]),
                   ("middle 50%", both[(q > q.quantile(0.25)) & (q < q.quantile(0.75))]),
                   ("bottom 25% line movement", both[q <= q.quantile(0.25)])]:
        r = _cmp(s.r_L5_xgchannel, s.r_open, s.window.to_numpy(), "L5 - opening")
        r["n"] = len(s)
        r["mkt close-open"] = round((s.r_close - s.r_open).mean(), 5)
        rows.append(r)
    print(pd.DataFrame(rows)[["n", "diff", "CI (cluster)", "verdict", "mkt close-open"]]
          .assign(subset=[r_["subset"] if "subset" in r_ else lbl for r_, lbl in
                          zip(rows, ["top 25% line movement", "middle 50%", "bottom 25% line movement"])])
          .set_index("subset").to_string())

    print("\n=== 6. L5 vs OPENING, by season ===")
    rows = [{"season": s, "n": len(gg), **_cmp(gg.r_L5_xgchannel, gg.r_open, gg.window.to_numpy(), "L5 - opening")}
            for s, gg in both.groupby("season")]
    print(pd.DataFrame(rows)[["season", "n", "diff", "CI (cluster)", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
