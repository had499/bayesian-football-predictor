"""WP011 diagnostics: does squad instability explain where the model loses?

Design note on power (the one decision that matters here): the obvious test —
regress realised model error on squad instability — is badly underpowered,
because per-match RPS difference is dominated by outcome noise. That's part of
why rest days came back at corr≈0.007 in WP009. So the PRIMARY test runs on
DISAGREEMENT with the market (|p_model - p_market|), which doesn't depend on
how the match turned out, and chains onto WP009's already-established result
that disagreement predicts losing. The realised-RPS version is reported too,
flagged as the weaker instrument.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WP007 = HERE.parent / "wp007_structural_fixes"
WP008 = HERE.parent / "wp008_market_benchmark"
WP010 = HERE.parent / "wp010_opening_vs_closing"
for p in (WP007, WP008, WP010):
    sys.path.insert(0, str(p))
from analysis import bootstrap_mean_ci, rps_hda, verdict  # noqa: E402
from market_analysis import B365, PIN, join_market, load_wp007_matches  # noqa: E402

PIN_OPEN = ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"]
BOOKS = {"pin_close": PIN, "pin_open": PIN_OPEN}
ARM = "L5_xgchannel"
KEY = ["window", "round", "home", "away"]


def build_base():
    """Per-match frame: model probabilities, both Pinnacle snapshots, RPS, line move."""
    M = load_wp007_matches([ARM, "L0_baseline"])
    odds = pd.read_pickle(WP008 / "odds_raw.pkl")
    joined, _, _ = join_market(M[M.arm == "L0_baseline"], odds, books=BOOKS)
    both = joined[joined[["pin_close_h", "pin_open_h"]].notna().all(axis=1)].copy()

    Ma = M[M.arm == ARM].set_index(KEY).loc[both.set_index(KEY).index].reset_index()
    for c in ("p_home", "p_draw", "p_away"):
        both[c] = Ma[c].to_numpy()

    both["r_model"] = rps_hda(both[["p_home", "p_draw", "p_away"]].to_numpy(), both.result)
    both["r_open"] = rps_hda(both[["pin_open_h", "pin_open_d", "pin_open_a"]].to_numpy(), both.result)
    both["model_worse"] = both.r_model - both.r_open          # vs the FAIR (pre-close) benchmark
    both["disagreement"] = (both.p_home - both.pin_open_h).abs()
    both["line_move"] = (both.pin_close_h - both.pin_open_h).abs()

    # "supremacy" = P(home) - P(away); the residual is how much more the MODEL
    # favours the home side than the market does, at the same information state.
    both["sup_model"] = both.p_home - both.p_away
    both["sup_market"] = both.pin_open_h - both.pin_open_a
    both["sup_residual"] = both.sup_model - both.sup_market

    fav_home = both.pin_close_h >= both.pin_close_a
    both["favourite"] = np.where(fav_home, "home", "away")
    both["upset"] = np.where(fav_home, both.result != "H", both.result != "A")
    return both


def boot_ols(y, X, groups, n_boot=2000, seed=0):
    """Cluster-bootstrapped OLS slopes. X: (n, k) WITHOUT intercept (added here).
    Returns {col: (coef, lo, hi)} — clustered by CV window, same convention as
    every other CI in this project."""
    y = np.asarray(y, float)
    Xd = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    ok = np.isfinite(y) & np.isfinite(Xd).all(1)
    y, Xd, groups = y[ok], Xd[ok], np.asarray(groups)[ok]

    def fit(yy, XX):
        return np.linalg.lstsq(XX, yy, rcond=None)[0]

    base = fit(y, Xd)
    ids = np.unique(groups)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        try:
            draws.append(fit(y[idx], Xd[idx]))
        except np.linalg.LinAlgError:
            continue
    D = np.array(draws)
    return base, np.percentile(D, 2.5, axis=0), np.percentile(D, 97.5, axis=0), int(ok.sum())


def report_ols(y, X, names, groups, label):
    b, lo, hi, n = boot_ols(y, X, groups)
    rows = []
    for i, nm in enumerate(["intercept"] + list(names)):
        rows.append({"term": nm, "coef": round(b[i], 5),
                     "CI (cluster)": f"[{lo[i]:+.5f}, {hi[i]:+.5f}]",
                     "verdict": "—" if nm == "intercept" else verdict(lo[i], hi[i], lower_is_better=False)})
    print(f"\n{label}  (n={n})")
    print(pd.DataFrame(rows).to_string(index=False))
    return b, lo, hi


def bucket(df, col, y, bins, labels=None):
    b = pd.cut(df[col], bins=bins, labels=labels)
    rows = []
    for k, g in df.groupby(b, observed=True):
        m, lo, hi = bootstrap_mean_ci(g[y].to_numpy(), groups=g.window.to_numpy())
        rows.append({col: str(k), "n": len(g), f"mean_{y}": round(m, 5),
                     "CI (cluster)": f"[{lo:+.5f}, {hi:+.5f}]"})
    return pd.DataFrame(rows)
