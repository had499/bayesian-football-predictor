"""Betting-market evaluation helpers (WP012).

Everything here is deliberately small, pure and numpy-only so it can be unit
tested against hand-computed cases and against simulated markets whose true
edge is known. That matters more here than anywhere else in the project: a
bug in bet settlement or in the bootstrap does not crash, it silently
manufactures (or hides) an "edge".

Conventions
-----------
* Outcome axis is always ``(H, D, A)``: a probability/odds matrix has shape
  ``(n_matches, 3)`` and an outcome one-hot has the same shape.
* Odds are decimal, as football-data.co.uk publishes them. A winning unit
  stake returns ``odds - 1`` profit, a losing one ``-1``.
* "Edge" means expected-value edge, ``odds * p_fair - 1``, not a difference
  of probabilities. It is the number a bettor actually cares about.
* Bet-level results are clustered by MATCH in every bootstrap: a match can
  contribute up to three bets (one per outcome) that are not independent.
* Column naming follows football-data.co.uk: a book prefix plus ``H/D/A``,
  e.g. ``PSCH`` = Pinnacle closing home. See ``BOOKS``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")

# name -> (opening-ish prefix, closing prefix) in football-data.co.uk columns.
BOOKS = {
    "pinnacle": ("PS", "PSC"),
    "bet365": ("B365", "B365C"),
    "average": ("Avg", "AvgC"),
    "max": ("Max", "MaxC"),
}

# WP003's crosswalk from Understat's team codes (as used in df_cv) to
# football-data.co.uk team names.
CODE_TO_FD = {
    "ARS": "Arsenal", "AVL": "Aston Villa", "BOU": "Bournemouth", "BRE": "Brentford",
    "BRI": "Brighton", "BUR": "Burnley", "CHE": "Chelsea", "CRY": "Crystal Palace",
    "EVE": "Everton", "FLH": "Fulham", "IPS": "Ipswich", "LED": "Leeds",
    "LEI": "Leicester", "LIV": "Liverpool", "LUT": "Luton", "MCI": "Man City",
    "MUN": "Man United", "NEW": "Newcastle", "NOR": "Norwich", "NOT": "Nott'm Forest",
    "SHE": "Sheffield United", "SOU": "Southampton", "SUN": "Sunderland",
    "TOT": "Tottenham", "WAT": "Watford", "WBA": "West Brom", "WHU": "West Ham",
    "WOL": "Wolves",
}


# --------------------------------------------------------------------------
# Odds and outcomes
# --------------------------------------------------------------------------

def odds_matrix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """(n, 3) decimal odds for one book/time, e.g. prefix='PSC' -> PSCH/PSCD/PSCA."""
    return df[[f"{prefix}{o}" for o in OUTCOMES]].to_numpy(float)


def devig(odds: np.ndarray) -> np.ndarray:
    """Proportional de-vig: normalise implied probabilities to sum to 1.
    Same method WP003 used as its default."""
    inv = 1.0 / np.asarray(odds, float)
    return inv / inv.sum(axis=1, keepdims=True)


def outcome_onehot(ftr) -> np.ndarray:
    """(n, 3) one-hot from a sequence of 'H'/'D'/'A' full-time results."""
    ftr = np.asarray(ftr)
    if not np.isin(ftr, OUTCOMES).all():
        raise ValueError("full-time result must be one of H/D/A")
    return np.stack([ftr == o for o in OUTCOMES], axis=1).astype(float)


def rps(probs: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    """Per-match ranked probability score for ordered (H, D, A) outcomes.
    Identical to the `rps_row` the WP003/WP010/WP011 notebooks used."""
    cp = np.cumsum(probs, axis=1)[:, :2]
    ce = np.cumsum(onehot, axis=1)[:, :2]
    return 0.5 * ((cp - ce) ** 2).sum(axis=1)


def blend(p_a: np.ndarray, p_b: np.ndarray, w: float) -> np.ndarray:
    """(1 - w) * p_a + w * p_b."""
    return (1.0 - w) * np.asarray(p_a) + w * np.asarray(p_b)


# --------------------------------------------------------------------------
# Bootstraps and intervals
# --------------------------------------------------------------------------

def bootstrap_ci(values, n_boot: int = 5000, seed: int = 0):
    """(mean, lo, hi): percentile 95% CI of the mean, resampling rows."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def ratio_bootstrap(num, den, n_boot: int = 5000, seed: int = 0):
    """(ratio, lo, hi) for sum(num)/sum(den), resampling MATCHES.

    Used for ROI (num = profit per match, den = stake per match) and for
    mean CLV per bet (num = summed CLV per match, den = bets per match).
    Returns NaNs if there is nothing staked at all. Resamples that happen to
    draw zero stake are dropped from the percentile."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if den.sum() == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    d = den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num[idx].sum(axis=1) / d
    r = r[d > 0]
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def wilson_interval(k: float, n: float, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def half_masks(dates) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for the first and second half of `dates` split at the
    median date — the robustness split used to guard the primary tests."""
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    cut = d.sort_values().iloc[len(d) // 2]
    first = (d < cut).to_numpy()
    return first, ~first


# --------------------------------------------------------------------------
# Bets
# --------------------------------------------------------------------------

def edge(p_fair: np.ndarray, book_odds: np.ndarray) -> np.ndarray:
    """Expected-value edge of a unit bet at `book_odds` if `p_fair` is right."""
    return book_odds * p_fair - 1.0


def settle(bet_mask: np.ndarray, book_odds: np.ndarray, onehot: np.ndarray):
    """Per-match (profit, stake) for unit stakes on every masked outcome."""
    per_bet = np.where(onehot == 1, book_odds - 1.0, -1.0)
    return (per_bet * bet_mask).sum(axis=1), bet_mask.sum(axis=1).astype(float)


def roi_table(p_fair, book_odds, onehot, taus, n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Flat-stake value betting: for each threshold tau, bet every outcome
    whose `edge(p_fair, book_odds) > tau`, settle at `book_odds`.

    `tau = -inf` is the "bet everything" control, whose ROI should sit near
    minus the book's margin. `claimed_edge` is the mean edge the fair
    probabilities said the selected bets had — compare it to realised ROI."""
    e = edge(p_fair, book_odds)
    rows = []
    for tau in taus:
        mask = e > tau
        profit, stake = settle(mask, book_odds, onehot)
        roi, lo, hi = ratio_bootstrap(profit, stake, n_boot, seed)
        n_bets = int(mask.sum())
        rows.append({
            "tau": tau, "n_bets": n_bets, "n_matches": int((stake > 0).sum()),
            "roi": roi, "lo": lo, "hi": hi,
            "claimed_edge": float(e[mask].mean()) if n_bets else np.nan,
            "hit_rate": float(onehot[mask].mean()) if n_bets else np.nan,
        })
    return pd.DataFrame(rows)


def odds_band_table(book_odds, onehot, edges, n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Blind flat-stake betting on every outcome whose odds fall in each band
    [edges[i], edges[i+1]). The favourite-longshot bias shows up as ROI
    differing systematically between the short and long bands."""
    rows = []
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        mask = (book_odds >= lo_e) & (book_odds < hi_e)
        profit, stake = settle(mask, book_odds, onehot)
        roi, lo, hi = ratio_bootstrap(profit, stake, n_boot, seed)
        n_bets = int(mask.sum())
        rows.append({
            "band": f"[{lo_e:g}, {hi_e:g})", "n_bets": n_bets,
            "implied": float((1.0 / book_odds[mask]).mean()) if n_bets else np.nan,
            "hit_rate": float(onehot[mask].mean()) if n_bets else np.nan,
            "roi": roi, "lo": lo, "hi": hi,
        })
    return pd.DataFrame(rows)


def clv_table(open_odds, p_close_fair, mask, onehot=None, n_boot: int = 5000, seed: int = 0):
    """Closing line value of the masked bets placed at `open_odds`:
    mean of `open_odds * p_close_fair - 1` per bet (positive = you got a
    better price than the eventual sharp close), clustered by match. Also
    returns the realised ROI of the same bets when `onehot` is given."""
    clv = edge(p_close_fair, open_odds)
    stake = mask.sum(axis=1).astype(float)
    num = (clv * mask).sum(axis=1)
    mean, lo, hi = ratio_bootstrap(num, stake, n_boot, seed)
    out = {"n_bets": int(mask.sum()), "clv": mean, "clv_lo": lo, "clv_hi": hi}
    if onehot is not None:
        profit, _ = settle(mask, open_odds, onehot)
        roi, rlo, rhi = ratio_bootstrap(profit, stake, n_boot, seed)
        out.update({"roi": roi, "roi_lo": rlo, "roi_hi": rhi})
    return out


def calibration_table(p, hit, edges) -> pd.DataFrame:
    """Reliability table over pooled outcome probabilities. `p` and `hit`
    are same-shaped arrays (any shape); rows are bins of `p`."""
    p, hit = np.asarray(p).ravel(), np.asarray(hit).ravel()
    rows = []
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        m = (p >= lo_e) & (p < hi_e if hi_e < edges[-1] else p <= hi_e)
        n = int(m.sum())
        lo, hi = wilson_interval(hit[m].sum(), n)
        rows.append({
            "bin": f"[{lo_e:g}, {hi_e:g}]", "n": n,
            "mean_p": float(p[m].mean()) if n else np.nan,
            "rate": float(hit[m].mean()) if n else np.nan, "lo": lo, "hi": hi,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Does the model add information beyond the market?
# --------------------------------------------------------------------------

def logodds_features(p: np.ndarray) -> np.ndarray:
    """(n, 2) log(pH/pD), log(pA/pD): the two free log-odds of a 3-way market."""
    p = np.asarray(p, float)
    return np.column_stack([np.log(p[:, 0] / p[:, 1]), np.log(p[:, 2] / p[:, 1])])


def leave_group_out_logloss(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float = 1e6) -> np.ndarray:
    """Per-match log loss of a multinomial logistic regression, predicting each
    group's rows from a model fit on every OTHER group (leave-one-window-out,
    the same cross-fitting WP004 used so no match's prediction has seen its
    own outcome). `C` is large so the fit is effectively unpenalised."""
    from sklearn.linear_model import LogisticRegression

    X, y, groups = np.asarray(X, float), np.asarray(y), np.asarray(groups)
    loss = np.empty(len(y))
    for g in np.unique(groups):
        test = groups == g
        clf = LogisticRegression(C=C, max_iter=2000).fit(X[~test], y[~test])
        proba = clf.predict_proba(X[test])
        col = {c: i for i, c in enumerate(clf.classes_)}
        loss[test] = -np.log(proba[np.arange(test.sum()), [col[c] for c in y[test]]])
    return loss


# --------------------------------------------------------------------------
# Joining model predictions to odds
# --------------------------------------------------------------------------

def model_fixtures(df_cv: pd.DataFrame, windows: list, ckpt: dict) -> pd.DataFrame:
    """One row per held-out EPL match in `ckpt`, with football-data team
    names, goals, the model's lambdas/rho, and its (H, D, A) probabilities.

    Same alignment WP003/WP011 used: within each window the test-round home
    rows of `df_cv`, sorted by datetime, line up one-to-one with that
    window's stored match predictions. `join_odds` verifies this against the
    real scores rather than trusting it."""
    from football_model.model.predict import dc_outcome_probs

    df_sorted = df_cv.sort_values("datetime").reset_index(drop=True)
    preds = ckpt["cv_match_predictions"]
    rows = []
    for w in sorted({m["window"] for m in preds}):
        win = windows[w - 1]
        sel = df_sorted[(df_sorted["is_home"] == 1)
                        & (df_sorted["round"] >= win["test_start"])
                        & (df_sorted["round"] <= win["test_end"])]
        wp = [m for m in preds if m["window"] == w]
        if len(sel) != len(wp):
            raise ValueError(f"window {w}: {len(sel)} fixtures but {len(wp)} stored predictions")
        for (_, r), m in zip(sel.iterrows(), wp):
            rows.append({
                "window": w, "date": pd.Timestamp(r["datetime"]).normalize(),
                "home_fd": CODE_TO_FD[r["team"]], "away_fd": CODE_TO_FD[r["opp_team"]],
                "goals_home": m["goals_home"], "goals_away": m["goals_away"],
                "lambda_home": m["lambda_home"], "lambda_away": m["lambda_away"],
                "rho_dc": m.get("rho_dc"),
            })
    df = pd.DataFrame(rows)
    probs = np.array([dc_outcome_probs(r.lambda_home, r.lambda_away, rho=r.rho_dc)
                      for r in df.itertuples()])
    df[["p_home_model", "p_draw_model", "p_away_model"]] = probs
    return df


def join_odds(fixtures: pd.DataFrame, odds_raw: pd.DataFrame, required_cols=()) -> pd.DataFrame:
    """Inner-join fixtures to football-data odds on (date, home, away), drop
    rows missing any of `required_cols`, and RAISE if any joined row's stored
    score disagrees with the odds file's full-time score (which would mean the
    fixture/prediction alignment is off and every downstream number is
    meaningless)."""
    j = fixtures.merge(odds_raw, left_on=["date", "home_fd", "away_fd"],
                       right_on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    bad = (j["goals_home"] != j["FTHG"]) | (j["goals_away"] != j["FTAG"])
    if bad.any():
        raise ValueError(f"{int(bad.sum())} joined matches have a score that disagrees with the odds file")
    if required_cols:
        j = j.dropna(subset=list(required_cols))
    return j.reset_index(drop=True)
