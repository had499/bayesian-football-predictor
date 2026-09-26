"""WP009 diagnostics: what characterizes the matches where the model
disagrees most with Pinnacle? Reuses WP007's predictions and WP008's odds
join — no new fetches needed for this part.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WP007 = Path(__file__).resolve().parents[1] / "wp007_structural_fixes"
WP008 = Path(__file__).resolve().parents[1] / "wp008_market_benchmark"
sys.path.insert(0, str(WP007))
sys.path.insert(0, str(WP008))
from analysis import CODE_TO_FD, attach_meta, bootstrap_mean_ci, load_checkpoints, rps_hda  # noqa: E402
from market_analysis import BOOKS, join_market  # noqa: E402


def rest_days(df_cv):
    """Days since each row's team last played (any competition Understat has
    — league only, since that's all this dataset contains; a real limitation,
    see README). NaN for a team's first match in the dataset (no prior game
    to measure from) rather than a fabricated default.

    Keyed by (team, date), NOT (team, round): the ISO-week-based round
    numbering (add_rounds_to_data) legitimately lets a team play twice within
    one round slot around rearranged/midweek fixtures — e.g. Arsenal played
    both Southampton (Dec 3) and Stoke (Dec 6) 2014 under round 14 — so
    (team, round) is not a unique key and merging on it silently duplicates
    rows. `date` (from `datetime`, normalised) is unique per team per match.
    """
    d = df_cv.sort_values("datetime").copy()
    d["date"] = pd.to_datetime(d["datetime"]).dt.normalize()
    d["prev_dt"] = d.groupby("team")["datetime"].shift(1)
    d["rest_days"] = (d["datetime"] - d["prev_dt"]).dt.total_seconds() / 86400
    dup = d[d.duplicated(subset=["team", "date"], keep=False)]
    if len(dup):
        raise ValueError(f"(team, date) not unique — {len(dup)} rows, e.g.\n{dup.head()}")
    return d[["team", "opp_team", "date", "round", "season", "is_home", "rest_days"]]


def build_diagnostic_frame(arm="L5_xgchannel"):
    """One row per (Pinnacle-matched) WP007/WP008 test match: disagreement
    with Pinnacle, model/market error, and candidate explanatory columns."""
    df_cv = pd.read_pickle(WP007 / "cv_shared_data.pkl")["df_cv"]
    matches, _ = load_checkpoints(WP007 / "checkpoints")
    matches = attach_meta(matches, df_cv)
    M = matches[matches.arm == arm].copy()

    joined, _, _ = join_market(M, pd.read_pickle(WP008 / "odds_raw.pkl"))
    key = ["window", "round", "home", "away"]
    # join_market's own `ident` extraction doesn't carry p_home/p_draw/p_away
    # (it wasn't needed for WP008's purposes), so pull those back in with a
    # plain key merge — NOT `.set_index().loc[]`, which silently cartesian-
    # producted this the first time round (1,314 correct rows -> 2,285 wrong
    # ones) despite both sides having unique keys individually.
    assert M[key].duplicated().sum() == 0 and joined[key].duplicated().sum() == 0
    j = joined.merge(M[key + ["p_home", "p_draw", "p_away", "season_round"]], on=key,
                     how="left", validate="one_to_one")
    j = j.dropna(subset=["pinnacle_h"]).copy()

    j["disagreement"] = (j.p_home - j.pinnacle_h).abs()
    j["rps_model"] = rps_hda(j[["p_home", "p_draw", "p_away"]].to_numpy(), j.result)
    j["rps_pinnacle"] = rps_hda(j[["pinnacle_h", "pinnacle_d", "pinnacle_a"]].to_numpy(), j.result)
    j["model_worse"] = j.rps_model - j.rps_pinnacle  # >0: model lost this match to the market

    # candidate explanatory columns
    j["date"] = pd.to_datetime(j["date"]).dt.normalize()
    rest = rest_days(df_cv)
    rest_h = rest.rename(columns={"team": "home_code", "rest_days": "rest_home"})[["home_code", "date", "rest_home"]]
    rest_a = rest.rename(columns={"team": "away_code", "rest_days": "rest_away"})[["away_code", "date", "rest_away"]]
    n0 = len(j)
    j = j.merge(rest_h, left_on=["home", "date"], right_on=["home_code", "date"], how="left")
    j = j.merge(rest_a, left_on=["away", "date"], right_on=["away_code", "date"], how="left")
    assert len(j) == n0, f"rest-days merge changed row count {n0} -> {len(j)} — a join key isn't unique"
    j["rest_diff"] = j.rest_home - j.rest_away  # positive: home side is more rested

    seasons = sorted(df_cv.season.unique())
    steams = {s: set(g.team) | set(g.opp_team) for s, g in df_cv.groupby("season")}
    promoted = {s: steams[s] - steams[seasons[i - 1]] for i, s in enumerate(seasons) if i}
    j["home_promoted"] = [h in promoted.get(s, set()) for h, s in zip(j.home, j.season)]
    j["away_promoted"] = [a in promoted.get(s, set()) for a, s in zip(j.away, j.season)]
    j["either_promoted"] = j.home_promoted | j.away_promoted

    j["goal_margin"] = (j.goals_home - j.goals_away).abs()
    j["total_goals"] = j.goals_home + j.goals_away
    j["favourite"] = np.where(j.pinnacle_h > j.pinnacle_a, "home",
                              np.where(j.pinnacle_a > j.pinnacle_h, "away", "even"))
    j["upset"] = (
        ((j.favourite == "home") & (j.result != "H"))
        | ((j.favourite == "away") & (j.result != "A"))
    )
    j["pinnacle_fav_prob"] = j[["pinnacle_h", "pinnacle_a"]].max(axis=1)
    return j


def sharpen(P, T):
    """Temperature-scale a (n, 3) probability matrix: softmax(log(p)/T).
    T<1 sharpens (more confident), T>1 flattens, T=1 is a no-op."""
    logp = np.log(np.clip(P, 1e-12, None)) / T
    e = np.exp(logp - logp.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(P, y, bounds=(0.3, 3.0)):
    """Fit T by log-loss on (P, y) — y integer-coded 0/1/2 (H/D/A)."""
    from scipy.optimize import minimize_scalar

    def negll(T):
        Ps = sharpen(P, T)
        return -np.log(np.clip(Ps[np.arange(len(y)), y], 1e-12, None)).sum()

    return minimize_scalar(negll, bounds=bounds, method="bounded").x


def evaluate_temperature_scaling(df, n_folds=2, seed=0):
    """2-fold (default) cross-validated temperature scaling, split by CV
    WINDOW (not match) so a window's shared model fit never appears in both
    train and test. Returns a DataFrame with one row per held-out match:
    rps_before/rps_after/window/favourite/upset, for downstream slicing.
    """
    P = df[["p_home", "p_draw", "p_away"]].to_numpy()
    y = df.result.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    windows = df.window.to_numpy()

    rng = np.random.default_rng(seed)
    uniq_w = np.unique(windows)
    rng.shuffle(uniq_w)
    folds = np.array_split(uniq_w, n_folds)

    out = []
    for i in range(n_folds):
        test_w = set(folds[i])
        test_mask = np.isin(windows, list(test_w))
        train_mask = ~test_mask
        T = fit_temperature(P[train_mask], y[train_mask])
        P_sharp = sharpen(P[test_mask], T)
        idx = np.where(test_mask)[0]
        out.append(pd.DataFrame({
            "row": idx, "window": windows[idx], "T": T,
            "rps_before": rps_hda(P[test_mask], df.result.to_numpy()[test_mask]),
            "rps_after": rps_hda(P_sharp, df.result.to_numpy()[test_mask]),
            "favourite": df.favourite.to_numpy()[idx], "upset": df.upset.to_numpy()[idx],
        }))
    return pd.concat(out, ignore_index=True)


def bucket_summary(df, col, bins, labels=None):
    b = pd.cut(df[col], bins=bins, labels=labels)
    g = df.groupby(b, observed=True)
    out = g.agg(n=("disagreement", "size"), mean_disagreement=("disagreement", "mean"),
               mean_model_worse=("model_worse", "mean"))
    out["ci"] = [
        f"[{lo:+.4f}, {hi:+.4f}]"
        for lo, hi in (bootstrap_mean_ci(g.get_group(k)["model_worse"].to_numpy())[1:] for k in out.index)
    ]
    return out
