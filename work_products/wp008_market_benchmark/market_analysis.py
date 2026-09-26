"""WP008 analysis helpers — joins WP007's model predictions to the odds
cached by fetch_odds.py and scores against the market.

Statistics (bootstrap CI, RPS, de-vig) are WP007's tested implementations,
imported rather than re-derived — same reasoning as WP004 reusing WP003's
odds cache: one implementation, not two that can quietly drift apart.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WP008 = Path(__file__).resolve().parent
WP007 = WP008.parent / "wp007_structural_fixes"
sys.path.insert(0, str(WP007))
from analysis import (  # noqa: E402
    CODE_TO_FD,
    attach_meta,
    bootstrap_mean_ci,
    complete_windows,
    devig_proportional,
    devig_shin,
    load_checkpoints,
    paired_diff,
    reliability_table,
    rps_hda,
    verdict,
    wide,
)

PIN = ["pinnacle_1x2_home_close", "pinnacle_1x2_draw_close", "pinnacle_1x2_away_close"]
B365 = ["bet365_1x2_home_close", "bet365_1x2_draw_close", "bet365_1x2_away_close"]
AVG = ["market_avg_1x2_home", "market_avg_1x2_draw", "market_avg_1x2_away"]
MAX = ["market_max_1x2_home", "market_max_1x2_draw", "market_max_1x2_away"]
BOOKS = {"pinnacle": PIN, "bet365": B365, "market_avg": AVG}  # MAX excluded from RPS bench (see README)


def load_wp007_matches(arms):
    """WP007 per-match predictions for `arms`, restricted to windows every one
    of them completed, with date/season/result attached."""
    df_cv = pd.read_pickle(WP007 / "cv_shared_data.pkl")["df_cv"]
    matches, _ = load_checkpoints(WP007 / "checkpoints")
    matches = attach_meta(matches, df_cv)
    have = complete_windows(matches, arms)
    return matches[matches.arm.isin(arms) & matches.window.isin(have)].copy()


def add_devig_probs(df, cols, prefix):
    """Add {prefix}_h/_d/_a (proportional) and {prefix}_shin_h/_d/_a columns
    from raw decimal odds `cols`, leaving NaN where odds are missing."""
    ok = df[cols].notna().all(axis=1).to_numpy()
    P = np.full((len(df), 3), np.nan)
    Ps = np.full((len(df), 3), np.nan)
    if ok.any():
        arr = df.loc[ok, cols].to_numpy(float)
        P[ok] = np.array([devig_proportional(r) for r in arr])
        Ps[ok] = np.array([devig_shin(r) for r in arr])
    df[[f"{prefix}_h", f"{prefix}_d", f"{prefix}_a"]] = P
    df[[f"{prefix}_shin_h", f"{prefix}_shin_d", f"{prefix}_shin_a"]] = Ps
    df[f"overround_{prefix}"] = np.where(
        ok, (1 / df[cols[0]] + 1 / df[cols[1]] + 1 / df[cols[2]]) - 1.0, np.nan
    )
    return df


def join_market(model_matches, odds_raw, books=None):
    """model_matches: WP007 per-match rows (any arm(s), needs date/home/away/
    result/window/round/goals_home/goals_away). `home`/`away` are Understat's
    3-letter codes (e.g. "ARS") — mapped through CODE_TO_FD to football-data-
    style full names ("Arsenal") before joining, same map WP003/WP007 use.

    `books`: {name: [home_col, draw_col, away_col]} to de-vig; defaults to
    BOOKS (the closing lines WP008 benchmarks). WP010 passes its own dict to
    add the pre-closing snapshots — a parameter rather than a mutated global
    so WP008's own results can't shift underneath it.

    Returns (joined, unmatched, mismatched): `joined` is one row per (window,
    round, home, away) — arm-independent — with de-vigged probabilities for
    every book in `books`, restricted to rows whose score MATCHES the odds
    source's own fthg/ftag (a mismatch means the join found the wrong
    fixture, e.g. a postponed/rescheduled date — better to drop it than
    silently mis-score).
    """
    ident = (
        model_matches[["window", "round", "home", "away", "date", "season", "result", "goals_home", "goals_away"]]
        .drop_duplicates(subset=["window", "round", "home", "away"])
        .assign(home_fd=lambda d: d.home.map(CODE_TO_FD), away_fd=lambda d: d.away.map(CODE_TO_FD))
        .copy()
    )
    missing_code = ident[ident.home_fd.isna() | ident.away_fd.isna()]
    if len(missing_code):
        raise ValueError(f"CODE_TO_FD is missing a team code — {missing_code[['home','away']].drop_duplicates()}")

    # odds_raw has its own 'season' (football-data.co.uk's "2020-21" style,
    # vs. model_matches' Understat "2020") — keep only what's needed from it
    # to avoid a season_x/season_y collision.
    books = BOOKS if books is None else books
    odds_cols = ["date", "home_team", "away_team", "fthg", "ftag"] + sum(books.values(), [])
    j = ident.merge(odds_raw[odds_cols], left_on=["date", "home_fd", "away_fd"],
                    right_on=["date", "home_team", "away_team"], how="left", indicator=True)
    unmatched = j[j["_merge"] == "left_only"]
    j = j[j["_merge"] == "both"].drop(columns="_merge")

    score_ok = (j.goals_home == j.fthg) & (j.goals_away == j.ftag)
    mismatched = j[~score_ok]
    j = j[score_ok].copy()

    for book, cols in books.items():
        j = add_devig_probs(j, cols, book)
    return j, unmatched, mismatched


def book_rps(joined, book, result_col="result", shin=False):
    """(n,) RPS array + the row index (for grouping by window) for one book,
    restricted to rows where that book's odds are present."""
    suf = "_shin" if shin else ""
    cols = [f"{book}{suf}_h", f"{book}{suf}_d", f"{book}{suf}_a"]
    ok = joined[cols].notna().all(axis=1)
    sub = joined[ok]
    return rps_hda(sub[cols].to_numpy(), sub[result_col]), sub
