"""Fetch EPL results + bookmaker odds and cache locally.

football-data.co.uk itself (WP003/WP004's source) is blocked by network policy
on the machine this was built on (a corporate web-filtering proxy returns a
block page, not a real 503 — see this folder's README). This fetches the same
underlying data via a public GitHub mirror instead: AnishKhetani/premier-
league-data (https://github.com/AnishKhetani/premier-league-data), which
states it sources from football-data.co.uk and preserves its odds columns
(PSCH/PSCD/PSCA etc, renamed to e.g. pinnacle_1x2_home_close) verbatim. Verified
before use: team-name spelling matches Understat's exactly (both ultimately
trace to football-data.co.uk), and Pinnacle/Bet365/market-average/market-max
are all present with full coverage for 2020-21 through 2024-25.

If football-data.co.uk becomes reachable again, prefer fetching from there
directly (see wp003/wp004's notebooks) — this mirror is a workaround, not a
long-term replacement, and is one more hop from the primary source than ideal.

    python fetch_odds.py   # writes odds_raw.pkl (results x odds, EPL only)
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MIRROR = "https://raw.githubusercontent.com/AnishKhetani/premier-league-data/main/data/processed"
OUT = HERE / "odds_raw.pkl"

PIN = ["pinnacle_1x2_home_close", "pinnacle_1x2_draw_close", "pinnacle_1x2_away_close"]
B365 = ["bet365_1x2_home_close", "bet365_1x2_draw_close", "bet365_1x2_away_close"]
AVG = ["market_avg_1x2_home", "market_avg_1x2_draw", "market_avg_1x2_away"]
MAX = ["market_max_1x2_home", "market_max_1x2_draw", "market_max_1x2_away"]

# Pre-closing ("opening") snapshots — the same books without the `C` infix.
# WP010 benchmarks against these: a model predicting days ahead cannot see the
# team news that moves a line in the final hour, so the closing line is an
# unfairly hard yardstick for it. The mirror's SPEC only documents that `C`
# means closing and doesn't say when the non-`C` snapshot is taken, so WP010
# establishes what they are empirically (higher overround + worse RPS than
# closing = an earlier, less-informed price) rather than trusting a label.
PIN_OPEN = ["pinnacle_1x2_home", "pinnacle_1x2_draw", "pinnacle_1x2_away"]
B365_OPEN = ["bet365_1x2_home", "bet365_1x2_draw", "bet365_1x2_away"]
ODDS_COLS = PIN + B365 + AVG + MAX + PIN_OPEN + B365_OPEN


def fetch():
    results = pd.read_csv(f"{MIRROR}/results.csv", low_memory=False)
    odds = pd.read_csv(f"{MIRROR}/results_with_odds.csv", low_memory=False)
    missing = [c for c in ODDS_COLS if c not in odds.columns]
    if missing:
        raise ValueError(f"mirror schema changed — missing columns {missing}")
    df = results.merge(odds[["match_id", *ODDS_COLS]], on="match_id", how="left")
    # EPL only — the mirror covers multiple English divisions in one file via season_code
    # (results.csv has no division column; match_id/season alone don't disambiguate a
    # second division, so restrict via team-set size: the top flight had 22 teams
    # 1992-93 through 1994-95, 20 every season since — cross-checked below rather
    # than assumed, so a real division-mixing bug still gets caught).
    df["date"] = pd.to_datetime(df["date"])
    per_season_teams = df.groupby("season").apply(
        lambda g: len(set(g.home_team) | set(g.away_team)), include_groups=False
    )
    bad = per_season_teams[~per_season_teams.isin([20, 22])]
    if len(bad):
        raise ValueError(f"expected 20 (or 22, pre-1995) teams/season (EPL) — got {bad.to_dict()}; "
                         "mirror may include another division, filter before using")
    return df


if __name__ == "__main__":
    df = fetch()
    df.to_pickle(OUT)
    print(f"wrote {OUT}: {len(df)} matches, seasons {df.season.min()}..{df.season.max()}")
    for name, cols in [("pinnacle", PIN), ("bet365", B365), ("market_avg", AVG), ("market_max", MAX),
                       ("pinnacle_open", PIN_OPEN), ("bet365_open", B365_OPEN)]:
        n = df[cols[0]].notna().sum()
        print(f"  {name}: {n}/{len(df)} matches have odds")
