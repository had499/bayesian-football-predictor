"""Lineup-continuity features (WP013).

For each (team, match): how "usual" is today's starting XI, unit by unit?

A player's **start share** for a given team-match is the fraction of that
team's previous `window` league matches he STARTED (in any position —
formation changes move a player between slots, they do not change how
regular he is). A unit's **continuity** is the mean start share of today's
starters in that unit: near 1 = the usual players, low = a reshuffled unit.
Nothing about a player's quality is estimated, so unlike WP008's xG+xA
deviation this cannot merely restate "this is a good team".

Leakage boundary: a team-match's history is the team's own strictly earlier
matches (by that team's own date-sorted sequence), and today's starters come
from the confirmed starting XI (`started`), never from substitutes, whose
minutes are a consequence of the match being predicted (same reasoning as
`get_player_data.py`). Any row's value is therefore unaffected by later
matches — this is unit-tested by truncation.

Rows are NaN (not zero) when the value is not meaningful, so a downstream
model can decide how to treat them explicitly:
* fewer than `min_history` earlier matches in the window,
* the team's first `neutral_first_n_of_season` matches of a season (summer
  signings and departures make a low share there a squad change, not a
  disruption),
* a gap of more than `max_gap_days` since the team's previous match resets
  its history (e.g. relegated and later promoted back).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Understat lineup slots (the `position` field of a START; 'Sub' = unused/
# bench). One GK per team per match; verified on the WP009 player logs.
DEFENCE_SLOTS = frozenset({"GK", "DC", "DL", "DR"})
MIDFIELD_SLOTS = frozenset({"DMC", "DML", "DMR", "MC", "ML", "MR", "AMC", "AML", "AMR"})
ATTACK_SLOTS = frozenset({"FW", "FWL", "FWR"})

UNITS = {
    "defence": DEFENCE_SLOTS,
    "midfield": MIDFIELD_SLOTS,
    "attack": ATTACK_SLOTS,
    "all": None,  # every starter
}


def build_continuity_table(
    player_match_data: pd.DataFrame,
    window: int = 10,
    min_history: int = 5,
    max_gap_days: int = 150,
    neutral_first_n_of_season: int = 5,
) -> pd.DataFrame:
    """One row per (team, date): `continuity_<unit>` for each of `UNITS`, plus
    `n_history` (how many earlier matches the shares were computed from).

    `player_match_data` needs columns: `team`, `date`, `season`, `player_id`,
    `position`, `started` (the WP009 player-match log format). `team` uses the
    same full names as `df_cv['team_long']`, so the table joins on
    `(team, date)` exactly like WP008's `lineup_dev_table`."""
    starters = player_match_data.loc[player_match_data["started"]].copy()
    starters["date"] = pd.to_datetime(starters["date"]).dt.normalize()

    rows = []
    for team, tdf in starters.groupby("team", sort=False):
        matches = [
            (date, g)
            for date, g in tdf.sort_values("date").groupby("date", sort=True)
        ]
        season_of = {date: g["season"].iloc[0] for date, g in matches}
        season_count: dict = {}
        prev_lineups: list[tuple[pd.Timestamp, set]] = []  # (date, starter ids), oldest first

        for date, g in matches:
            season = season_of[date]
            season_count[season] = season_count.get(season, 0) + 1
            season_match_no = season_count[season]

            # A long gap since the previous match wipes the history.
            if prev_lineups and (date - prev_lineups[-1][0]).days > max_gap_days:
                prev_lineups = []
            history = [ids for _, ids in prev_lineups[-window:]]
            n_hist = len(history)

            row = {"team": team, "date": date, "n_history": n_hist}
            valid = n_hist >= min_history and season_match_no > neutral_first_n_of_season
            share = {}
            if valid:
                for pid in g["player_id"]:
                    share[pid] = sum(pid in ids for ids in history) / n_hist
            for unit, slots in UNITS.items():
                sel = g if slots is None else g[g["position"].isin(slots)]
                row[f"continuity_{unit}"] = (
                    float(np.mean([share[p] for p in sel["player_id"]])) if valid and len(sel) else np.nan
                )
            rows.append(row)
            prev_lineups.append((date, set(g["player_id"])))

    return pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)


# Fixed standardisation constants for the defence unit: mean and SD of
# `continuity_defence` over the 3,960 valid team-matches in 2020-2026 (WP013).
# Fixed numbers, not refit per window, so training and prediction always see
# the same transform; two scalars of look-ahead, immaterial for a covariate
# with SD 0.14.
DEFENCE_CONTINUITY_CENTER = 0.714
DEFENCE_CONTINUITY_SCALE = 0.140


def continuity_feature_table(
    table: pd.DataFrame,
    unit: str = "defence",
    center: float = DEFENCE_CONTINUITY_CENTER,
    scale: float = DEFENCE_CONTINUITY_SCALE,
) -> pd.DataFrame:
    """The table `prepare_model_data(continuity_table=...)` consumes: columns
    `team`, `date`, `continuity_z` = (continuity - center) / scale, with NaN
    (early history, season starts, gaps) mapped to 0 — the neutral, mean value —
    so an unmeasurable match contributes no effect instead of a missing value."""
    z = (table[f"continuity_{unit}"] - center) / scale
    return pd.DataFrame({"team": table["team"], "date": table["date"], "continuity_z": z.fillna(0.0)})
