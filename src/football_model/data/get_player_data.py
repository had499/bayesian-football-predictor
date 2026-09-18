"""Per-player match-log acquisition from Understat (WP007/WP008).

Two-step discovery, both via `understatapi` (already a project dependency —
`get_data.py` uses its `league` endpoint; this uses `team` and `player`):

1. `team.get_player_data(team, season)` for every (team, season) already in
   the existing dataset — gives each squad's player IDs plus which
   team/season each ID was registered to (needed below to resolve which
   side of `h_team`/`a_team` a given match-log row belongs to).
2. `player.get_match_data(player_id)`, once per UNIQUE player ID (it returns
   that player's entire Understat-tracked history in one call, not just one
   team/season — confirmed in WP007) — union of squads across teams/seasons
   overlaps heavily (a player usually stays on a team for several seasons),
   so this is far fewer calls than one-per-team-season would be.

No rate limiting is built into `understatapi` (confirmed in WP007) — this
module adds its own delay between requests.
"""
from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
from understatapi import UnderstatClient
from understatapi.exceptions import InvalidPlayer, InvalidTeam

# Numeric fields understatapi returns as strings on every payload — cast
# explicitly rather than relying on pandas' dtype inference, which would
# silently leave them as strings if a fetch hiccups mid-column.
_NUMERIC_MATCH_FIELDS = [
    "goals", "shots", "xG", "time", "h_goals", "a_goals", "xA",
    "assists", "key_passes", "npg", "npxG", "xGChain", "xGBuildup",
]


def _polite_sleep(delay: float) -> None:
    if delay > 0:
        time.sleep(delay)


def discover_squads(
    teams: Iterable[str],
    seasons: Iterable[str],
    client: UnderstatClient | None = None,
    delay: float = 0.3,
) -> pd.DataFrame:
    """One row per (team, season, player) the squad had registered.

    `teams` must be Understat's own team-name format with underscores for
    spaces (e.g. "Manchester_City") — the same format `get_data.py` already
    uses for the `league` endpoint's team filtering elsewhere in this
    pipeline. Missing/invalid (team, season) combinations (e.g. a team not
    in the league that season) are skipped, not fatal.
    """
    client = client or UnderstatClient()
    rows = []
    for season in seasons:
        for team in teams:
            try:
                squad = client.team(team=team).get_player_data(season=str(season))
            except InvalidTeam:
                continue
            for p in squad:
                rows.append({
                    "player_id": p["id"],
                    "player_name": p["player_name"],
                    "team_title": p["team_title"],  # Understat's own full name, matches df_cv's team_long
                    "season": str(season),
                })
            _polite_sleep(delay)
    return pd.DataFrame(rows)


def fetch_player_match_logs(
    player_ids: Iterable[str],
    client: UnderstatClient | None = None,
    delay: float = 0.3,
) -> pd.DataFrame:
    """One row per (player, match) — the actual per-player match-level data.

    `player_ids` should already be de-duplicated by the caller (one fetch
    per unique player, not per team-season they appeared in) — that
    de-duplication is the whole reason `discover_squads` and this are split
    into two functions instead of one.
    """
    client = client or UnderstatClient()
    rows = []
    for pid in set(player_ids):
        try:
            matches = client.player(player=str(pid)).get_match_data()
        except InvalidPlayer:
            continue
        for m in matches:
            row = dict(m)
            row["player_id"] = str(pid)
            rows.append(row)
        _polite_sleep(delay)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in _NUMERIC_MATCH_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["started"] = df["position"] != "Sub"  # confirmed in WP007/WP008: mean 85 vs 25 minutes
    return df


def resolve_player_team(
    squads: pd.DataFrame, match_logs: pd.DataFrame
) -> pd.DataFrame:
    """Attach a `team`/`opponent` column to each match-log row, resolved
    against that player's actual squad registrations — not guessed from
    `h_team`/`a_team` naming alone, since a player's own team isn't
    explicitly flagged on the per-match payload (checked in WP007/WP008;
    it's genuinely absent). Rows where neither `h_team` nor `a_team`
    matches any of the player's known (team, season) registrations are
    dropped — most commonly a player's history at a club outside this
    project's tracked teams/seasons (e.g. a foreign club before their move),
    which the eventual lineup feature has no use for anyway.

    Resolved per (player, season), not per player alone — a player who
    transfers between two of this project's tracked clubs must not have
    their squad history collapsed across seasons, or a later fixture
    against their old club could resolve to the wrong side. (Still a known,
    accepted limitation: a same-season transfer immediately followed by a
    fixture against the old club, within that same season, before the next
    squad-list update, is genuinely ambiguous from this data alone — rare
    enough not to be worth more machinery for v1.)
    """
    known = squads.groupby(["player_id", "season"])["team_title"].apply(set).to_dict()

    def _team_and_opponent(row):
        teams_for_player = known.get((row["player_id"], str(row["season"])), set())
        if row["h_team"] in teams_for_player:
            return row["h_team"], row["a_team"]
        if row["a_team"] in teams_for_player:
            return row["a_team"], row["h_team"]
        return None, None

    resolved = match_logs.copy()
    resolved[["team", "opponent"]] = resolved.apply(
        lambda r: pd.Series(_team_and_opponent(r)), axis=1
    )
    return resolved.dropna(subset=["team"]).reset_index(drop=True)


def get_player_match_data(
    teams: Iterable[str],
    seasons: Iterable[str],
    client: UnderstatClient | None = None,
    delay: float = 0.3,
) -> pd.DataFrame:
    """End-to-end: discover squads, fetch each unique player's full match
    log once, resolve which side of each match they were actually on.
    Returns one row per (player, match) with `team`/`opponent` resolved.
    """
    client = client or UnderstatClient()
    squads = discover_squads(teams, seasons, client=client, delay=delay)
    if squads.empty:
        return squads
    logs = fetch_player_match_logs(squads["player_id"].unique(), client=client, delay=delay)
    if logs.empty:
        return logs
    return resolve_player_team(squads, logs)
