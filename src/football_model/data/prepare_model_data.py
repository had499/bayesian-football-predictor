from football_model.types.model_data import *
import numpy as np


def max_round_for_cutoff_date(df: pd.DataFrame, cutoff_date) -> int:
    """WP011: the highest `round` value among a league's own matches on or
    before `cutoff_date`.

    Multi-league training needs a `max_round` for leagues that aren't the
    one being evaluated (see prepare_multileague_data) — and those leagues'
    round numbers don't line up with the evaluated league's (different
    season lengths, different fixture calendars), so passing the same
    integer max_round to every league would either leak future matches into
    training for a shorter-season league or needlessly discard already-past
    matches for a longer one. Using a shared cutoff DATE instead and
    resolving it to each league's own max_round keeps every league's
    training data anchored to the same real-world point in time — the
    leakage-safety property that actually matters — regardless of how many
    rounds each competition plays.

    Strict `<=`: a match ON the cutoff date is included (matches
    prepare_model_data's own `df["round"] <= max_round` convention, which is
    also inclusive), a match after it is not.
    """
    cutoff = pd.Timestamp(cutoff_date)
    eligible = df.loc[pd.to_datetime(df["datetime"]) <= cutoff, "round"]
    if len(eligible) == 0:
        return 0
    return int(eligible.max())


def prepare_multileague_data(
    engineered_dfs: dict, cutoff_date, season_start_window: int = 5,
) -> dict:
    """WP011: build one ModelData per league, independently.

    `engineered_dfs`: {league_name: df}, each df already run through the
    same feature-engineering pipeline a single-league df goes through
    (add_rounds_to_data / add_match_ids / add_home_away_goals_xg) —
    unchanged, called once per league exactly as it already is for EPL
    alone. Nothing about that pipeline or about prepare_model_data itself
    needed to change for multi-league support: each league's df is prepared
    completely independently, with its own local team-index space
    (team_mapping) and its own time axis starting at 0 — teams in different
    leagues never share a match, and there's no shared "global round
    calendar" for prepare_model_data to get confused about, because each
    call only ever sees one league's own df.

    `cutoff_date`: shared real-world "as of" date across every league (see
    max_round_for_cutoff_date for why a date, not a round number, is the
    leakage-safe way to align competitions with different season lengths).

    A league with ZERO matches on or before `cutoff_date` — the real case
    this guards against: a non-eval league deliberately fetched with fewer
    seasons than the eval league (WP011's "trim non-EPL history" compute
    lever), so an early eval-league window's cutoff date can fall before
    that league's trimmed data even starts — is silently OMITTED from the
    result rather than passed into prepare_model_data, which would crash
    (`max_round=0` makes its `df[df["round"] <= max_round]` filter select
    nothing, and `nan`-length axes downstream) rather than degrade
    gracefully. This is the league-level analogue of a single league's own
    `active_mask`: a team/league that hasn't started yet contributes
    nothing to that window, it doesn't error the whole run. Found and fixed
    via a real end-to-end test (see WP011's README), not written
    defensively up front — worth being upfront that this needs checking
    for any newly-added league/trim combination, not just trusted blind.

    Returns {league_name: ModelData} — MAY have fewer keys than
    `engineered_dfs` for early cutoff dates (see above); always has at
    least the eval league, since callers are expected to trim only the
    NON-eval leagues, never the one actually being evaluated.
    """
    result = {}
    for name, df in engineered_dfs.items():
        max_round = max_round_for_cutoff_date(df, cutoff_date)
        if max_round == 0:
            print(f"  prepare_multileague_data: skipping {name!r} for this window — "
                  f"no matches on/before {pd.Timestamp(cutoff_date).date()} (league's fetched history starts later)")
            continue
        result[name] = prepare_model_data(
            df, max_round=max_round, season_start_window=season_start_window,
        )
    return result


def prepare_model_data(
    df: pd.DataFrame, max_round, season_start_window: int = 5,
    lineup_dev_table: pd.DataFrame = None,
    continuity_table: pd.DataFrame = None,
) -> ModelData:
    """`lineup_dev_table` (WP008, optional): the output of
    football_model.features.lineup_features.build_lineup_deviation_table —
    columns `team` (full name, matching `team_long`/`opp_team_long` below,
    NOT the short codes in `team`/`opp_team`), `date`, `lineup_dev`. Joined
    onto each observation by (team, match date); a match with no matching
    row (lineup data doesn't cover that far back, or wasn't fetched)
    defaults to lineup_dev=0.0 — "no known deviation from normal" is exactly
    the safe default when there's no information either way, not a missing
    value that needs to propagate or crash downstream. Leave as None (the
    default) to skip lineup features entirely, same as every call site
    before WP008.
    """
    # Sort & index teams
    df = df.sort_values("datetime").reset_index(drop=True)
    teams = pd.unique(df[['team', 'opp_team']].values.ravel())
    team_idx_map = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    df["team_id"] = df["team"].map(team_idx_map)
    df["opp_id"] = df["opp_team"].map(team_idx_map)
    df["t"] = df["cum_round"].astype(int)
    df["match_id"] = df["match_id"].astype(int)

    # Only train data - exclude max_round to prevent leakage
    df_train = df[df["round"] <= max_round].copy()
    n_time = df_train["t"].max() + 1
    n_matches = df_train["match_id"].nunique()

    # Each match appears twice in df_train — once from each team's
    # perspective (is_home==1 and is_home==0). Team-level rolling stats
    # below (xG history, active-team mask) need BOTH perspectives to see a
    # team's full home+away record. The final per-observation arrays built
    # further down, however, must use only ONE row per match: goals_home/
    # goals_away (and theta_home/theta_away in model.py) already encode the
    # full match from a single home-perspective row, so keeping both rows
    # would feed every match's outcome into the Poisson likelihood twice —
    # once correctly (with the home team's home_adv bonus) and once under a
    # mismatched mean function that omits it. See notes in model.py.

    # Calculate team-level xG features using ONLY historical data (no data leakage)
    # For each match at time t, use team's average xG from matches BEFORE time t
    team_xg_at_t = np.zeros((n_time, n_teams))  # (time, team) array
    team_xga_at_t = np.zeros((n_time, n_teams))  # xG allowed

    # Initialize with league average for time 0
    league_avg_xg = 1.5  # Reasonable default
    team_xg_at_t[0, :] = league_avg_xg
    team_xga_at_t[0, :] = league_avg_xg

    # Build rolling averages (using expanding window) from both perspectives,
    # so each team's home and away matches both count toward its history.
    for t in range(1, n_time):
        # For each team, calculate average xG from all matches up to (but not including) time t
        historical_data = df_train[df_train["t"] < t]

        for team_id in range(n_teams):
            # Team's attacking xG (when they are the team)
            team_matches = historical_data[historical_data["team_id"] == team_id]
            if len(team_matches) > 0:
                team_xg_at_t[t, team_id] = team_matches["xG"].mean()
            else:
                team_xg_at_t[t, team_id] = league_avg_xg

            # Team's defensive xG allowed (when they are opponent)
            opp_matches = historical_data[historical_data["opp_id"] == team_id]
            if len(opp_matches) > 0:
                team_xga_at_t[t, team_id] = opp_matches["xG"].mean()
            else:
                team_xga_at_t[t, team_id] = league_avg_xg

    # Active-team mask & season-start window.
    # A team is "active" at time t if it played any match in the season that
    # round t belongs to — this stops relegated teams' latent attack/defence
    # from drifting on pure random-walk noise while they're out of the
    # league, and stops them from diluting the per-round centering mean.
    # `season_start_mask` flags each season's first `season_start_window`
    # rounds, where we allow extra innovation variance to reflect summer
    # squad turnover (transfers affect every team, not just promoted ones).
    # Computed from both perspectives so a team only having played away
    # fixtures so far this season still counts as active.
    active_mask = np.zeros((n_time, n_teams), dtype="float32")
    season_start_mask = np.zeros(n_time, dtype="float32")

    season_teams = df_train.groupby("season").apply(
        lambda g: set(g["team_id"]).union(set(g["opp_id"])), include_groups=False
    )
    season_of_t = df_train.groupby("t")["season"].first()
    season_min_t = df_train.groupby("season")["t"].min()

    for t in range(n_time):
        if t in season_of_t.index:
            season = season_of_t.loc[t]
            teams_this_season = list(season_teams.loc[season])
            active_mask[t, teams_this_season] = 1.0
            season_round = t - season_min_t.loc[season] + 1
            if season_round <= season_start_window:
                season_start_mask[t] = 1.0
        else:
            # No observed matches at this t (e.g. the unused t=0 slot before
            # the first round) — treat as fully active, no inflation, so it
            # behaves like a no-op in downstream masking.
            active_mask[t, :] = 1.0

    # --- Build the actual per-observation arrays: one row per match ---
    # Keep only the home-perspective row of each match. team_id is always
    # the home team and opp_id always the away team here, so this also
    # simplifies the xG lookup below (no more is_home branching needed).
    df_obs = df_train[df_train["is_home"] == 1].copy()

    xG_home_baseline = np.zeros(len(df_obs))
    xG_away_baseline = np.zeros(len(df_obs))

    obs_t = df_obs["t"].to_numpy()
    obs_home_team = df_obs["team_id"].to_numpy()
    obs_away_team = df_obs["opp_id"].to_numpy()

    for idx in range(len(df_obs)):
        t = obs_t[idx]
        home_team = obs_home_team[idx]
        away_team = obs_away_team[idx]

        # Home team xG = their attacking xG adjusted by opponent's defensive xG
        xG_home_baseline[idx] = (team_xg_at_t[t, home_team] + team_xga_at_t[t, away_team]) / 2
        xG_away_baseline[idx] = (team_xg_at_t[t, away_team] + team_xga_at_t[t, home_team]) / 2

    if lineup_dev_table is not None and len(lineup_dev_table) > 0:
        ldt_dates = pd.to_datetime(lineup_dev_table["date"]).dt.normalize()
        lineup_dev_map = dict(zip(zip(lineup_dev_table["team"], ldt_dates), lineup_dev_table["lineup_dev"]))
        obs_dates = df_obs["datetime"].dt.normalize()
        lineup_dev_home = np.array(
            [lineup_dev_map.get((t, d), 0.0) for t, d in zip(df_obs["team_long"], obs_dates)]
        )
        lineup_dev_away = np.array(
            [lineup_dev_map.get((t, d), 0.0) for t, d in zip(df_obs["opp_team_long"], obs_dates)]
        )
    else:
        lineup_dev_home = np.zeros(len(df_obs))
        lineup_dev_away = np.zeros(len(df_obs))

    # WP013: same join, same neutral default. `continuity_table` has columns
    # `team` (full name), `date`, `continuity_z` (standardised defence
    # continuity, NaN already mapped to 0 — see
    # football_model.features.continuity_features.continuity_feature_table).
    if continuity_table is not None and len(continuity_table) > 0:
        ct_dates = pd.to_datetime(continuity_table["date"]).dt.normalize()
        cont_map = dict(zip(zip(continuity_table["team"], ct_dates), continuity_table["continuity_z"]))
        obs_dates_c = df_obs["datetime"].dt.normalize()
        defence_cont_home = np.array(
            [cont_map.get((t, d), 0.0) for t, d in zip(df_obs["team_long"], obs_dates_c)]
        )
        defence_cont_away = np.array(
            [cont_map.get((t, d), 0.0) for t, d in zip(df_obs["opp_team_long"], obs_dates_c)]
        )
    else:
        defence_cont_home = np.zeros(len(df_obs))
        defence_cont_away = np.zeros(len(df_obs))

    return ModelData(
        n_teams=n_teams,
        n_matches=n_matches,
        n_time=n_time,
        t_idx=df_obs["t"].values.astype("int32"),
        team_idx=df_obs["team_id"].values.astype("int32"),
        opp_idx=df_obs["opp_id"].values.astype("int32"),
        match_idx=df_obs["match_id"].values.astype("int32"),
        home=df_obs["is_home"].values.astype("float32"),
        goals_home=df_obs["goals_home"].values.astype("int32"),
        goals_away=df_obs["goals_away"].values.astype("int32"),
        xG_home=xG_home_baseline.astype("float32"),
        xG_away=xG_away_baseline.astype("float32"),
        team_mapping=team_idx_map,
        active_mask=active_mask,
        season_start_mask=season_start_mask,
        lineup_dev_home=lineup_dev_home.astype("float32"),
        lineup_dev_away=lineup_dev_away.astype("float32"),
        defence_cont_home=defence_cont_home.astype("float32"),
        defence_cont_away=defence_cont_away.astype("float32"),
    )
