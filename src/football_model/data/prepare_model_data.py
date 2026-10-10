from football_model.types.model_data import *
import numpy as np
import pandas as pd

def _elo_by_walk_step(grid, df_train, n_time, n_teams, team_idx_map):
    """(cum_round x team) Elo -> (n_time x n_teams) on the walk's index.

    Collapses to the walk grid when time_grid_step > 1 by taking the FIRST round
    mapped to each node, which is the one a team would actually enter on.
    """
    out = np.zeros((int(n_time), int(n_teams)), dtype="float32")
    rounds = df_train[["t", "t_true"]].drop_duplicates().sort_values("t_true")
    for t_walk, t_true in zip(rounds.t.to_numpy(), rounds.t_true.to_numpy()):
        if t_true not in grid.index:
            continue
        row = grid.loc[t_true]
        for team, i in team_idx_map.items():
            if team in row.index and out[t_walk, i] == 0.0:
                out[t_walk, i] = float(row[team])
    return out


def prepare_model_data(df: pd.DataFrame, max_round, season_start_window: int = 5,
                       time_grid_step: int = 1, elo_z_grid=None) -> ModelData:
    """Build ModelData for a training window.

    time_grid_step (WP023, performance): put the latent walk on a COARSER grid by
    collapsing every `k` rounds into one node. Sampling cost is quadratic in the
    number of rounds -- `seconds ~ 0.0024 * rounds^2`, because the latent array is
    (n_time x n_teams x 2), about 29,000 parameters at the largest window -- so
    k=2 roughly halves it and k=3 cuts it to a third.

    The justification is that the fitted walk is extremely smooth: rho ~ 0.993 and
    sigma ~ 0.016 per round, so a team's strength moves ~0.023 over two rounds,
    far below the spread the latents span. Whether that costs accuracy is an
    empirical question, not an assumption -- compare RPS before using it.

    k=1 (the default) leaves every existing result bit-identical.
    """
    # Sort & index teams
    df = df.sort_values("datetime").reset_index(drop=True)
    teams = pd.unique(df[['team', 'opp_team']].values.ravel())
    team_idx_map = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    df["team_id"] = df["team"].map(team_idx_map)
    df["opp_id"] = df["opp_team"].map(team_idx_map)
    # `t_true` is the real round, and is what ModelData.t_idx reports. It is used
    # to SELECT test rows and to LABEL matches for the odds join, so it must never
    # be coarsened. `t` is the LATENT WALK's index and is the thing k collapses.
    # Conflating the two is a live bug: coarsening t_idx silently mislabels rounds.
    df["t_true"] = df["cum_round"].astype(int)
    df["t"] = df["t_true"] // int(time_grid_step) if time_grid_step > 1 else df["t_true"]
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
    # (strictly t' < t), from both perspectives so a team's home and away
    # matches both count. Vectorised with cumulative sums — this used to be a
    # per-(t, team) pandas filter (~n_time*n_teams full-frame scans per call),
    # which dominated runtime once training spans 10+ seasons. Semantics are
    # unchanged (see tests/test_prepare_model_data.py's slow reference).
    league_avg_xg = 1.5  # Reasonable default where a team has no history yet

    tt = df_train["t"].to_numpy(int)
    tid = df_train["team_id"].to_numpy(int)
    oid = df_train["opp_id"].to_numpy(int)
    xg_row = df_train["xG"].to_numpy(float)

    sum_f = np.zeros((n_time, n_teams)); cnt_f = np.zeros((n_time, n_teams))
    sum_a = np.zeros((n_time, n_teams)); cnt_a = np.zeros((n_time, n_teams))
    np.add.at(sum_f, (tt, tid), xg_row); np.add.at(cnt_f, (tt, tid), 1.0)  # team's own attacking xG
    np.add.at(sum_a, (tt, oid), xg_row); np.add.at(cnt_a, (tt, oid), 1.0)  # xG allowed (team as opponent)

    def _expanding_mean_before(sums, cnts):
        cs, cc = np.cumsum(sums, axis=0), np.cumsum(cnts, axis=0)
        out = np.full((n_time, n_teams), league_avg_xg)
        # row t uses everything up to and including t-1
        np.divide(cs[:-1], cc[:-1], out=out[1:], where=cc[:-1] > 0)
        out[1:][cc[:-1] == 0] = league_avg_xg
        return out

    team_xg_at_t = _expanding_mean_before(sum_f, cnt_f)    # (time, team)
    team_xga_at_t = _expanding_mean_before(sum_a, cnt_a)   # xG allowed

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
    # 0-based season index, ordered chronologically (for season-varying xG calibration)
    season_codes = pd.Categorical(df_obs["season"],
                                  categories=sorted(df_train["season"].unique()),
                                  ordered=True).codes

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

    return ModelData(
        n_teams=n_teams,
        n_matches=n_matches,
        n_time=n_time,
        t_idx=df_obs["t_true"].values.astype("int32"),
        t_walk=df_obs["t"].values.astype("int32"),
        team_idx=df_obs["team_id"].values.astype("int32"),
        opp_idx=df_obs["opp_id"].values.astype("int32"),
        match_idx=df_obs["match_id"].values.astype("int32"),
        home=df_obs["is_home"].values.astype("float32"),
        goals_home=df_obs["goals_home"].values.astype("int32"),
        goals_away=df_obs["goals_away"].values.astype("int32"),
        xG_home=xG_home_baseline.astype("float32"),
        xG_away=xG_away_baseline.astype("float32"),
        # Realised in-match xG (an observation, for the WP007 measurement
        # channel) — floored at 0.02 so a Gamma log-density stays finite for
        # the odd xG==0.00 row. Not a predictor for the match itself.
        # WP014: if the caller's df carries non-penalty xG (npxg_home/away), the
        # measurement channel observes THAT instead of raw xG. Penalties are ~7% of
        # total xG but far less repeatable than open-play xG, so they inject noise
        # into the channel roughly in proportion to their share while contributing
        # little signal about team strength.
        xG_obs_home=np.clip(
            (df_obs["npxg_home"] if "npxg_home" in df_obs else df_obs["xG_home"]).to_numpy(float),
            0.02, None).astype("float32"),
        xG_obs_away=np.clip(
            (df_obs["npxg_away"] if "npxg_away" in df_obs else df_obs["xG_away"]).to_numpy(float),
            0.02, None).astype("float32"),
        # WP019: realised shot counts, if the caller's df carries them.
        shots_home=(df_obs["shots_home"].to_numpy(float).astype("float32")
                    if "shots_home" in df_obs else None),
        shots_away=(df_obs["shots_away"].to_numpy(float).astype("float32")
                    if "shots_away" in df_obs else None),
        # WP054: GBM residual-correction covariate, if the caller's df carries it.
        # WP061: availability offset inputs, if the caller's df carries them.
        avail_out_h=(df_obs["avail_out_h"].to_numpy(float).astype("float32")
                     if "avail_out_h" in df_obs else None),
        avail_out_a=(df_obs["avail_out_a"].to_numpy(float).astype("float32")
                     if "avail_out_a" in df_obs else None),
        gbm_z_home=(df_obs["gbm_z_home"].to_numpy(float).astype("float32")
                    if "gbm_z_home" in df_obs else None),
        gbm_z_away=(df_obs["gbm_z_away"].to_numpy(float).astype("float32")
                    if "gbm_z_away" in df_obs else None),
        # WP039: market-implied rates, if the caller's df carries them.
        market_lh=(df_obs["market_lh"].to_numpy(float).astype("float32")
                   if "market_lh" in df_obs else None),
        market_la=(df_obs["market_la"].to_numpy(float).astype("float32")
                   if "market_la" in df_obs else None),
        market_obs_w=(df_obs["market_obs_w"].to_numpy(float).astype("float32")
                      if "market_obs_w" in df_obs else None),
        # WP031: crowdless indicator, if the caller's df carries it.
        crowdless=(df_obs["crowdless"].to_numpy(float).astype("float32")
                   if "crowdless" in df_obs else None),
        # WP030: squad intactness, if the caller's df carries it.
        intact_home=(df_obs["intact_home"].to_numpy(float).astype("float32")
                     if "intact_home" in df_obs else None),
        intact_away=(df_obs["intact_away"].to_numpy(float).astype("float32")
                     if "intact_away" in df_obs else None),
        # WP028: end-of-season stakes, if the caller's df carries them.
        stakes_home=(df_obs["stakes_home"].to_numpy(float).astype("float32")
                     if "stakes_home" in df_obs else None),
        stakes_away=(df_obs["stakes_away"].to_numpy(float).astype("float32")
                     if "stakes_away" in df_obs else None),
        stakes_match=(df_obs["stakes_match"].to_numpy(float).astype("float32")
                      if "stakes_match" in df_obs else None),
        # WP027: congestion covariates, if the caller's df carries them.
        rest_home=(df_obs["rest_home"].to_numpy(float).astype("float32")
                   if "rest_home" in df_obs else None),
        rest_away=(df_obs["rest_away"].to_numpy(float).astype("float32")
                   if "rest_away" in df_obs else None),
        load_home=(df_obs["load_home"].to_numpy(float).astype("float32")
                   if "load_home" in df_obs else None),
        load_away=(df_obs["load_away"].to_numpy(float).astype("float32")
                   if "load_away" in df_obs else None),
        # WP025: standardised Elo per (walk step, team), read only at entry rounds.
        entry_elo_z=_elo_by_walk_step(elo_z_grid, df_train, n_time, n_teams,
                                      team_idx_map) if elo_z_grid is not None else None,
        # WP022: match age in days, for time-decay weighting.
        match_day=((pd.to_datetime(df_obs["datetime"]) -
                    pd.to_datetime(df_obs["datetime"]).min()).dt.total_seconds()
                   / 86400.0).to_numpy().astype("float64"),
        # WP020: deep completions, if the caller's df carries them.
        deep_home=(df_obs["deep_home"].to_numpy(float).astype("float32")
                   if "deep_home" in df_obs else None),
        deep_away=(df_obs["deep_away"].to_numpy(float).astype("float32")
                   if "deep_away" in df_obs else None),
        season_idx=season_codes.astype("int32"),
        n_seasons=int(season_codes.max()) + 1 if len(season_codes) else 0,
        # WP012 squad-absence covariate; absent if the caller's df doesn't carry it.
        # df_obs is home-perspective only, so "self" is the home side.
        absent_home=(df_obs["absent_self"].to_numpy(float).astype("float32")
                     if "absent_self" in df_obs else np.zeros(len(df_obs), "float32")),
        absent_away=(df_obs["absent_opp"].to_numpy(float).astype("float32")
                     if "absent_opp" in df_obs else np.zeros(len(df_obs), "float32")),
        team_mapping=team_idx_map,
        active_mask=active_mask,
        season_start_mask=season_start_mask,
    )
