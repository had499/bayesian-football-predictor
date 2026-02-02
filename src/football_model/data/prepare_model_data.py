from football_model.types.model_data import *
import numpy as np

def prepare_model_data(df: pd.DataFrame, max_round) -> ModelData:
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
    
    # Calculate team-level xG features using ONLY historical data (no data leakage)
    # For each match at time t, use team's average xG from matches BEFORE time t
    team_xg_at_t = np.zeros((n_time, n_teams))  # (time, team) array
    team_xga_at_t = np.zeros((n_time, n_teams))  # xG allowed
    
    # Initialize with league average for time 0
    league_avg_xg = 1.5  # Reasonable default
    team_xg_at_t[0, :] = league_avg_xg
    team_xga_at_t[0, :] = league_avg_xg
    
    # Build rolling averages (using expanding window)
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
    
    # Now for each match, use the pre-match team xG averages
    xG_home_baseline = np.zeros(len(df_train))
    xG_away_baseline = np.zeros(len(df_train))
    
    for idx in range(len(df_train)):
        t = df_train.iloc[idx]["t"]
        home_team = df_train.iloc[idx]["team_id"] if df_train.iloc[idx]["is_home"] == 1 else df_train.iloc[idx]["opp_id"]
        away_team = df_train.iloc[idx]["opp_id"] if df_train.iloc[idx]["is_home"] == 1 else df_train.iloc[idx]["team_id"]
        
        # Home team xG = their attacking xG adjusted by opponent's defensive xG
        xG_home_baseline[idx] = (team_xg_at_t[t, int(home_team)] + team_xga_at_t[t, int(away_team)]) / 2
        xG_away_baseline[idx] = (team_xg_at_t[t, int(away_team)] + team_xga_at_t[t, int(home_team)]) / 2

    return ModelData(
        n_teams=n_teams,
        n_matches=n_matches,
        n_time=n_time,
        t_idx=df_train["t"].values.astype("int32"),
        team_idx=df_train["team_id"].values.astype("int32"),
        opp_idx=df_train["opp_id"].values.astype("int32"),
        match_idx=df_train["match_id"].values.astype("int32"),
        home=df_train["is_home"].values.astype("float32"),
        goals_home=df_train["goals_home"].values.astype("int32"),
        goals_away=df_train["goals_away"].values.astype("int32"),
        xG_home=xG_home_baseline.astype("float32"),
        xG_away=xG_away_baseline.astype("float32"),
        team_mapping=team_idx_map,
    )
