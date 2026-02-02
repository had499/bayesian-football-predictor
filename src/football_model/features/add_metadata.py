import pandas as pd 
from datetime import timedelta

def add_rounds_to_data(df):
    """Add round numbers to understat data.
    
    Original function that expects datetime column for proper round calculation.
    This is the sophisticated logic for handling seasons and cumulative rounds.
    """
    def adjust_round(season_df):
        """Adjust round numbers to be consecutive within a season."""
        first_round = season_df.iloc[0]['round']
        season_df['round'] = season_df['round'].apply(lambda x: x+52 if x < first_round else x)
        season_df['round'] = season_df['round'].apply(lambda x: x - first_round + 1)
        return season_df

    def make_rounds_consecutive(season_df):
        """Reassign round numbers to be consecutive starting from 1 inter-season."""
        season_df = season_df.sort_values('datetime').copy()
        unique_rounds = sorted(season_df['round'].unique())
        round_map = {r: i+1 for i, r in enumerate(unique_rounds)}
        season_df['round'] = season_df['round'].map(round_map)
        return season_df
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['dt_round'] = df['datetime']

    # Convert Mondays to previous Sunday
    df.loc[df['dt_round'].dt.isocalendar().day == 1, 'dt_round'] -= timedelta(days=1)

    # Compute ISO week
    df['round'] = df['dt_round'].dt.isocalendar().week

    df = df.groupby('season', group_keys=False).apply(adjust_round)
    df = df.drop(columns='dt_round')
    df = df.groupby('season', group_keys=False).apply(make_rounds_consecutive)

    # Now compute cumulative rounds
    season_max_rounds = df.groupby('season')['round'].max().sort_index()
    season_offset = season_max_rounds.cumsum().shift(fill_value=0)
    df['cum_round'] = df['season'].map(season_offset) + df['round']
    df['round'] = df['cum_round']

    return df



def add_match_ids(df):
    df["match_key"] = (
        df["datetime"].astype(str) + "_" +
        df[["team", "opp_team"]].apply(
            lambda x: "_".join(sorted(x)), axis=1
        )
    )

    df["match_id"] = df["match_key"].astype("category").cat.codes
    
    return df



def add_home_away_goals_xg(df):
    df.loc[df.is_home==1,'goals_home']=df[df.is_home==1].goals
    df.loc[df.is_home==0,'goals_home']=df[df.is_home==0].goals
    df.loc[df.is_home==1,'xG_home']=df[df.is_home==1].xG
    df.loc[df.is_home==0,'xG_home']=df[df.is_home==0].xG


    df.loc[df.is_home==1,'goals_away']=df[df.is_home==1].goals_against
    df.loc[df.is_home==0,'goals_away']=df[df.is_home==0].goals_against
    df.loc[df.is_home==1,'xG_away']=df[df.is_home==1].xGA
    df.loc[df.is_home==0,'xG_away']=df[df.is_home==0].xGA

    return df