import pandas as pd
import requests
import numpy as np

mappings = {
    'Manchester City': 'ManCity',
    'Manchester United': 'ManUnited', 
    'Newcastle United': 'Newcastle',
    'Nottingham Forest': 'Forest',
    'Queens Park Rangers': 'QPR',
    'West Bromwich Albion': 'WestBrom',
    'Wolverhampton Wanderers': 'Wolves',
}

def merge_elo(df_main,df_elo,team_col,mappings=mappings,elo_prefix=None):
    df1 = df_main.copy()
    df2 = df_elo.copy()
    df2 = df2.rename(columns={'Club':team_col})
    
    df1[team_col] = df1[team_col].apply(lambda x: mappings[x] 
                                                 if x in mappings.keys() 
                                                 else x.replace(' ',''))
    df2[team_col] = df2[team_col].apply(lambda x: str(x).replace(' ','') )

    df2 = df2[~df2.Elo.isna()].reset_index(drop=True)
    
    df1.datetime = df1.datetime.dt.normalize()
    df2.From = pd.to_datetime(df2.From)
    df2.To = pd.to_datetime(df2.To)


    results = []

    print(f"Starting with {len(df1)} rows in df1")

    for team in df1[team_col].unique():
        if team not in df2[team_col].values:
            print(f"⚠️ Skipping {team} - not found in Elo data")
            # Still add the original data without Elo ratings
            team_data = df1[df1[team_col] == team].copy()
            team_data['Elo'] = None
            team_data['From'] = None
            team_data['To'] = None
            results.append(team_data)
            continue

        # Get data for this team
        team_df1 = df1[df1[team_col] == team].sort_values('datetime').copy()
        team_df2 = df2[df2[team_col] == team].sort_values('From').copy()


        # Perform merge_asof for this team only
        try:
            team_merged = pd.merge_asof(
                team_df1,
                team_df2,
                left_on='datetime',
                right_on='From',
                direction='backward',
            )

            results.append(team_merged)


        except Exception as e:
            print(f"Failed to merge {team}: {e}")
            # Add original data without Elo if merge fails
            team_data = team_df1.copy()
            team_data['Elo'] = None
            team_data['From'] = None  
            team_data['To'] = None
            results.append(team_data)
            continue
    result= pd.concat(results)

    if elo_prefix:
        result = result.rename(columns={
            'Elo': f'{elo_prefix}_Elo',
            'From': f'{elo_prefix}_From',
            'To': f'{elo_prefix}_To'
        })
    return result



def get_elo(club_name, mappings=mappings):
    
    if club_name in mappings.keys():
        club_name = mappings[club_name]
    
    res = requests.get(f'http://api.clubelo.com/{club_name.replace(' ','')}')
    club_elo_df = pd.DataFrame([x.split(',') for x in res.text.split('\n')[1:]], 
                 columns=[x for x in res.text.split('\n')[0].split(',')])
    club_elo_df[~club_elo_df.fillna(np.nan).Elo.isna()]
    return club_elo_df