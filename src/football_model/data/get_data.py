from understatapi import UnderstatClient
from football_model.features.match_transformer import MatchDataTransformer
import pandas as pd

def get_understat_data(years=['2024'],
                                  leagues = ['EPL', 'RFPL','Bundesliga', 'La_Liga', 'Serie_A', 'Ligue_1']):
    """Process data using the sklearn transformer approach."""
    
    understat = UnderstatClient()
    final_df_list = []

    
    # Initialize the transformer
    transformer = MatchDataTransformer()
    
    for year in years:
        for league in leagues:
            try:
                league_player_data = understat.league(league=league)
                match_data = league_player_data.get_match_data(year)
                
                if not match_data:
                    print(f"Warning: No data returned for {league} {year}")
                    continue
                
                df = pd.DataFrame(match_data)
                
                if df.empty:
                    print(f"Warning: Empty dataframe for {league} {year}")
                    continue
              
                transformed_data = transformer.fit_transform(df)
                
                transformed_data['season'] = str(year)
                transformed_data["gd"] = transformed_data["goals"] - transformed_data["goals_against"]
                
                final_df_list.append(transformed_data)
                
            except Exception as e:
                print(f"Error fetching {league} {year}: {str(e)}")
                continue
            
    final_df = pd.concat(final_df_list, ignore_index=True)
    
    if final_df.empty:
        raise ValueError(f"No data could be fetched for any league/year combination. Tried: leagues={leagues}, years={years}")

    return final_df
