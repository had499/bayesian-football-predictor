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
            league_player_data = understat.league(league=league)

            df = pd.DataFrame(league_player_data.get_match_data(year))
          
            transformed_data = transformer.fit_transform(df)
            
            transformed_data['season'] = year
            transformed_data["gd"] = transformed_data["goals"] - transformed_data["goals_against"]

            
            final_df_list.append(transformed_data)
            
            
    final_df = pd.concat(final_df_list, ignore_index=True)

    return final_df
