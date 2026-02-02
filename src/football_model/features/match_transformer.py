import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin


class MatchDataTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for converting wide-format match data to long format and applying transformations.
    
    This transformer handles the conversion from match-level data (home vs away) to team-level data
    and applies the FootballDataTransformer for feature engineering.
    
    Parameters
    ----------
    football_transformer : FootballDataTransformer, optional
        Pre-configured FootballDataTransformer instance. If None, uses default settings.
    """
    
    def __init__(self):
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer."""
        # self._validate_match_input(X)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform match data to long format and apply feature engineering."""
        if not self.is_fitted_:
            raise ValueError("This MatchDataTransformer instance is not fitted yet.")
        
        # self._validate_match_input(X)

        # Clean the data
        X_clean = self._clean_data(X)
        
        # Convert to long format
        long_data = self._convert_to_long_format(X_clean)




        return long_data
    
    # def _validate_match_input(self, X: pd.DataFrame):
    #     """Validate match-level input data."""
    #     required_columns = ['datetime', 'h_short_title', 'a_short_title']
    #     missing_columns = [col for col in required_columns if col not in X.columns]
        
    #     if missing_columns:
    #         raise ValueError(f"Missing required columns: {missing_columns}")
        
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = pd.concat(
            [
                df[['id', 'datetime']],
                pd.json_normalize(df['h']).add_prefix('h_'),
                pd.json_normalize(df['a']).add_prefix('a_'),
                pd.json_normalize(df['goals']).add_suffix('_g'),
                pd.json_normalize(df['xG']).add_suffix('_xG'),
                # pd.json_normalize(df['forecast']).add_prefix('forecast_')
            ],
            axis=1
        )


        df_clean = df_clean[['datetime', 'h_short_title', 'h_title', 'a_short_title', 'a_title',
                              'h_g', 'a_g','h_xG', 'a_xG', ]]

        df_clean.h_g = df_clean.h_g.fillna(0).astype(int)
        df_clean.a_g = df_clean.a_g.fillna(0).astype(int)
        df_clean.h_xG = df_clean.h_xG.fillna(0).astype(float)
        df_clean.a_xG = df_clean.a_xG.fillna(0).astype(float)

        df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])
        df_clean = df_clean.sort_values(by='datetime').reset_index(drop=True)

        return df_clean
    
    def _convert_to_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert wide match format to long team format."""
        # Create long format data
        home_data = df[['datetime', 'h_short_title','a_short_title', 'h_title','a_title', 
                        'h_g', 'h_xG', 'a_g', 'a_xG']].rename(
            columns={'h_short_title': 'team', 'a_short_title':'opp_team', 'h_title':'team_long', 'a_title':'opp_team_long',
                      'h_g': 'goals', 'h_xG': 'xG', 'a_xG': 'xGA', 'a_g': 'goals_against',}
        )
        home_data['is_home']=1
        away_data = df[['datetime', 'a_short_title', 'h_short_title', 'a_title', 'h_title',
                         'a_g', 'a_xG', 'h_xG', 'h_g']].rename(
            columns={'a_short_title': 'team', 'h_short_title':'opp_team', 'a_title':'team_long', 'h_title':'opp_team_long',
                     'a_g': 'goals', 'a_xG': 'xG', 'h_xG': 'xGA', 'h_g': 'goals_against'}
        )
        away_data['is_home']=0
        long_data = pd.concat([home_data, away_data], axis=0).sort_values(by='datetime').reset_index(drop=True)

        long_data.loc[long_data.goals > long_data.goals_against, 'win'] = 1
        long_data.loc[long_data.goals < long_data.goals_against, 'loss'] = 1
        long_data.loc[long_data.goals == long_data.goals_against, 'draw'] = 1
        long_data[['win', 'loss', 'draw']] = long_data[['win', 'loss', 'draw']].fillna(0)
            
        return long_data