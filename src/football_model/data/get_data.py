from understatapi import UnderstatClient
from football_model.features.match_transformer import MatchDataTransformer
import pandas as pd
import unicodedata

_TEAM_NAME_COLS = ["team", "opp_team", "team_long", "opp_team_long"]


def _normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """WP011: defence against Unicode normalization mismatches (e.g. 'e' +
    combining acute accent vs. the single precomposed 'e-acute' codepoint —
    same displayed character, different bytes, silent join/groupby failures
    downstream). Checked empirically against real fetched data across all 5
    leagues (Bundesliga/La_Liga/Ligue_1/Serie_A all come back fully ASCII,
    already-anglicized team names, e.g. "Alaves" not "Alavés" — so this is
    currently a no-op in practice), but applied unconditionally rather than
    assumed, since Understat's own naming could change and this is nearly
    free."""
    for col in _TEAM_NAME_COLS:
        if col in df.columns:
            df[col] = df[col].map(lambda s: unicodedata.normalize("NFC", s) if isinstance(s, str) else s)
    return df


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

    final_df = _normalize_team_names(final_df)
    return final_df
