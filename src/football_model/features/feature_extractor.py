import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Union


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer for football match data preprocessing.
    
    This transformer applies various feature engineering techniques including:
    - Lagged features for goals, xG, goals_against, xGA
    - Rolling averages and exponential weighted means
    - Win rates (home/away)
    - Form and streak calculations
    - Rest days between matches
    
    Parameters
    ----------
    lags : list, default=[1, 3, 5]
        List of lag periods to create lagged features
    rolling_window : int, default=5
        Window size for rolling averages and form calculations
    add_win_rates : bool, default=True
        Whether to compute home and away win rates
    add_form_streaks : bool, default=True
        Whether to compute form and streak features
    add_rest_days : bool, default=True
        Whether to compute rest days between matches
    fill_method : str, default='ffill'
        Method to fill NaN values ('ffill', 'bfill', 'median', 'mean')
    """
    
    def __init__(
        self,
        lags: List[int] = [1, 3, 5],
        rolling_window: int = 5,
        add_win_rates: bool = True,
        add_form_streaks: bool = True,
        add_rest_days: bool = True,
        fill_method: str = 'ffill'
    ):
        self.lags = lags
        self.rolling_window = rolling_window
        self.add_win_rates = add_win_rates
        self.add_form_streaks = add_form_streaks
        self.add_rest_days = add_rest_days
        self.fill_method = fill_method
        
        # These will be set during fit
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer (learn feature names and validate input).
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data in long format with columns: team, datetime, goals, xG, 
            goals_against, xGA, win, loss, draw, is_home
        y : ignored
            Not used, present for API consistency
            
        Returns
        -------
        self : object
            Returns the instance itself
        """
        # Validate input
        self._validate_input(X)
        
        # Store input feature names
        self.feature_names_in_ = list(X.columns)
        
        # Mark as fitted
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by applying feature engineering.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data in long format
            
        Returns
        -------
        X_transformed : pd.DataFrame
            Transformed data with engineered features
        """
        # Check if fitted
        if not self.is_fitted_:
            raise ValueError("This FootballDataTransformer instance is not fitted yet.")
        
        # Validate input
        self._validate_input(X)
        
        # Make a copy to avoid modifying original data
        X_transformed = X.copy()
        
        # Ensure datetime is properly formatted
        if 'datetime' in X_transformed.columns:
            X_transformed['datetime'] = pd.to_datetime(X_transformed['datetime'])
        
        # Sort by team and datetime
        X_transformed = X_transformed.sort_values(['team', 'datetime']).reset_index(drop=True)
        
        # Apply transformations
        X_transformed = self._add_lagged_features(X_transformed)
        X_transformed = self._add_aggregated_features(X_transformed)
        
        if self.add_win_rates:
            X_transformed = self._compute_win_rates(X_transformed)
        
        if self.add_form_streaks:
            X_transformed = self._compute_form_and_streaks(X_transformed)
        
        if self.add_rest_days:
            X_transformed = self._compute_rest_days(X_transformed)
        
        # Handle missing values
        X_transformed = self._handle_missing_values(X_transformed)
        
        # Store output feature names
        self.feature_names_out_ = list(X_transformed.columns)
        
        return X_transformed
    
    def _validate_input(self, X: pd.DataFrame):
        """Validate that input DataFrame has required columns."""
        required_columns = ['team', 'datetime']
        missing_columns = [col for col in required_columns if col not in X.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for numeric columns that should exist for meaningful transformations
        numeric_columns = ['goals', 'xG', 'goals_against', 'xGA']
        available_numeric = [col for col in numeric_columns if col in X.columns]
        
        if len(available_numeric) == 0:
            raise ValueError("No numeric columns found for feature engineering. "
                           f"Expected at least one of: {numeric_columns}")
    
    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for goals, xG, goals_against, xGA, etc."""
        for lag in self.lags:
            # Core performance metrics
            if 'goals' in df.columns:
                df[f'goals_lag_{lag}'] = df.groupby('team')['goals'].shift(lag)
            if 'xG' in df.columns:
                df[f'xG_lag_{lag}'] = df.groupby('team')['xG'].shift(lag)
            if 'goals_against' in df.columns:
                df[f'goals_against_lag_{lag}'] = df.groupby('team')['goals_against'].shift(lag)
            if 'xGA' in df.columns:
                df[f'xGA_lag_{lag}'] = df.groupby('team')['xGA'].shift(lag)
            
            # Match results
            if 'win' in df.columns:
                df[f'win_lag_{lag}'] = df.groupby('team')['win'].shift(lag)
            if 'loss' in df.columns:
                df[f'loss_lag_{lag}'] = df.groupby('team')['loss'].shift(lag)
            if 'draw' in df.columns:
                df[f'draw_lag_{lag}'] = df.groupby('team')['draw'].shift(lag)
            
            # Home/away indicator
            if 'is_home' in df.columns:
                df[f'is_home_lag_{lag}'] = df.groupby('team')['is_home'].shift(lag)
        
        return df
    
    def _add_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling averages and cumulative features."""
        # Rolling averages (shifted to prevent target leakage)
        if 'goals' in df.columns:
            df['rolling_goals_avg'] = (
                df.groupby('team')['goals']
                .shift(1)
                .ewm(span=self.rolling_window)
                .mean()
                .reset_index(0, drop=True)
            )
        
        if 'xG' in df.columns:
            df['rolling_xG_avg'] = (
                df.groupby('team')['xG']
                .shift(1)
                .ewm(span=self.rolling_window)
                .mean()
                .reset_index(0, drop=True)
            )
        
        if 'goals_against' in df.columns:
            df['rolling_goals_against_avg'] = (
                df.groupby('team')['goals_against']
                .shift(1)
                .ewm(span=self.rolling_window)
                .mean()
                .reset_index(0, drop=True)
            )
        
        if 'xGA' in df.columns:
            df['rolling_xGA_avg'] = (
                df.groupby('team')['xGA']
                .shift(1)
                .ewm(span=self.rolling_window)
                .mean()
                .reset_index(0, drop=True)
            )
        
        # Cumulative sums (shifted to prevent target leakage)
        if 'goals' in df.columns:
            df['cumsum_goals'] = df.groupby('team')['goals'].shift(1).groupby(df['team']).cumsum()
        if 'xG' in df.columns:
            df['cumsum_xG'] = df.groupby('team')['xG'].shift(1).groupby(df['team']).cumsum()
        if 'goals_against' in df.columns:
            df['cumsum_goals_against'] = df.groupby('team')['goals_against'].shift(1).groupby(df['team']).cumsum()
        if 'xGA' in df.columns:
            df['cumsum_xGA'] = df.groupby('team')['xGA'].shift(1).groupby(df['team']).cumsum()
        
        # Efficiency ratios (handling potential division by zero)
        if 'rolling_goals_avg' in df.columns and 'rolling_xG_avg' in df.columns:
            df['goal_efficiency'] = df['rolling_goals_avg'] / df['rolling_xG_avg']
        
        if 'rolling_goals_against_avg' in df.columns and 'rolling_xGA_avg' in df.columns:
            df['defensive_efficiency'] = 1 - (df['rolling_goals_against_avg'] / df['rolling_xGA_avg'])
        
        # Replace infinite values (from division by zero) with NaN
        df.replace([float('inf'), -float('inf')], np.nan, inplace=True)
        
        return df
    
    def _compute_win_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling home and away win rates for each team."""
        if 'win' not in df.columns or 'is_home' not in df.columns:
            return df
        
        df = df.sort_values(by=["team", "datetime"])
        
        # Home win rate
        home_mask = df["is_home"] == 1
        if home_mask.any():
            df.loc[home_mask, "home_win_rate"] = (
                df.loc[home_mask]
                .groupby("team")["win"]
                .shift(1)
                .transform(lambda x: x.ewm(span=self.rolling_window).mean())
            )
        
        # Away win rate
        away_mask = df["is_home"] == 0
        if away_mask.any():
            df.loc[away_mask, "away_win_rate"] = (
                df.loc[away_mask]
                .groupby("team")["win"]
                .shift(1)
                .transform(lambda x: x.ewm(span=self.rolling_window).mean())
            )
        
        return df
    
    def _compute_form_and_streaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute recent form and win/loss streaks."""
        if not all(col in df.columns for col in ['win', 'draw', 'loss']):
            return df
        
        df = df.copy()
        
        # Ensure numeric columns
        df['win'] = df['win'].astype(int)
        df['draw'] = df['draw'].astype(int)
        df['loss'] = df['loss'].astype(int)
        
        # Sort data by team and match date
        df = df.sort_values(by=['team', 'datetime']).reset_index(drop=True)
        
        # Compute recent form (sum of last N games excluding the current one)
        df['recent_form'] = (
            df.groupby('team')['win'].shift(1).rolling(self.rolling_window, min_periods=1).sum() * 3 +
            df.groupby('team')['draw'].shift(1).rolling(self.rolling_window, min_periods=1).sum()
        )
        
        # Function to compute streaks
        def compute_streak(series):
            streak = (series != 0).astype(int)
            return streak.groupby((series == 0).cumsum()).cumsum()
        
        # Compute win/loss streaks using past data only
        df['win_streak'] = df.groupby('team')['win'].apply(
            lambda x: compute_streak(x.shift(1))
        ).reset_index(level=0, drop=True)
        
        df['loss_streak'] = df.groupby('team')['loss'].apply(
            lambda x: compute_streak(x.shift(1))
        ).reset_index(level=0, drop=True)
        
        # Compute unbeaten streak
        def compute_unbeaten_streak(team_df):
            unbeaten_indicator = (team_df['win'].shift(1) + team_df['draw'].shift(1)) > 0
            return compute_streak(unbeaten_indicator.astype(int))
        
        df['unbeaten_streak'] = df.groupby('team').apply(
            compute_unbeaten_streak
        ).reset_index(level=0, drop=True)
        
        return df
    
    def _compute_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rest days between matches."""
        if 'datetime' not in df.columns:
            return df
        
        df['prev_match_date'] = df.groupby('team')['datetime'].shift(1)
        df['days_since_last_match'] = (df['datetime'] - df['prev_match_date']).dt.days
        df = df.drop(columns=['prev_match_date'])
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on the specified method."""
        if self.fill_method == 'ffill':
            # Forward fill within each team group
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df.groupby('team')[col].ffill()
        elif self.fill_method == 'bfill':
            # Backward fill within each team group
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df.groupby('team')[col].bfill()
        elif self.fill_method == 'median':
            # Fill with median values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif self.fill_method == 'mean':
            # Fill with mean values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        return df
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if not self.is_fitted_:
            raise ValueError("This FootballDataTransformer instance is not fitted yet.")
        
        return self.feature_names_out_
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self

