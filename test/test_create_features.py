import pandas as pd
import numpy as np
import unittest
from bill2predict.features.create_features import create_date_features, create_lag_features, create_rolling_window_features

class TestCreateFeatures(unittest.TestCase):
    # create mock dataframe
    mock_df = pd.DataFrame(
            np.random.default_rng(31).standard_normal((1_000, 1)),
            index = pd.date_range("2024-01-01", periods=1_000, freq="D"),
            columns=['Global_active_power']
    )

    lag_feature_names = [
        'lag_1h', 
        'lag_2h', 
        'lag_3h', 
        'lag_24h',
        'lag_168h'
    ]

    rolling_window_feature_names = [
        'roll_mean_3h',
        'roll_mean_24h',
        'roll_mean_168h'
    ]

    def test_create_date_features_creates_date_features(self):
        df = pd.DataFrame({'Datetime': ['16/06/2025 13:42:11']})
        df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=True, errors='raise')
        result_df = create_date_features(df)

        expected = {
            'hour': 13,
            'day_of_week': 0,
            'month': 6,
            'weekday': True,
            'weekend': False
        }

        for feature in expected:
            self.assertIn(feature, result_df.columns)
            pd.testing.assert_series_equal(result_df[feature], pd.Series(expected[feature], name=feature), check_dtype=False)
    
    def test_create_lag_features_creates_lag_features(self): 
        result = create_lag_features(self.mock_df)
        for feature_name in self.lag_feature_names:
            self.assertIn(feature_name, result.columns)
    
    def test_create_lag_features_has_no_missing_values(self):
        result = create_lag_features(self.mock_df)
        self.assertFalse(result.isna().values.any())

    def test_create_rolling_window_features_creates_rolling_window_features(self):
        result = create_rolling_window_features(self.mock_df)
        for feature_name in self.rolling_window_feature_names:
            self.assertIn(feature_name, result.columns)

    def test_create_rolling_windows_features_has_no_missing_values(self):
        result = create_rolling_window_features(self.mock_df)
        self.assertFalse(result.isna().values.any())
