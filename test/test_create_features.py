import unittest
import pandas as pd
from bill2predict.features.create_features import create_date_features, create_lag_features, create_rolling_window_features

class TestCreateFeatures(unittest.TestCase):
    def test_create_date_features_creates_date_features(self):
        df = pd.DataFrame({'Dataframe': ['16/06/2025 13:42:11']})
        result_df = create_date_features(df)

        expected = {
            'hour': 13,
            'day_of_week': 0,
            'month': 6,
            'weekday': True,
            'weekend': False
        }

        for feature in expected.keys:
            self.assertIn(feature, result_df.columns)
            pd.testing.assert_series_equal(result_df[feature], pd.Series(expected[feature]))
    
    def test_create_lag_features_creates_lag_features(self): 
        raise NotImplementedError
    
    def test_create_lag_features_has_no_missing_values(self):
        raise NotImplementedError

    def test_create_rolling_window_features_creates_rolling_window_features(self):
        raise NotImplementedError

    def test_create_rolling_windows_features_has_no_missing_values(self):
        raise NotImplementedError
