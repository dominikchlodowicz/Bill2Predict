import unittest
import pandas as pd
import numpy as np
from bill2predict.data.clean_uci_data import merge_date_time, drop_empty_days, drop_multicollinearity_features

class TestCleanUciData(unittest.TestCase):
    numeric_cols = [
        'Global_active_power',
        'Global_reactive_power',
        'Voltage',
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3'
    ]

    def test_merge_date_time_creates_datetime_column(self):
        data = {'Date':['13/04/2011', '12/01/2001'],
                 'Time':['14:04:32', '08:03:23']}
        df = pd.DataFrame(data)
        result = merge_date_time(df.copy())
        # Should drop Date and Time
        self.assertNotIn('Date', result.columns)
        self.assertNotIn('Time', result.columns)
        # Should create Datetime
        self.assertIn('Datetime', result.columns)
        # Check type
        self.assertTrue(
            pd.api.types.is_datetime64_ns_dtype(result['Datetime']),
            msg=f'Expected datetime64 dtype, got {result['Datetime'].dtype}.'
        )
        # Check values
        expected = pd.Series(pd.to_datetime(
            ['13/04/2011 14:04:32', '12/01/2001 08:03:23'],
            format='%d/%m/%Y %H:%M:%S',
            dayfirst=True
        ))
        pd.testing.assert_series_equal(result['Datetime'], expected, check_names=False)


    def test_merge_date_time_error_if_missing_time(self):        
        df = pd.DataFrame({'Date': ['13/04/2011', '12/01/2001']})
        with self.assertRaises(ValueError) as caught:
            merge_date_time(df)
        self.assertIn("Date or Time", str(caught.exception))
    

    def test_merge_date_time_error_if_missing_date(self):        
        df = pd.DataFrame({'Time':['14:04:32', '08:03:23']})
        with self.assertRaises(ValueError) as caught:
            merge_date_time(df)
        self.assertIn("Date or Time", str(caught.exception))

    def test_drop_empty_days_drops_days_exceeding_threshold(self):
        # Create a DataFrame with two days: one with missing > threshold, one below
        dates = pd.to_datetime([
            '2020-01-01 00:00:00', '2020-01-01 01:00:00',
            '2020-01-02 00:00:00', '2020-01-02 01:00:00'
        ])
        df = pd.DataFrame({
            'Datetime': dates,
            'Global_active_power': [1.0, np.nan, 1.0, 1.0],
            'Global_reactive_power': [1.0, np.nan, 1.0, 1.0],
            'Voltage': [1.0, np.nan, 1.0, 1.0],
            'Global_intensity': [1.0, np.nan, 1.0, 1.0],
            'Sub_metering_1': [1.0, np.nan, 1.0, 1.0],
            'Sub_metering_2': [1.0, np.nan, 1.0, 1.0],
            'Sub_metering_3': [1.0, np.nan, 1.0, 1.0],
        })
        result = drop_empty_days(df, threshold=49.0)

        expected = pd.to_datetime([
            '2020-01-02 00:00:00',
            '2020-01-02 01:00:00'
        ])
        self.assertListEqual(list(result['Datetime']), list(expected))


    def test_drop_empty_days_no_days_dropped_below_threshold(self):
        # All days have missing below threshold
        dates = pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 01:00:00'])
        df = pd.DataFrame({
            'Datetime': dates,
            'Global_active_power': [1.0, np.nan],
            'Global_reactive_power': [1.0, np.nan],
            'Voltage': [1.0, np.nan],
            'Global_intensity': [1.0, np.nan],
            'Sub_metering_1': [1.0, np.nan],
            'Sub_metering_2': [1.0, np.nan],
            'Sub_metering_3': [1.0, np.nan],
        })
        result = drop_empty_days(df, threshold=60.0)
        self.assertEqual(len(result), 2)

    def test_drop_empty_days_raises_on_invalid_threshold(self):
        dates = pd.to_datetime(['2020-01-01 00:00:00'])
        df = pd.DataFrame({
            'Datetime': dates,
            **{col: [1.0] for col in self.numeric_cols}
        })
        with self.assertRaises(ValueError):
            drop_empty_days(df, threshold=-1)
        with self.assertRaises(ValueError):
            drop_empty_days(df, threshold=101)

    def test_drop_empty_days_raises_if_datetime_missing(self):
        df = pd.DataFrame({
            **{col: [1.0] for col in self.numeric_cols}
        })
        with self.assertRaises(ValueError):
            drop_empty_days(df)

    def test_drop_empty_days_raises_if_numeric_cols_missing(self):
        dates = pd.to_datetime(['2020-01-01 00:00:00'])
        df = pd.DataFrame({
            'Datetime': dates
        })
        with self.assertRaises(ValueError) as cm:
            drop_empty_days(df)
        self.assertIn("Missing required numeric columns", str(cm.exception))

    def test_drop_empty_days_raises_if_no_readings(self):
        dates = pd.to_datetime([])
        df = pd.DataFrame({
            'Datetime': dates,
            **{col: pd.Series([], dtype=float) for col in self.numeric_cols}
        })
        with self.assertRaises(ValueError):
            drop_empty_days(df)

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame()
        result = drop_multicollinearity_features(df)
        self.assertTrue(result.empty)

    def test_single_column_df_returns_same(self):
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = drop_multicollinearity_features(df)
        pd.testing.assert_frame_equal(result, df)

    def test_no_cols_dropped_when_no_corr_exceeds_threshold(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [3, 1, 2]
        })
        result = drop_multicollinearity_features(df, threshold=0.99)
        pd.testing.assert_frame_equal(result, df)
        
if __name__ == '__main__':
    unittest.main()

