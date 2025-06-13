import unittest
import pandas as pd
import numpy as np
from bill2predict.data.clean_uci_data import merge_date_time, drop_empty_days, drop_multicollinearity_features

class TestCleanUciData(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()

