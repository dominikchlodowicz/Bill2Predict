import pandas as pd

def merge_date_time(df:pd.DataFrame) -> pd.DataFrame:
    """Merges *df* 'Date' and 'Time' columns into 'Datetime' with dd/mm/yyyy hh:mm:ss format."""
    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S',
        dayfirst=True
    )

    df.drop(columns=['Date', 'Time'], inplace=True)

    return df


def drop_empty_days(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """Drop all rows from *df* for any date on which more than *threshold* percent of values are missing."""
    date_only = df['Datetime'].dt.date

    numeric_cols = [
        'Global_active_power',
        'Global_reactive_power',
        'Voltage',
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3'
    ]

    missing_mask = df[numeric_cols].isna().any(axis=1)

    # number of reading per one day
    total_count_of_values = df.groupby(date_only).size()

    missing_values = missing_mask.groupby(date_only).sum()

    ratio_df = pd.DataFrame({
        'total_count_of_values': total_count_of_values,
        'missing_values': missing_values
    })

    ratio_df['ratio'] = (ratio_df['missing_values'] / ratio_df['total_count_of_values']) * 100

    days_to_delete = ratio_df[ratio_df['ratio'] > 50.0].index

    df = df[~date_only.isin(days_to_delete)].copy()

    return df