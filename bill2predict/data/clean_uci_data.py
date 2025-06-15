import pandas as pd
import numpy as np


def merge_date_time(df: pd.DataFrame) -> pd.DataFrame:
    """Merges *df* 'Date' and 'Time' columns into 'Datetime' with dd/mm/yyyy hh:mm:ss format."""
    if 'Time' not in df.columns or 'Date' not in df.columns:
        raise ValueError('Missing required Date or Time columns.')

    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S',
        dayfirst=True
    )

    df.drop(columns=['Date', 'Time'], inplace=True)

    return df


def drop_empty_days(df: pd.DataFrame, threshold: float = 49.0) -> pd.DataFrame:
    """Drop all rows from *df* for any date on which more than *threshold* percent of values are missing."""
    
    if not (0 <= threshold <= 100):
        raise ValueError(f"threshold must be between 0 and 100; got {threshold}.")

    if 'Datetime' not in df.columns:
        raise ValueError('Parse error: Datetime column missing from dataframe.')
    
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

    missing = [c for c in numeric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required numeric columns: {missing}.")

    missing_mask = df[numeric_cols].isna().any(axis=1)

    # number of reading per one day
    total_count_of_values = df.groupby(date_only).size()

    if total_count_of_values.empty:
        raise ValueError("Not enough readings per day.")

    missing_values = missing_mask.groupby(date_only).sum()

    ratio_df = pd.DataFrame({
        'total_count_of_values': total_count_of_values,
        'missing_values': missing_values
    })

    ratio_df['ratio'] = (ratio_df['missing_values'] / ratio_df['total_count_of_values']) * 100
    days_to_delete = ratio_df[ratio_df['ratio'] > threshold].index
    df = df[~date_only.isin(days_to_delete)].copy()

    return df


def drop_multicollinearity_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    "Drop multicollinearity features based on the result of pearson's *|r|* result between features."
    corr = df.corr()
    
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    upper = corr.where(mask)

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    df = df.drop(columns=to_drop).copy()

    return df
