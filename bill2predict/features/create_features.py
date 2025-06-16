import pandas as pd

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new columns in the dataframe, which split the data into parts of the week."""
    df['hour'] = df['Datetime'].dt.hour
    df['day_of_week'] = df['Datetime'].dt.weekday
    df['month'] = df['Datetime'].dt.month
    df['weekday'] = df['day_of_week'] <= 5
    df['weekend'] = df['day_of_week'] > 5
    return df

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new columns in the dataframe, that are past datapoints."""
    # lag featues
    df['lag_1h'] = df['Global_active_power'].shift(1)
    df['lag_2h'] = df['Global_active_power'].shift(2)
    df['lag_3h'] = df['Global_active_power'].shift(3)
    #daily
    df['lag_24h'] = df['Global_active_power'].shift(24)
    #weekly
    df['lag_168h'] = df['Global_active_power'].shift(168)
    # empty rows cleanup, that are being created from the first datapoints.
    df = df.dropna().copy()
    return df

def create_rolling_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new columns in the dataframe, with calculated mean values for the past data points"""
    # rolling windows: 3h, 24h, 168h
    for window in [3, 24, 168]:
        df[f'roll_mean_{window}h'] = (
            df['Global_active_power']
                .rolling(window=window)
                .mean()
                .shift(1)
        )
    # empty rows cleanup, that are being created from the first datapoints. 
    df = df.dropna().copy()
    return df
