"""
Feature Engineering Module
Handles the creation of time-based features, rolling averages, 
lagged features, and interaction terms.
"""

import pandas as pd
import numpy as np

def create_time_features(df):
    """Extracts date/time components and encodes them cyclically for deep learning."""
    df = df.copy()
    hour = df.index.hour
    day_of_week = df.index.dayofweek
    month = df.index.month
    
    df['hour'] = hour
    df['day_of_week'] = day_of_week
    df['month'] = month
    
    # 0-4 are Monday-Friday, 5-6 are Saturday-Sunday
    df['is_weekend'] = day_of_week.isin([5, 6]).astype(int)
    
    # Cyclical encoding for time features (crucial for neural networks)
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
    df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    
    return df

def create_rolling_features(df, target_col='Appliances', weather_cols=['T_out', 'RH_out'], windows=[6, 18, 36]): 
    """
    Computes rolling statistics (mean, std, min, max) for target and weather features.
    windows: 6 (1h), 18 (3h), 36 (6h).
    """
    df = df.copy()
    
    # Target rolling stats (shifted to prevent data leakage)
    for w in windows:
        df[f'{target_col}_roll_mean_{w}'] = df[target_col].shift(1).rolling(window=w).mean()
        df[f'{target_col}_roll_std_{w}'] = df[target_col].shift(1).rolling(window=w).std()
        df[f'{target_col}_roll_max_{w}'] = df[target_col].shift(1).rolling(window=w).max()
        df[f'{target_col}_roll_min_{w}'] = df[target_col].shift(1).rolling(window=w).min()
        
    # Weather rolling stats (no shift needed for exogenous features if at current time, 
    # but shifted if we only have past info. Standard is no shift for weather, but lets shift to be safe for forecasting)
    for col in weather_cols:
        if col in df.columns:
            for w in [6, 18]:
                df[f'{col}_roll_mean_{w}'] = df[col].shift(1).rolling(window=w).mean()
    
    return df

def create_lagged_features(df, target_col='Appliances', lags=[1, 3, 6, 12, 144]):
    """
    Creates lagged features using past values.
    lag=1 (10m), 3 (30m), 6 (1h), 12 (2h), 144 (24h)
    """
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
    # Add diff features (rate of change)
    df[f'{target_col}_diff_1'] = df[target_col].shift(1) - df[target_col].shift(2)
        
    return df

def create_interaction_features(df):
    """Generates interaction terms and aggregated sensor features."""
    df = df.copy()
    
    # Differential features
    if 'T1' in df.columns and 'T_out' in df.columns:
        df['Temp_diff_indoor_outdoor'] = df['T1'] - df['T_out']
    
    if 'RH_1' in df.columns and 'RH_out' in df.columns:
        df['Humidity_diff_indoor_outdoor'] = df['RH_1'] - df['RH_out']
        
    # Average temperatures across all rooms to represent general house state
    temp_cols = [f'T{i}' for i in range(1, 10) if f'T{i}' in df.columns]
    if temp_cols:
        df['Avg_indoor_Temp'] = df[temp_cols].mean(axis=1)
        
    rh_cols = [f'RH_{i}' for i in range(1, 10) if f'RH_{i}' in df.columns]
    if rh_cols:
        df['Avg_indoor_RH'] = df[rh_cols].mean(axis=1)

    return df

def feature_selection(df, target_col='Appliances'):
    """
    Cleans up resulting dataset and handles NaN values generated.
    """
    # Drop columns with high correlation to each other (e.g. rv1 and rv2 are identical)
    if 'rv1' in df.columns and 'rv2' in df.columns:
        df = df.drop(columns=['rv2'])
        
    # Lagging and rolling creates NaNs at the beginning
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    print("Feature Engineering module ready to be imported.")
