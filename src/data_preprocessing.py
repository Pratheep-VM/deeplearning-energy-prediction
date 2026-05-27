"""
Data Preprocessing Module
Handles loading the dataset, general cleaning, missing value imputation, 
outlier detection, and data scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(file_path):
    """Loads the dataset and sets the datetime index."""
    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df

def handle_missing_values(df):
    """Interpolates missing values using time-based method and drops random variables."""
    # Drop rv1 and rv2 as they are purely random and not useful for prediction
    cols_to_drop = ['rv1', 'rv2']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Time-based interpolation
    df = df.interpolate(method='time')
    return df

def treat_outliers(df, columns=None, upper_percentile=0.99):
    """Caps extreme outliers at the 99th percentile to prevent model skewing."""
    if columns is None:
        columns = ['Appliances'] # Usually, we mostly care about capping target spikes
        
    for col in columns:
        if col in df.columns:
            upper_limit = df[col].quantile(upper_percentile)
            # Clip values to the upper limit (Winsorization)
            df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
    return df

def scale_data(train_df, test_df):
    """
    Scales features using MinMaxScaler to [0, 1] range.
    Fits the scaler ONLY on the training data to prevent data leakage.
    """
    scaler = MinMaxScaler()
    
    # Fit on train, transform both
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df), 
                                columns=train_df.columns, 
                                index=train_df.index)
    
    test_scaled = pd.DataFrame(scaler.transform(test_df), 
                               columns=test_df.columns, 
                               index=test_df.index)
    
    return train_scaled, test_scaled, scaler

def split_time_series(df, train_ratio=0.8):
    """Splits time-series data ensuring temporal consistency (no shuffling)."""
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df

if __name__ == "__main__":
    # Quick sanity check for local testing
    print("Testing preprocessing module...")
    try:
        sample_df = load_data('../data/raw/energy_data_set.csv')
        print(f"Data loaded successfully. Shape: {sample_df.shape}")
    except FileNotFoundError:
        print("Data file not found. Make sure to download energy_data_set.csv into data/raw/")
