"""
Data Preprocessing Module
Handles loading, cleaning, and scaling the data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(file_path):
    """Loads the csv and sets date as the index."""
    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df

def handle_missing_values(df):
    """Fills NaNs with time-based interpolation and drops useless columns."""
    # Drop rv1 and rv2 (they are just random noise)
    cols_to_drop = ['rv1', 'rv2']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Fill missing values based on time
    df = df.interpolate(method='time')
    return df

def treat_outliers(df, columns=None, upper_percentile=0.99):
    """Capping the top 1% to handle crazy energy spikes."""
    if columns is None:
        columns = ['Appliances'] # We mainly just want to cap the target
        
    for col in columns:
        if col in df.columns:
            upper_limit = df[col].quantile(upper_percentile)
            # Clip values (Winsorization)
            df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
    return df

def scale_data(train_df, test_df):
    """
    Scale features to 0-1.
    Important: Only fit the scaler on train to avoid data leakage!
    """
    scaler = MinMaxScaler()
    
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df), 
                                columns=train_df.columns, 
                                index=train_df.index)
    
    test_scaled = pd.DataFrame(scaler.transform(test_df), 
                               columns=test_df.columns, 
                               index=test_df.index)
    
    return train_scaled, test_scaled, scaler

def split_time_series(df, train_ratio=0.8):
    """Standard time-series split (don't shuffle!)"""
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df

if __name__ == "__main__":
    print("Testing preprocessing...")
    try:
        sample_df = load_data('../data/raw/energy_data_set.csv')
        print(f"Loaded. Shape: {sample_df.shape}")
    except FileNotFoundError:
        print("Couldn't find the data. Check data/raw/energy_data_set.csv")
