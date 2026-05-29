"""
Model definitions
Contains a basic RF baseline and the LSTM/GRU deep learning models.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input
from tensorflow.keras.optimizers import Adam

def get_baseline_model(model_type='rf'):
    """
    Get a standard ML model to use as a baseline.
    """
    if model_type == 'rf':
        # n_jobs=-1 to use all cores
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        return LinearRegression()

def build_lstm_model(input_shape, learning_rate=0.001):
    """
    Standard LSTM for time-series.
    input_shape: (time_steps, features)
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),  # drop 20% of nodes to help with overfitting
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)       # linear output for regression
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model

def build_gru_model(input_shape, learning_rate=0.001):
    """
    GRU version (usually faster to train than LSTM).
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model

if __name__ == "__main__":
    print("Testing LSTM build...")
    dummy_model = build_lstm_model((6, 15))
    dummy_model.summary()
