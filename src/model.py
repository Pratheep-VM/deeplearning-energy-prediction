"""
Model Development Module
Defines Baseline (Scikit-Learn) and Deep Learning (LSTM, GRU) architectures for time-series forecasting.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

def get_baseline_model(model_type='rf'):
    """
    Returns a traditional Machine Learning model to serve as a performance benchmark.
    model_type: 'rf' for Random Forest, 'lr' for Linear Regression.
    """
    if model_type == 'rf':
        # n_jobs=-1 uses all CPU cores for faster training
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        return LinearRegression()

def build_lstm_model(input_shape, learning_rate=0.001):
    """
    Builds a standard LSTM architecture for time-series forecasting.
    input_shape: (time_steps, number_of_features)
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),  # Prevents overfitting by randomly turning off 20% of neurons
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)       # Linear activation output for predicting continuous energy values
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model

def build_gru_model(input_shape, learning_rate=0.001):
    """
    Builds a GRU (Gated Recurrent Unit) architecture. 
    GRUs are often faster to train than LSTMs and perform similarly on datasets of this size.
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
    # If run directly, just print out a dummy summary to verify architecture
    print("Testing LSTM Compilation...")
    dummy_model = build_lstm_model((6, 15))
    dummy_model.summary()
