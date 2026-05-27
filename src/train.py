"""
Model Training and Evaluation Module
Handles creating 3D sequences, training loops, early stopping, and evaluation.
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

def create_sequences(data, target_col_index, time_steps=6):
    """
    Transforms 2D tabular data into the 3D sliding-window format required by LSTMs/GRUs.
    Returns X (features over time) and y (the target to predict at the next step).
    """
    X, y = [], []
    # Slide the window down the dataset
    for i in range(len(data) - time_steps):
        # Extract the block of 'time_steps' rows
        X.append(data[i:(i + time_steps), :])
        # The target is the value exactly one step after the block ends
        y.append(data[i + time_steps, target_col_index])
        
    return np.array(X), np.array(y)

def calculate_metrics(y_true, y_pred):
    """
    Calculates regression evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (Mean Absolute Percentage Error) needs protection against division by zero
    epsilon = 1e-10 
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def train_deep_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=64, model_name="trained_model.h5"):
    """
    Trains the deep learning model with Early Stopping to prevent overfitting.
    Saves the best weights to the /models/ folder.
    """
    # Callback 1: Stop training if the validation loss doesn't improve for 10 epochs
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    # Callback 2: Save the physical model file to disk whenever it hits a new best score
    checkpoint = ModelCheckpoint(f'../models/{model_name}', monitor='val_loss', save_best_only=True, verbose=0)
    
    print(f"Starting training loop for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        verbose=1 # Prints the progress bar
    )
    
    return history

if __name__ == "__main__":
    print("Training module contains pipeline logic. Please run the notebook pipeline instead.")
