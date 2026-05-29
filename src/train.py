"""
Train and Evaluate
Scripts to handle 3D sequence creation, the main training loop, and scoring.
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

def create_sequences(data, target_col_index, time_steps=6):
    """
    Format 2D table into 3D sliding windows for the LSTM.
    Returns X (history) and y (next step target).
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        # grab 'time_steps' block
        X.append(data[i:(i + time_steps), :])
        # target is the row right after the block
        y.append(data[i + time_steps, target_col_index])
        
    return np.array(X), np.array(y)

def calculate_metrics(y_true, y_pred):
    """
    Calculate MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # avoid division by zero for MAPE
    epsilon = 1e-10 
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def train_deep_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=64, model_name="trained_model.h5"):
    """
    Run training fit loop with early stopping.
    """
    # Stop if validation loss doesn't improve for 10 epochs
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    # Save the best model physically
    checkpoint = ModelCheckpoint(f'../models/{model_name}', monitor='val_loss', save_best_only=True, verbose=0)
    
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    return history

if __name__ == "__main__":
    print("Run the Model_Training_Pipeline.ipynb notebook instead to start training.")
