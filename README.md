# Appliance Energy Prediction Using Deep Learning

## Overview
This project focuses on predicting the energy consumption of appliances in a low-energy building using a multivariate time-series dataset. The pipeline encompasses robust data preprocessing, temporal feature engineering, and the implementation of Deep Learning models (LSTM and GRU) to forecast energy usage accurately.

## Methodology & Engineering Choices

### 1. Data Preprocessing
- **Missing Values & Noise:** Interpolated missing values temporally. Dropped strictly random variables (`rv1`, `rv2`) to reduce noise.
- **Outlier Handling:** Applied Winsorization at the 99th percentile to cap extreme energy spikes, maintaining model stability.
- **Data Leakage Prevention:** `MinMaxScaler` was fitted **strictly on the training set** and applied to the test set. This ensures no future data distribution information leaks into the training phase.

### 2. Feature Engineering
Time-series forecasting requires explicit temporal features. We engineered:
- **Datetime Components:** Hour, day of the week, and weekend binary flags.
- **Rolling & Lagged Features:** Implemented 1-hour and 3-hour rolling averages. **Crucially, we applied a `.shift(1)` operation** prior to rolling calculations to ensure the model never accidentally targets current-interval data (preventing temporal leakage).
- **Interaction Terms:** Created domain-specific features, such as the indoor/outdoor temperature differential (`T1 - T_out`), which directly impacts HVAC/heating load.

### 3. Model Architecture
- **Data Formatting:** Transformed tabular 2D pandas data into the 3D tensor format `(batch_size, time_steps, features)` required by Keras recurrent layers. Used a sliding window approach with `time_steps=6` (1 hour of historical context context).
- **LSTM / GRU:** Deployed Long Short-Term Memory (LSTM) networks to capture non-linear temporal dependencies. 
- **Regularization:** Integrated `Dropout(0.2)` layers to organically prevent overfitting by dropping 20% of neuron connections during training.
- **Training Constraints:** Utilized the `Adam` optimizer with `Mean Squared Error` loss (which heavily penalizes large forecasting errors) and `EarlyStopping` monitoring validation loss to halt training when the model stops generalizing.

## Project Structure
```text
├── data/ 
│   ├── raw/                 # Put the downloaded dataset here 
│   └── processed/           # Processed datasets ready for modeling
├── notebooks/ 
│   ├── EDA.ipynb            # Exploratory Data Analysis & visual checks
│   └── Model_Training_Pipeline.ipynb # End-to-end execution script
├── src/ 
│   ├── data_preprocessing.py # Cleaning, outlier handling, and scaling
│   ├── feature_engineering.py # Temporal feature extraction
│   ├── model.py             # Model definitions (Baseline, LSTM, GRU)
│   └── train.py             # Sequence generation and training loop
├── models/                  # Saved model artifacts (.h5)
├── reports/                 # Final assessment reports
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Setup & Execution

1. **Environment Setup**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Data Placement**: 
   Download the "Appliance Energy Prediction Dataset" and save it in the `data/raw/` directory as `energy_data_set.csv`.

3. **Running the Pipeline**:
   Open VS Code, select your `.venv` python interpreter, and run the `notebooks/Model_Training_Pipeline.ipynb` notebook from top to bottom. It will map through the `src/` modules, train the LSTM, and output the final prediction graphs and Evaluation Metrics (MAE, RMSE, MAPE).
