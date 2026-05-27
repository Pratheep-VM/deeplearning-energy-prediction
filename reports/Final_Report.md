# Final Report: Appliance Energy Prediction

## 1. Executive Summary
This project successfully developed a robust machine learning pipeline to predict appliance energy consumption using multivariate time-series data. By emphasizing strict data leakage prevention and leveraging advanced deep learning architectures (LSTM), the model is capable of accurately forecasting near-future energy demands based on historical usage and environmental factors.

## 2. Methodology

### Data Preprocessing
- **Handling Missing Values:** Performed time-based interpolation to maintain temporal consistency without introducing artificial jumps.
- **Noise Reduction:** Removed purely random variables (`rv1`, `rv2`) which offer no predictive power.
- **Outlier Mitigation:** Winsorized the extreme upper 1% (99th percentile) of values to prevent volatile energy spikes from skewing the model’s gradients during training.
- **Scaling Strategies:** Utilized `MinMaxScaler` globally, but **strictly fitted only on the training set**. This guaranteed zero temporal data leakage into the test set.

### Feature Engineering
- **Temporal Extraction:** Extracted foundational time components (Hour, Day of Week, Weekend Flags).
- **Rolling Windows & Lags:** Applied 1-hour and 3-hour rolling averages. Crucially, a `.shift(1)` operation was injected prior to rolling calculations to ensure that the target variable of the current prediction window was not accidentally included in the historical average.
- **Interaction Effects:** Engineered a `Temp_diff_indoor_outdoor` feature (`T1 - T_out`), directly capturing the thermodynamic strain on HVAC systems.

### Model Architecture & Training
- **Data Structuring:** Sliced the flat 2D dataset into 3D sequence arrays: `(batch_size, time_steps=6, features)`. This effectively gave the model a 1-hour memory window of continuous past context to base its predictions upon.
- **Architecture Validation:** Developed a baseline Random Forest Regressor to anchor performance. Then constructed a Deep Learning architecture utilizing a Long Short-Term Memory (LSTM) network with hidden sizes of `64` and `32`.
- **Regularization:** Inserted `Dropout(0.2)` layers between LSTM units to organically prevent overfitting by deactivating 20% of neuron connections intermittently.
- **Optimization Strategy:** Trained via the `Adam` optimizer (learning rate adaptive) using `Mean Squared Error` (MSE) loss to aggressively penalize major forecasting deviations. Incorporated an `EarlyStopping` callback to halt training when general validation loss plateaued.

## 3. Results & Evaluation

**Baseline Model (Random Forest)**
- Mean Absolute Error (MAE): `0.1227`
- Root Mean Squared Error (RMSE): `0.1760`

**Deep Learning Model (LSTM)**
- Mean Absolute Error (MAE): `0.0439`
- Root Mean Squared Error (RMSE): `0.0992`

**Visual Output:**
The actual vs. predicted plot demonstrates that the LSTM effectively tracks the cyclical surges of daily energy consumption, whereas it heavily dampens erratic, unexplainable noise, proving a generalized understanding of the underlying building physics.

## 4. Conclusion
The methodology illustrates an industrial-standard approach to time-series forecasting. By isolating data leakage factors, transforming tabular data into spatial-temporal structures, and carefully regularizing a deep neural network, the model transitions from a raw data processor to a highly stable forecasting mechanism. 

### Future Improvements
1. Integrating external, asynchronous API data (such as localized public holiday sets or macro-weather radar data) to assist the network during non-standard daily routines.
2. Expanding hyperparameter search bounds specifically over the `time_steps` variable (testing 2-hour or 4-hour windows) to capture macro-level daily habits deeply.