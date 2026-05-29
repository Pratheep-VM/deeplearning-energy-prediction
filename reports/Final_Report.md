# Final Report: Appliance Energy Prediction

## 1. Quick Summary
For this project, I built a machine learning pipeline to predict appliance energy consumption using multivariate time-series data. I focused heavily on preventing data leakage and used an LSTM deep learning model. The model does a pretty good job of forecasting short-term energy demand based on recent usage and weather constraints.

## 2. What I Did

### Data Preprocessing
- **Missing Values:** I used time-based interpolation to fill in any gaps so we didn't throw off the sequence.
- **Noise:** Dropped the `rv1` and `rv2` columns since they were completely random variables and just added noise.
- **Outliers:** I capped the top 1% of energy usage values (Winsorization). There were a few crazy spikes that I felt were messing up the model's training process.
- **Scaling:** Used `MinMaxScaler`. I made sure to fit the scaler *only* on the training data so that test set stats didn't accidentally leak into training.

### Feature Engineering
- **Time Features:** Added basic stuff like Hour, Day of Week, and a Weekend indicator. I also added some sine/cosine cyclical encoding since neural networks don't naturally understand that 11 PM and 12 AM are right next to each other.
- **Rolling Windows & Lags:** Created 1-hour and 3-hour rolling averages. The most important part here was doing a `.shift(1)` before the rolling stats to make sure the model isn't looking at "current" data when guessing the "current" target.
- **Interactions:** Added a `Temp_diff_indoor_outdoor` feature (`T1 - T_out`) to help capture heating/cooling load nicely.

### Modeling
- **Data Shape:** Shifted the 2D tabular data into 3D arrays: `(batch_size, time_steps=6, features)`. This gave the model 1 hour of past context per prediction block.
- **Setup:** I started with a Random Forest Regressor just to get a baseline score, and then built an LSTM network with hidden sizes of 64 and 32.
- **Regularization:** Added `Dropout(0.2)` layers to drop 20% of the neurons each pass, which helped keep the model from overfitting.
- **Training:** Used the Adam optimizer and Mean Squared Error (MSE). I also threw in an `EarlyStopping` callback to automatically stop training once the validation loss stopped improving.

## 3. Results

**Baseline Model (Random Forest)**
- Mean Absolute Error (MAE): 0.1227
- Root Mean Squared Error (RMSE): 0.1760

**Deep Learning Model (LSTM)**
- Mean Absolute Error (MAE): 0.0439
- Root Mean Squared Error (RMSE): 0.0992

The LSTM performed significantly better than the baseline. Looking at the plots, it mostly catches the daily spikes really well while ignoring the small erratic noise.

## 4. Final Thoughts
Overall, the pipeline works well for time-series forecasting. Handling the temporal data shift correctly and throwing in cyclical feature engineering made the biggest difference in keeping the LSTM stable.

### Next Steps / Ideas
1. It would be cool to add some external API data (like local holidays) since behaviors change drastically during off-days.
2. I'd like to try running this with different `time_steps` (maybe a 2-hour or 4-hour lookback window) to see if having more historical context improves things further.