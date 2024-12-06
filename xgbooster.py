

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('tesla_stock_data_final_cleaneddata(noduplciates_nomissingvalues).csv', parse_dates=['timestamp'], index_col='timestamp')

# Create lag features
df['lag_3_high'] = df['high'].shift(3)
df['lag_7_high'] = df['high'].shift(7)
df['lag_14_high'] = df['high'].shift(14)

df['lag_3_low'] = df['low'].shift(3)
df['lag_7_low'] = df['low'].shift(7)
df['lag_14_low'] = df['low'].shift(14)

# Create rolling mean and standard deviation
df['rolling_7_mean_high'] = df['high'].rolling(window=7).mean()
df['rolling_14_mean_high'] = df['high'].rolling(window=14).mean()
df['rolling_7_std_high'] = df['high'].rolling(window=7).std()

df['rolling_7_mean_low'] = df['low'].rolling(window=7).mean()
df['rolling_14_mean_low'] = df['low'].rolling(window=14).mean()
df['rolling_7_std_low'] = df['low'].rolling(window=7).std()

# Trend features
df['trend_high'] = df['high'] - df['lag_7_high']
df['trend_low'] = df['low'] - df['lag_7_low']

# Time-based features
df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month

# Drop missing values caused by lag and rolling operations
df.dropna(inplace=True)

# Define feature matrix X and targets
X = df.drop(['high', 'low'], axis=1)
y_high = df['high']
y_low = df['low']

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Train-test split
X_train, X_test, y_train_high, y_test_high = train_test_split(X, y_high, test_size=0.2, random_state=42)
X_train, X_test, y_train_low, y_test_low = train_test_split(X, y_low, test_size=0.2, random_state=42)

# Train XGBoost models
xgb_model_high = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model_high.fit(X_train, y_train_high)

xgb_model_low = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model_low.fit(X_train, y_train_low)

# Predictions
y_pred_high = xgb_model_high.predict(X_test)
y_pred_low = xgb_model_low.predict(X_test)

# Evaluate performance
mse_high = mean_squared_error(y_test_high, y_pred_high)
mae_high = mean_absolute_error(y_test_high, y_pred_high)
r2_high = r2_score(y_test_high, y_pred_high)

mse_low = mean_squared_error(y_test_low, y_pred_low)
mae_low = mean_absolute_error(y_test_low, y_pred_low)
r2_low = r2_score(y_test_low, y_pred_low)

print(f"High Price - MSE: {mse_high}, MAE: {mae_high}, R2: {r2_high}")
print(f"Low Price - MSE: {mse_low}, MAE: {mae_low}, R2: {r2_low}")

import shap
import pandas as pd

# Assuming your model (xgb_model_high and xgb_model_low) and X_train are already defined

# Initialize SHAP TreeExplainer for both high and low models
explainer_high = shap.TreeExplainer(xgb_model_high)
explainer_low = shap.TreeExplainer(xgb_model_low)

# For large datasets, use a sample subset to speed up SHAP calculations
sample_size = 10000  # You can adjust this sample size based on your data size
X_train_sample = X_train.sample(n=sample_size, random_state=42)

# Compute SHAP values for the sampled training data
shap_values_high = explainer_high.shap_values(X_train_sample)
shap_values_low = explainer_low.shap_values(X_train_sample)

# Summary plot for global feature importance (Top 10 features)
shap.summary_plot(shap_values_high, X_train_sample, max_display=10, plot_size=(10, 6))
shap.summary_plot(shap_values_low, X_train_sample, max_display=10, plot_size=(10, 6))

# Local interpretability: Force plot for a specific instance (index 10 for example)
i = 10  # Choose an index of a sample to explain

shap.initjs()

# Force plot for the high price model
shap.force_plot(explainer_high.expected_value, shap_values_high[i], X_train_sample.iloc[i])

# Force plot for the low price model
shap.force_plot(explainer_low.expected_value, shap_values_low[i], X_train_sample.iloc[i])

# Optionally, calculate and print the average SHAP values for global feature importance
mean_shap_high = shap_values_high.mean(axis=0)
mean_shap_low = shap_values_low.mean(axis=0)

print("Mean SHAP values for high price model:", mean_shap_high)
print("Mean SHAP values for low price model:", mean_shap_low)

import pandas as pd

# Function to generate predictions for a user-defined date
def predict_for_date(input_date, df, xgb_model_high, xgb_model_low):
    """
    Predict high and low prices for a specific future date.

    Parameters:
    - input_date: The future date for which prediction is needed (format: 'YYYY-MM-DD').
    - df: Original DataFrame with historical data.
    - xgb_model_high: Trained model for high price prediction.
    - xgb_model_low: Trained model for low price prediction.

    Returns:
    - A DataFrame with predicted high and low prices for the given date.
    """
    # Ensure input_date is a pandas datetime object
    input_date = pd.to_datetime(input_date)

    # Prepare a single-row DataFrame for the input date
    future_df = pd.DataFrame(index=[input_date])

    # Populate lag and rolling features using the last available data
    for feature in ['high', 'low']:
        for lag in [3, 7, 14]:
            future_df[f'lag_{lag}_{feature}'] = df[feature].iloc[-lag:].mean()
        for window in [7, 14]:
            future_df[f'rolling_{window}_mean_{feature}'] = df[feature].iloc[-window:].mean()

    # Add rolling standard deviation features (if used during training)
    for feature in ['high', 'low']:
        for window in [7]:  # Assuming you used a 7-day rolling window for standard deviation
            future_df[f'rolling_{window}_std_{feature}'] = df[feature].rolling(window).std().iloc[-1]

    # Add trend features
    future_df['trend_high'] = future_df['lag_7_high'] - future_df['lag_14_high']
    future_df['trend_low'] = future_df['lag_7_low'] - future_df['lag_14_low']

    # Add time-based features
    future_df['hour'] = input_date.hour
    future_df['day'] = input_date.day
    future_df['month'] = input_date.month

    # Add missing features (e.g., 'close', 'vwap', 'volume', 'trade_count', 'open') with last known values from the data
    future_df['close'] = df['close'].iloc[-1]
    future_df['vwap'] = df['vwap'].iloc[-1]
    future_df['volume'] = df['volume'].iloc[-1]
    future_df['trade_count'] = df['trade_count'].iloc[-1]
    future_df['open'] = df['open'].iloc[-1]  # Adding 'open' feature

    # Get the feature names from the trained model
    feature_order = xgb_model_high.get_booster().feature_names

    # Reorder the future_df to match the feature order of the model
    future_df = future_df[feature_order]

    # Predict high and low prices
    predicted_high = xgb_model_high.predict(future_df)
    predicted_low = xgb_model_low.predict(future_df)

    # Return predicted high and low prices as a DataFrame
    future_df['predicted_high'] = predicted_high
    future_df['predicted_low'] = predicted_low

    return future_df[['predicted_high', 'predicted_low']]

# Example Usage
input_date = '2024-12-19'  # Specify the future date
result = predict_for_date(input_date, df, xgb_model_high, xgb_model_low)
print(result)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Evaluate High Price Model
y_pred_high_test = xgb_model_high.predict(X_test)
mse_high_test = mean_squared_error(y_test_high, y_pred_high_test)
mae_high_test = mean_absolute_error(y_test_high, y_pred_high_test)
r2_high_test = r2_score(y_test_high, y_pred_high_test)

print(f"High Price Model - Test MSE: {mse_high_test}, Test MAE: {mae_high_test}, Test R2: {r2_high_test}")

# Evaluate Low Price Model
y_pred_low_test = xgb_model_low.predict(X_test)
mse_low_test = mean_squared_error(y_test_low, y_pred_low_test)
mae_low_test = mean_absolute_error(y_test_low, y_pred_low_test)
r2_low_test = r2_score(y_test_low, y_pred_low_test)

print(f"Low Price Model - Test MSE: {mse_low_test}, Test MAE: {mae_low_test}, Test R2: {r2_low_test}")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predictions on the training data
train_pred_high = xgb_model_high.predict(X_train)
train_pred_low = xgb_model_low.predict(X_train)

# High price model metrics
train_mse_high = mean_squared_error(y_train_high, train_pred_high)
train_mae_high = mean_absolute_error(y_train_high, train_pred_high)
train_r2_high = r2_score(y_train_high, train_pred_high)

# Low price model metrics
train_mse_low = mean_squared_error(y_train_low, train_pred_low)
train_mae_low = mean_absolute_error(y_train_low, train_pred_low)
train_r2_low = r2_score(y_train_low, train_pred_low)

# Print results
print(f"High Price Model - Training MSE: {train_mse_high}, Training MAE: {train_mae_high}, Training R2: {train_r2_high}")
print(f"Low Price Model - Training MSE: {train_mse_low}, Training MAE: {train_mae_low}, Training R2: {train_r2_low}")

"""Observations:

Very Close Metrics:


The training and testing errors (MSE and MAE) are very close.

The R² values for both sets are almost identical.

There is no significant performance gap between training and testing data.

No Significant Overfitting:

The model generalizes well to unseen data, indicating that overfitting is not an issue. This suggests that the model has captured the underlying patterns in the data effectively.

Possible Slight Underfitting?


Since both training and test errors are nearly equal and not exactly zero, the model may still slightly underfit, meaning it could be improved marginally with feature engineering or additional complexity. However, these results are strong.

After checking the shap need to add some more features
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('tesla_stock_data_final_cleaneddata(noduplciates_nomissingvalues).csv', parse_dates=['timestamp'], index_col='timestamp')

# Create lag features for high and low prices (including longer lags)
for lag in [3, 7, 14, 30, 60]:
    df[f'lag_{lag}_high'] = df['high'].shift(lag)
    df[f'lag_{lag}_low'] = df['low'].shift(lag)

# Create rolling mean and standard deviation (with longer windows)
for window in [7, 14, 30, 60]:
    df[f'rolling_{window}_mean_high'] = df['high'].rolling(window=window).mean()
    df[f'rolling_{window}_mean_low'] = df['low'].rolling(window=window).mean()
    df[f'rolling_{window}_std_high'] = df['high'].rolling(window=window).std()
    df[f'rolling_{window}_std_low'] = df['low'].rolling(window=window).std()

# Add Exponential Moving Average (EMA) features
for span in [7, 14, 30, 60]:
    df[f'ema_{span}_high'] = df['high'].ewm(span=span, adjust=False).mean()
    df[f'ema_{span}_low'] = df['low'].ewm(span=span, adjust=False).mean()

# Percentage change features (daily, weekly)
df['pct_change_high'] = df['high'].pct_change()
df['pct_change_low'] = df['low'].pct_change()

# Trend features
df['trend_high'] = df['high'] - df['lag_7_high']
df['trend_low'] = df['low'] - df['lag_7_low']

# Time-based features (hour, day, month)
df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month

# Interaction terms (Example: interaction between lag_7 and rolling mean)
df['lag_7_rolling_7_mean_high'] = df['lag_7_high'] * df['rolling_7_mean_high']
df['lag_7_rolling_7_mean_low'] = df['lag_7_low'] * df['rolling_7_mean_low']

# Drop missing values caused by lag and rolling operations
df.dropna(inplace=True)

# Define feature matrix X and targets
X = df.drop(['high', 'low'], axis=1)
y_high = df['high']
y_low = df['low']

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Train-test split
X_train, X_test, y_train_high, y_test_high = train_test_split(X, y_high, test_size=0.2, random_state=42)
X_train, X_test, y_train_low, y_test_low = train_test_split(X, y_low, test_size=0.2, random_state=42)

# Train XGBoost models
xgb_model_high = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model_high.fit(X_train, y_train_high)

xgb_model_low = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model_low.fit(X_train, y_train_low)

# Predictions
y_pred_high = xgb_model_high.predict(X_test)
y_pred_low = xgb_model_low.predict(X_test)

# Evaluate performance
mse_high = mean_squared_error(y_test_high, y_pred_high)
mae_high = mean_absolute_error(y_test_high, y_pred_high)
r2_high = r2_score(y_test_high, y_pred_high)

mse_low = mean_squared_error(y_test_low, y_pred_low)
mae_low = mean_absolute_error(y_test_low, y_pred_low)
r2_low = r2_score(y_test_low, y_pred_low)

print(f"High Price - MSE: {mse_high}, MAE: {mae_high}, R2: {r2_high}")
print(f"Low Price - MSE: {mse_low}, MAE: {mae_low}, R2: {r2_low}")

# Check if 'open' and 'volume' are in the DataFrame
print(df.columns)

def predict_for_date(input_date, df, xgb_model_high, xgb_model_low):
    """
    Predict high and low prices for a specific future date.

    Parameters:
    - input_date: The future date for which prediction is needed (format: 'YYYY-MM-DD').
    - df: Original DataFrame with historical data.
    - xgb_model_high: Trained model for high price prediction.
    - xgb_model_low: Trained model for low price prediction.

    Returns:
    - A DataFrame with predicted high and low prices for the given date.
    """
    # Ensure input_date is a pandas datetime object
    input_date = pd.to_datetime(input_date)

    # Prepare a single-row DataFrame for the input date
    future_df = pd.DataFrame(index=[input_date])

    # Populate lag and rolling features using the last available data
    for feature in ['high', 'low']:
        for lag in [3, 7, 14, 30, 60]:
            future_df[f'lag_{lag}_{feature}'] = df[feature].iloc[-lag:].mean()
        for window in [7, 14, 30, 60]:
            future_df[f'rolling_{window}_mean_{feature}'] = df[feature].iloc[-window:].mean()

    # Add rolling standard deviation features (if used during training)
    for feature in ['high', 'low']:
        for window in [7, 14, 30, 60]:
            future_df[f'rolling_{window}_std_{feature}'] = df[feature].rolling(window).std().iloc[-1]

    # Add Exponential Moving Average (EMA) features
    for span in [7, 14, 30, 60]:
        future_df[f'ema_{span}_high'] = df['high'].ewm(span=span, adjust=False).mean().iloc[-1]
        future_df[f'ema_{span}_low'] = df['low'].ewm(span=span, adjust=False).mean().iloc[-1]

    # Add trend features
    future_df['trend_high'] = future_df['lag_7_high'] - future_df['lag_7_high']
    future_df['trend_low'] = future_df['lag_7_low'] - future_df['lag_7_low']

    # Add missing features from training
    future_df['close'] = df['close'].iloc[-1]  # Adjust based on last known value
    future_df['vwap'] = df['vwap'].iloc[-1]    # Ensure correct population
    future_df['trade_count'] = df['trade_count'].iloc[-1]  # Adjust if necessary

    # Handle missing 'open' and 'volume' columns
    if 'open' not in df.columns:
        future_df['open'] = df['close'].iloc[-1]  # Set 'open' to the last known 'close'
    else:
        future_df['open'] = df['open'].iloc[-1]

    if 'volume' not in df.columns:
        future_df['volume'] = df['volume'].iloc[-1]  # Set 'volume' to the last known 'volume'
    else:
        future_df['volume'] = df['volume'].iloc[-1]

    # Add datetime-based features (hour, day, month)
    future_df['hour'] = input_date.hour
    future_df['day'] = input_date.day
    future_df['month'] = input_date.month

    # Add pct_change features for high and low
    future_df['pct_change_high'] = df['high'].pct_change().iloc[-1]  # Adjust with last known change
    future_df['pct_change_low'] = df['low'].pct_change().iloc[-1]  # Adjust with last known change

    # Add lag_7_rolling_7_mean features
    future_df['lag_7_rolling_7_mean_high'] = df['high'].iloc[-7:].mean()
    future_df['lag_7_rolling_7_mean_low'] = df['low'].iloc[-7:].mean()

    # Ensure the column order matches the training data
    expected_columns = ['close', 'trade_count', 'open', 'volume', 'vwap', 'lag_3_high', 'lag_3_low',
                        'lag_7_high', 'lag_7_low', 'lag_14_high', 'lag_14_low', 'lag_30_high', 'lag_30_low',
                        'lag_60_high', 'lag_60_low', 'rolling_7_mean_high', 'rolling_7_mean_low',
                        'rolling_7_std_high', 'rolling_7_std_low', 'rolling_14_mean_high', 'rolling_14_mean_low',
                        'rolling_14_std_high', 'rolling_14_std_low', 'rolling_30_mean_high', 'rolling_30_mean_low',
                        'rolling_30_std_high', 'rolling_30_std_low', 'rolling_60_mean_high', 'rolling_60_mean_low',
                        'rolling_60_std_high', 'rolling_60_std_low', 'ema_7_high', 'ema_7_low', 'ema_14_high',
                        'ema_14_low', 'ema_30_high', 'ema_30_low', 'ema_60_high', 'ema_60_low',
                        'pct_change_high', 'pct_change_low', 'trend_high', 'trend_low', 'hour', 'day', 'month',
                        'lag_7_rolling_7_mean_high', 'lag_7_rolling_7_mean_low']

    # Reorder the columns of future_df to match the training data
    future_df = future_df[expected_columns]

    # Predict high and low prices
    future_high = xgb_model_high.predict(future_df)
    future_low = xgb_model_low.predict(future_df)

    return pd.DataFrame({'Predicted High': future_high, 'Predicted Low': future_low}, index=[input_date])

# Example of predicting high and low prices for a given future date
predictions = predict_for_date('2025-03-21', df, xgb_model_high, xgb_model_low)
print(predictions)

def predict_for_date(input_date, df, xgb_model_high, xgb_model_low):
    """
    Predict high and low prices for a specific future date.

    Parameters:
    - input_date: The future date for which prediction is needed (format: 'YYYY-MM-DD').
    - df: Original DataFrame with historical data.
    - xgb_model_high: Trained model for high price prediction.
    - xgb_model_low: Trained model for low price prediction.

    Returns:
    - A DataFrame with predicted high and low prices for the given date.
    """
    # Ensure input_date is a pandas datetime object
    input_date = pd.to_datetime(input_date)

    # Prepare a single-row DataFrame for the input date
    future_df = pd.DataFrame(index=[input_date])

    # Populate lag and rolling features using the last available data
    for feature in ['high', 'low']:
        for lag in [3, 7, 14, 30, 60]:
            future_df[f'lag_{lag}_{feature}'] = df[feature].iloc[-lag:].mean()
        for window in [7, 14, 30, 60]:
            future_df[f'rolling_{window}_mean_{feature}'] = df[feature].iloc[-window:].mean()

    # Add rolling standard deviation features (if used during training)
    for feature in ['high', 'low']:
        for window in [7, 14, 30, 60]:
            future_df[f'rolling_{window}_std_{feature}'] = df[feature].rolling(window).std().iloc[-1]

    # Add Exponential Moving Average (EMA) features
    for span in [7, 14, 30, 60]:
        future_df[f'ema_{span}_high'] = df['high'].ewm(span=span, adjust=False).mean().iloc[-1]
        future_df[f'ema_{span}_low'] = df['low'].ewm(span=span, adjust=False).mean().iloc[-1]

    # Add trend features
    future_df['trend_high'] = future_df['lag_7_high'] - future_df['lag_7_high']
    future_df['trend_low'] = future_df['lag_7_low'] - future_df['lag_7_low']

    # Add missing features from training
    future_df['close'] = df['close'].iloc[-1]  # Adjust based on last known value
    future_df['vwap'] = df['vwap'].iloc[-1]    # Ensure correct population
    future_df['trade_count'] = df['trade_count'].iloc[-1]  # Adjust if necessary

    # Handle missing 'open' and 'volume' columns
    if 'open' not in df.columns:
        future_df['open'] = df['close'].iloc[-1]  # Set 'open' to the last known 'close'
    else:
        future_df['open'] = df['open'].iloc[-1]

    if 'volume' not in df.columns:
        future_df['volume'] = df['volume'].iloc[-1]  # Set 'volume' to the last known 'volume'
    else:
        future_df['volume'] = df['volume'].iloc[-1]

    # Add datetime-based features (hour, day, month)
    future_df['hour'] = input_date.hour
    future_df['day'] = input_date.day
    future_df['month'] = input_date.month

    # Add pct_change features for high and low
    future_df['pct_change_high'] = df['high'].pct_change().iloc[-1]  # Adjust with last known change
    future_df['pct_change_low'] = df['low'].pct_change().iloc[-1]  # Adjust with last known change

    # Add lag_7_rolling_7_mean features
    future_df['lag_7_rolling_7_mean_high'] = df['high'].iloc[-7:].mean()
    future_df['lag_7_rolling_7_mean_low'] = df['low'].iloc[-7:].mean()

    # Ensure the column order matches the training data
    expected_columns = ['close', 'trade_count', 'open', 'volume', 'vwap', 'lag_3_high', 'lag_3_low',
                        'lag_7_high', 'lag_7_low', 'lag_14_high', 'lag_14_low', 'lag_30_high', 'lag_30_low',
                        'lag_60_high', 'lag_60_low', 'rolling_7_mean_high', 'rolling_7_mean_low',
                        'rolling_7_std_high', 'rolling_7_std_low', 'rolling_14_mean_high', 'rolling_14_mean_low',
                        'rolling_14_std_high', 'rolling_14_std_low', 'rolling_30_mean_high', 'rolling_30_mean_low',
                        'rolling_30_std_high', 'rolling_30_std_low', 'rolling_60_mean_high', 'rolling_60_mean_low',
                        'rolling_60_std_high', 'rolling_60_std_low', 'ema_7_high', 'ema_7_low', 'ema_14_high',
                        'ema_14_low', 'ema_30_high', 'ema_30_low', 'ema_60_high', 'ema_60_low',
                        'pct_change_high', 'pct_change_low', 'trend_high', 'trend_low', 'hour', 'day', 'month',
                        'lag_7_rolling_7_mean_high', 'lag_7_rolling_7_mean_low']

    # Reorder the columns of future_df to match the training data
    future_df = future_df[expected_columns]

    # Predict high and low prices
    future_high = xgb_model_high.predict(future_df)
    future_low = xgb_model_low.predict(future_df)

    # Adjust to ensure the predicted high is the higher value and low is the lower
    if future_high[0] > future_low[0]:
        predicted_high = future_high[0]
        predicted_low = future_low[0]
    else:
        predicted_high = future_low[0]
        predicted_low = future_high[0]

    return pd.DataFrame({'Predicted High': [predicted_high], 'Predicted Low': [predicted_low]}, index=[input_date])

# Example of predicting high and low prices for a given future date
predictions = predict_for_date('2025-11-21', df, xgb_model_high, xgb_model_low)
print(predictions)

import joblib

# Save the models
joblib.dump(xgb_model_high, 'xgb_model_high.pkl')
joblib.dump(xgb_model_low, 'xgb_model_low.pkl')

print("Models saved successfully.")
