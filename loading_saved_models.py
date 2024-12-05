# -*- coding: utf-8 -*-
"""Loading_saved_models.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xkFgNLhUgDgR1wx7dpz4Ottttpl00wCh
"""

import pandas as pd
import numpy as np
import joblib

# Load the saved models
best_xgb_high = joblib.load('best_xgb_high_model.pkl')
best_xgb_low = joblib.load('best_xgb_low_model.pkl')

print("Models loaded successfully!")

def get_historical_data_for_date(input_date):
    """
    Retrieve historical data for the required date. This should return data that includes
    the previous days' prices as well as the last 30 days' closing, high, and low prices
    for rolling calculations.
    """
    # Load your stock data (this assumes you're loading it from a CSV file or database)
    # Replace this with the actual path to your stock data
    df = pd.read_csv("/content/tesla_stock_data_final_cleaneddata(noduplciates_nomissingvalues).csv", parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Ensure the timestamp column is in UTC (if it's not already)
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

    # Convert input_date (without time) to datetime with time set to midnight
    input_date = pd.to_datetime(input_date)  # Convert to datetime
    input_date = input_date.replace(hour=0, minute=0, second=0, microsecond=0)  # Set time to midnight

    # Convert input_date to UTC if it's naive
    if input_date.tz is None:
        input_date = input_date.tz_localize('UTC')

    # Filter the dataframe for the data up to the input_date
    df = df[df['timestamp'] <= input_date]

    # Sort by timestamp to ensure correct order
    df = df.sort_values('timestamp')

    # Get the last 30 rows for rolling features (or use a window of your choice)
    historical_data = df.tail(30)

    return historical_data

def calculate_features_for_date(input_date):
    """
    Calculate all the required features for the given input_date based on historical data.
    """
    # Get historical data up to the input_date
    historical_data = get_historical_data_for_date(input_date)

    # Compute rolling features (30-day window)
    close_rolling_mean_30 = historical_data['close'].mean()
    high_rolling_mean_30 = historical_data['high'].mean()
    low_rolling_mean_30 = historical_data['low'].mean()

    close_rolling_std_30 = historical_data['close'].std()
    high_rolling_std_30 = historical_data['high'].std()
    low_rolling_std_30 = historical_data['low'].std()

    # Calculate other features
    price_diff = historical_data['high'].iloc[-1] - historical_data['low'].iloc[-1]
    close_open_diff = historical_data['close'].iloc[-1] - historical_data['open'].iloc[-1]
    price_range = historical_data['high'].iloc[-1] - historical_data['low'].iloc[-1]

    close_50ma = historical_data['close'].tail(50).mean() if len(historical_data) >= 50 else np.nan
    close_200ma = historical_data['close'].tail(200).mean() if len(historical_data) >= 200 else np.nan

    # Lag features (previous day's data)
    high_lag_1 = historical_data['high'].iloc[-2] if len(historical_data) > 1 else np.nan
    low_lag_1 = historical_data['low'].iloc[-2] if len(historical_data) > 1 else np.nan
    close_lag_1 = historical_data['close'].iloc[-2] if len(historical_data) > 1 else np.nan

    # Create a dictionary with features
    features = {
        'year': input_date.year,
        'month': input_date.month,
        'day_of_month': input_date.day,
        'day_of_week': input_date.weekday(),
        'week_of_year': input_date.isocalendar().week,
        'quarter': input_date.quarter,
        'close_rolling_mean_30': close_rolling_mean_30,
        'high_rolling_mean_30': high_rolling_mean_30,
        'low_rolling_mean_30': low_rolling_mean_30,
        'close_rolling_std_30': close_rolling_std_30,
        'high_rolling_std_30': high_rolling_std_30,
        'low_rolling_std_30': low_rolling_std_30,
        'price_diff': price_diff,
        'close_open_diff': close_open_diff,
        'price_range': price_range,
        'close_50ma': close_50ma,
        'close_200ma': close_200ma,
        'high_lag_1': high_lag_1,
        'low_lag_1': low_lag_1,
        'close_lag_1': close_lag_1
    }

    return features

def predict_for_date(input_date, model_high, model_low):
    """
    Predict high and low prices for a specific date based on the trained models.
    """
    # Convert input date to datetime object and ensure it is in UTC
    input_date = pd.to_datetime(input_date)

    # Get the features for the input date
    features = calculate_features_for_date(input_date)

    # Convert features dictionary to a DataFrame
    input_features = pd.DataFrame([features])

    # Make predictions for high and low prices using the models
    predicted_high = model_high.predict(input_features)
    predicted_low = model_low.predict(input_features)

    # Adjust predictions if low price is greater than high price
    if predicted_low > predicted_high:
        # Swap predicted values if low is greater than high
        predicted_high, predicted_low = predicted_low, predicted_high

    # Inverse scale predictions (if necessary, depending on scaling)
    predicted_high = predicted_high[0]
    predicted_low = predicted_low[0]

    return predicted_high, predicted_low

# Example Usage: Predict prices for a specific date
input_date = '2024-12-22'  # Replace with the date you want to predict for
predicted_high, predicted_low = predict_for_date(input_date, best_xgb_high, best_xgb_low)

# Print the results
print(f"Predicted High Price for {input_date}: {predicted_high:.4f}")
print(f"Predicted Low Price for {input_date}: {predicted_low:.4f}")