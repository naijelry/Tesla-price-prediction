from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained models
xgb_model_high = joblib.load('xgb_model_high.pkl')
xgb_model_low = joblib.load('xgb_model_low.pkl')
scaler = joblib.load('scaler.pkl')
stacking_model = joblib.load('stacking_model.pkl')

# Load your DataFrame (adjust the file path accordingly)
df = pd.read_csv("tesla_stock_data_final_cleaneddata(noduplciates_nomissingvalues).csv")

# Columns used for training the trend prediction model (ensure they match the training dataset)
training_columns = ['close', 'high', 'low', 'trade_count', 'open', 'volume', 'vwap',
                    'SMA_30', 'SMA_90', 'SMA_200', 'EMA_30', 'EMA_90', 'EMA_200', 'RSI_14',
                    'ATR_30', 'ATR_60', 'Lag_30', 'Lag_60']

# Global variable to track consecutive bearish predictions
consecutive_bearish = 0

# Function to preprocess the input date for prediction (for trend prediction)
def preprocess_date_input(input_date, df, training_columns):
    input_date = pd.to_datetime(input_date)

    # Prepare the DataFrame for the given date (features must match the training set)
    date_features = pd.DataFrame(index=[input_date])

    # Add basic date-related features
    date_features['day_of_week'] = input_date.weekday  # Weekday as a feature
    date_features['day_of_month'] = input_date.day
    date_features['month'] = input_date.month
    date_features['year'] = input_date.year

    # Include price features like 'high', 'low', 'close' if they were used in training
    date_features['high'] = df['high'].iloc[-1]  # Last available high
    date_features['low'] = df['low'].iloc[-1]  # Last available low
    date_features['close'] = df['close'].iloc[-1]  # Last available close
    date_features['open'] = df['open'].iloc[-1]  # Last available open

    # Add percentage change (assuming this was used in training)
    date_features['pct_change'] = df['close'].pct_change().iloc[-1]  # Percentage change from previous close

    # Add rolling and lag features (ensure to use the same lags and windows used in training)
    for feature in ['high', 'low', 'close', 'open']:
        for lag in [30, 60]:
            if len(df) >= lag:
                date_features[f'lag_{lag}_{feature}'] = df[feature].iloc[-lag:].mean()
            else:
                date_features[f'lag_{lag}_{feature}'] = df[feature].iloc[-1]  # Fallback to most recent value
        for window in [30, 60]:
            if len(df) >= window:
                date_features[f'rolling_{window}_mean_{feature}'] = df[feature].iloc[-window:].mean()
                date_features[f'rolling_{window}_std_{feature}'] = df[feature].iloc[-window:].std()
            else:
                date_features[f'rolling_{window}_mean_{feature}'] = df[feature].iloc[-1]  # Fallback to most recent value
                date_features[f'rolling_{window}_std_{feature}'] = df[feature].iloc[-1]  # Use most recent std as default

    # Ensure the columns match the training set
    missing_columns = [col for col in training_columns if col not in date_features.columns]
    for col in missing_columns:
        date_features[col] = 0  # Default value for missing columns

    # Reorder columns to match the training set
    date_features = date_features[training_columns]

    return date_features

# Function to predict high/low prices using XGBoost models
def predict_for_date(input_date, df, xgb_model_high, xgb_model_low, scaler, stacking_model, training_columns):
    global consecutive_bearish

    # Predict high and low prices using the XGBoost models
    # Prepare a single-row DataFrame for the input date (same steps as before)
    input_date = pd.to_datetime(input_date)
    future_df = pd.DataFrame(index=[input_date])

    for feature in ['high', 'low']:
        for lag in [3, 7, 14, 30, 60]:
            future_df[f'lag_{lag}_{feature}'] = df[feature].iloc[-lag:].mean()
        for window in [7, 14, 30, 60]:
            future_df[f'rolling_{window}_mean_{feature}'] = df[feature].iloc[-window:].mean()

    for feature in ['high', 'low']:
        for window in [7, 14, 30, 60]:
            future_df[f'rolling_{window}_std_{feature}'] = df[feature].rolling(window).std().iloc[-1]

    for span in [7, 14, 30, 60]:
        future_df[f'ema_{span}_high'] = df['high'].ewm(span=span, adjust=False).mean().iloc[-1]
        future_df[f'ema_{span}_low'] = df['low'].ewm(span=span, adjust=False).mean().iloc[-1]

    future_df['trend_high'] = future_df['lag_7_high'] - future_df['lag_7_high']
    future_df['trend_low'] = future_df['lag_7_low'] - future_df['lag_7_low']

    future_df['close'] = df['close'].iloc[-1]
    future_df['vwap'] = df['vwap'].iloc[-1]
    future_df['trade_count'] = df['trade_count'].iloc[-1]

    if 'open' not in df.columns:
        future_df['open'] = df['close'].iloc[-1]
    else:
        future_df['open'] = df['open'].iloc[-1]

    if 'volume' not in df.columns:
        future_df['volume'] = df['volume'].iloc[-1]
    else:
        future_df['volume'] = df['volume'].iloc[-1]

    future_df['hour'] = input_date.hour
    future_df['day'] = input_date.day
    future_df['month'] = input_date.month

    future_df['pct_change_high'] = df['high'].pct_change().iloc[-1]
    future_df['pct_change_low'] = df['low'].pct_change().iloc[-1]

    future_df['lag_7_rolling_7_mean_high'] = df['high'].iloc[-7:].mean()
    future_df['lag_7_rolling_7_mean_low'] = df['low'].iloc[-7:].mean()

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

    future_df = future_df[expected_columns]

    future_high = xgb_model_high.predict(future_df)
    future_low = xgb_model_low.predict(future_df)

    if future_high[0] > future_low[0]:
        predicted_high = future_high[0]
        predicted_low = future_low[0]
    else:
        predicted_high = future_low[0]
        predicted_low = future_high[0]

    # Now, predict the trend using the stacking model
    date_features = preprocess_date_input(input_date, df, training_columns)
    scaled_date_features = scaler.transform(date_features)
    y_pred_proba = stacking_model.predict_proba(scaled_date_features)[:, 1]
    threshold = 0.4
    y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

    # Adjust trend if consecutive bearish predictions occur
    if consecutive_bearish == 2:
        y_pred_adjusted = 1

    trend_label = 'Bullish' if y_pred_adjusted == 1 else 'Bearish'

    # Update consecutive bearish counter
    if trend_label == "Bearish":
        consecutive_bearish += 1
    else:
        consecutive_bearish = 0

    return predicted_high, predicted_low, trend_label

# Route for the home page (renders the HTML form)
@app.route('/')
def index():
    return render_template('index2.html')

# Route for making predictions (handles POST request from the form)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the date from the request
        data = request.get_json()
        input_date = data.get('date')

        if not input_date:
            return jsonify({'error': 'Date is required'}), 400

        # Validate date format
        try:
            datetime.strptime(input_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400

        # Make predictions using the input date
        predicted_high, predicted_low, trend_label = predict_for_date(input_date, df, xgb_model_high, xgb_model_low, scaler, stacking_model, training_columns)

        # Convert numpy.float32 to Python float (JSON serializable)
        predicted_high = float(predicted_high)
        predicted_low = float(predicted_low)

        # Return the predictions as a JSON response
        return jsonify({
            'predicted_high': predicted_high,
            'predicted_low': predicted_low,
            'predicted_trend': trend_label
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

