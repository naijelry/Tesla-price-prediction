
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Load Data (Replace with your dataset file path)
df = pd.read_csv("/content/tesla_stock_data_final_cleaneddata(noduplciates_nomissingvalues).csv")

# Define trend: 1 for up (close > open), 0 for down (close <= open)
df['trend'] = (df['close'] > df['open']).astype(int)

# Convert `timestamp` to datetime and extract the date only
df['date'] = pd.to_datetime(df['timestamp']).dt.date

# Feature Engineering for Long-Term Trends
def create_long_term_features(data):
    # Moving Averages
    data['SMA_30'] = data['close'].rolling(window=30).mean()
    data['SMA_90'] = data['close'].rolling(window=90).mean()
    data['SMA_200'] = data['close'].rolling(window=200).mean()

    data['EMA_30'] = data['close'].ewm(span=30, adjust=False).mean()
    data['EMA_90'] = data['close'].ewm(span=90, adjust=False).mean()
    data['EMA_200'] = data['close'].ewm(span=200, adjust=False).mean()

    # RSI (Relative Strength Index)
    data['RSI_14'] = compute_rsi(data['close'], 14)

    # ATR (Average True Range) for volatility
    data['ATR_30'] = compute_atr(data, 30)
    data['ATR_60'] = compute_atr(data, 60)

    # Lag features
    data['Lag_30'] = data['close'].shift(30)
    data['Lag_60'] = data['close'].shift(60)

    # Drop rows with NaN values created by shifting or rolling
    return data.dropna()

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def compute_atr(data, window):
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift()).abs()
    low_close = (data['low'] - data['close'].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()

    return atr

# Apply feature engineering
df = create_long_term_features(df)

# Define features and target variable
X = df.drop(columns=['timestamp', 'date', 'trend'])  # Drop non-predictive columns
y = df['trend']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with class weights (balanced)
logistic_regression = LogisticRegression(max_iter=5000, class_weight='balanced')

# XGBoost with class weights (adjusted for imbalance)
xgboost = XGBClassifier(eval_metric='logloss', scale_pos_weight=(len(y) - sum(y)) / sum(y))

# Define stacking model
stacking_model = StackingClassifier(
    estimators=[('logistic', logistic_regression), ('xgb', xgboost)],
    final_estimator=LogisticRegression()
)

# Train the stacking model
stacking_model.fit(X_train_scaled, y_train)

# Get predicted probabilities for class 1 (Bullish)
y_pred_proba = stacking_model.predict_proba(X_test_scaled)[:, 1]  # probabilities for class 1

# Adjust threshold for class 1 (Bullish) prediction
threshold = 0.2  # Lower threshold to favor Bullish predictions more
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)  # Predict 1 if probability >= threshold, otherwise 0

# Evaluate adjusted predictions
print("Adjusted Threshold Test Accuracy:", accuracy_score(y_test, y_pred_adjusted))
print("Classification Report (Adjusted Threshold):\n", classification_report(y_test, y_pred_adjusted))

# Evaluate on training data
y_train_pred = stacking_model.predict(X_train_scaled)
print("Train Accuracy (Stacking Model): ", accuracy_score(y_train, y_train_pred))

"""after doing imbalancing"""

# Evaluate the stacking model performance
y_pred = stacking_model.predict(X_test_scaled)
print("Classification Report (Stacking Model):\n", classification_report(y_test, y_pred))
print("Stacking Model Test Accuracy:", accuracy_score(y_test, y_pred))

# Fit the base models individually before making predictions
logistic_regression.fit(X_train_scaled, y_train)
xgboost.fit(X_train_scaled, y_train)

# Evaluate on training data for the stacking model
y_train_pred = stacking_model.predict(X_train_scaled)
print("Train Accuracy (Stacking Model): ", accuracy_score(y_train, y_train_pred))

# Accuracy for Logistic Regression Model
logistic_train_pred = logistic_regression.predict(X_train_scaled)
logistic_test_pred = logistic_regression.predict(X_test_scaled)
print("Logistic Regression Train Accuracy: ", accuracy_score(y_train, logistic_train_pred))
print("Logistic Regression Test Accuracy: ", accuracy_score(y_test, logistic_test_pred))

# Accuracy for XGBoost Model
xgboost_train_pred = xgboost.predict(X_train_scaled)
xgboost_test_pred = xgboost.predict(X_test_scaled)
print("XGBoost Train Accuracy: ", accuracy_score(y_train, xgboost_train_pred))
print("XGBoost Test Accuracy: ", accuracy_score(y_test, xgboost_test_pred))

# Evaluate the stacking model performance
y_pred = stacking_model.predict(X_test_scaled)
print("Classification Report (Stacking Model):\n", classification_report(y_test, y_pred))
print("Stacking Model Test Accuracy:", accuracy_score(y_test, y_pred))

"""before imbalance"""

# Fit the base models individually before making predictions
logistic_regression.fit(X_train_scaled, y_train)
xgboost.fit(X_train_scaled, y_train)

# Evaluate on training data
y_train_pred = stacking_model.predict(X_train_scaled)
print("Train Accuracy (Stacking Model): ", accuracy_score(y_train, y_train_pred))

# Accuracy for Logistic Regression Model
logistic_train_pred = logistic_regression.predict(X_train_scaled)
logistic_test_pred = logistic_regression.predict(X_test_scaled)
print("Logistic Regression Train Accuracy: ", accuracy_score(y_train, logistic_train_pred))
print("Logistic Regression Test Accuracy: ", accuracy_score(y_test, logistic_test_pred))

# Accuracy for XGBoost Model
xgboost_train_pred = xgboost.predict(X_train_scaled)
xgboost_test_pred = xgboost.predict(X_test_scaled)
print("XGBoost Train Accuracy: ", accuracy_score(y_train, xgboost_train_pred))
print("XGBoost Test Accuracy: ", accuracy_score(y_test, xgboost_test_pred))

# Evaluate the stacking model performance
y_pred = stacking_model.predict(X_test_scaled)
print("Classification Report (Stacking Model):\n", classification_report(y_test, y_pred))
print("Stacking Model Test Accuracy:", accuracy_score(y_test, y_pred))

print(df['trend'].value_counts())
