import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

# Step 1: Download stock data for multiple companies
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 
          'BABA', 'BIDU', 'BILI', 'BNTX', 'CSCO', 'DIS', 'F', 
          'GE', 'GM', 'GS', 'IBM', 'INTC', 'JNJ', 'JPM', 
          'KO', 'LLY', 'MCD', 'MRK', 'MS', 'NKE', 
          'PFE', 'PG', 'T', 'VZ', 'WMT', 'XOM']

# Download stock data
df = yf.download(stocks, start='2014-01-01', end='2024-01-01')

# View data
print(df.head())
# Print the DataFrame columns
print(df.columns)  # Check the structure of the DataFrame columns

# Save to CSV with full path
df.to_csv('stock_data.csv')

# Step 2: Data Preprocessing
# Fill missing values
df.ffill(inplace=True)  # Updated fillna method

# For this example, let's focus on one stock (e.g., AAPL) for prediction
data = df['Close']['AAPL'].copy().reset_index()
data.columns = ['Date', 'Close']

# Feature Engineering: Adding Technical Indicators
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

# Prepare data for modeling
data = data.dropna()  # Drop rows with NaN values

# Scale data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close', 'SMA_50', 'EMA_20']])

# Split data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create X_train and y_train for LSTM
X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i])  # Use all features
    y_train.append(train_data[i, 0])  # Predicting 'Close'

# Convert to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Step 3: Build and train LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))  # input_shape with 3 features
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32)

# Prepare test data for LSTM
X_test, y_test = [], scaled_data[train_size:, 0]
for i in range(60, len(scaled_data[train_size:])):
    X_test.append(scaled_data[train_size + i - 60:train_size + i])  # Include all features in test data
X_test = np.array(X_test)

# Reshape X_test for LSTM
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 3))  # Ensure to use 3 features

# Make predictions with LSTM
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = scaler.inverse_transform(np.concatenate((lstm_predictions, np.zeros((lstm_predictions.shape[0], 2))), axis=1))[:, 0]  # Inverse transform for 'Close' only

# Check lengths
print(f"Length of y_test: {len(y_test)}")
print(f"Length of lstm_predictions: {len(lstm_predictions)}")

# Adjust y_test to match the length of predictions
y_test = y_test[60:]  # Adjust y_test to match the length of predictions

# Evaluate LSTM model
lstm_rmse = mean_squared_error(y_test, lstm_predictions, squared=False)
print(f"LSTM RMSE: {lstm_rmse}")

# Step 4: Train-test split for Random Forest
# Prepare features and target variable, ensuring no Date column is included
X = data[['SMA_50', 'EMA_20']]  # Only include the features (SMA, EMA)
y = data['Close']  # Target variable (Close price)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train_rf, y_train_rf)

# Make predictions with Random Forest
rf_predictions = rf_model.predict(X_test_rf)

# Evaluate Random Forest model
rf_rmse = mean_squared_error(y_test_rf, rf_predictions, squared=False)
print(f"Random Forest RMSE: {rf_rmse}")

# Step 5: ARIMA Model
arima_model = auto_arima(data['Close'], seasonal=False, stepwise=True)
arima_model.fit(data['Close'])
arima_predictions = arima_model.predict(n_periods=len(y_test_rf))

# Evaluate ARIMA model
arima_rmse = mean_squared_error(y_test_rf, arima_predictions, squared=False)
print(f"ARIMA RMSE: {arima_rmse}")

# Step 6: Plot predictions
plt.figure(figsize=(14, 7))
plt.plot(data['Date'][train_size:], y_test_rf, label='Actual Prices', color='blue')
plt.plot(data['Date'][train_size:], rf_predictions, label='Random Forest Predictions', color='orange')

# Use the correct index for LSTM predictions
lstm_dates = data['Date'].values[train_size + 60:train_size + 60 + len(lstm_predictions)]
plt.plot(lstm_dates, lstm_predictions, label='LSTM Predictions', color='green')
plt.plot(data['Date'][train_size:], arima_predictions, label='ARIMA Predictions', color='red')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
