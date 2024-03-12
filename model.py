import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm_model(data, look_back=60, split_ratio=0.95):
  """
  Trains an LSTM model for stock price prediction.

  Args:
      data (pd.DataFrame): DataFrame containing 'Date' and 'Close' columns.
      look_back (int, optional): Sequence length for LSTM model. Defaults to 60.
      split_ratio (float, optional): Ratio of data to use for training. Defaults to 0.95.

  Returns:
      tuple: A tuple containing the trained model and the MinMaxScaler object.
  """

  data = data['Close']
  dataset = data.values

  # Calculate training data length
  training_data_len = int(np.ceil(len(dataset) * split_ratio))

  # Perform data preprocessing
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(data)

  # Separate training data
  train_data = scaled_data[0:training_data_len, :]

  # Create training sequences
  x_train, y_train = [], []
  for i in range(look_back, len(train_data)):
    x_train.append(train_data[i-look_back:i, 1])  # Select only close price (assuming 'Close' is at index 1)
    y_train.append(train_data[i, 1])

  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

  # Build the LSTM model
  model = Sequential()
  model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
  model.add(LSTM(64, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))

  # Compile the model
  model.compile(optimizer='adam', loss='mean_squared_error')

  # Train the model
  model.fit(x_train, y_train, batch_size=1, epochs=1)  # Adjust epochs as needed

  return model, scaler


def make_predictions(model, scaler, data, look_back=60, forecast_days=30):
  """
  Makes predictions for stock prices using the trained LSTM model.

  Args:
      model (tf.keras.Model): Trained LSTM model.
      scaler (MinMaxScaler): MinMaxScaler object used for normalization.
      look_back (int, optional): Sequence length for predictions. Defaults to 60.
      forecast_days (int, optional): Number of days to predict. Defaults to 30.

  Returns:
      list: A list of predicted closing prices.
  """

  # Get the last sequence from the scaled data
  test_dates = data.tail(look_back)['Date'].to_numpy()
  test_data = scaler.transform(data.tail(look_back)[:, 1:])
  # Reshape the data for prediction
  last_test_data = test_data.reshape((1, look_back, 1))

  # Create an empty list to store predictions
  predicted_values = []

  # Make predictions for the specified number of days
  for _ in range(forecast_days):
    prediction = model.predict(last_test_data)
    predicted_value = scaler.inverse_transform(prediction)[0][0]

    # Update last_test_data for subsequent predictions
    last_test_data = np.append(last_test_data[:, 1:], prediction, axis=1)
    last_test_data = last_test_data.reshape((1, look_back, 1))

    # Append the predicted value
    predicted_values.append(predicted_value)
    predicted_dates = np.concatenate((test_dates[1:], predicted_dates))
 

    return predicted_dates, predicted_values 


# Example usage (assuming your data is in a DataFrame called 'df')

