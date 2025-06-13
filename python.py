!pip install kagglehub --quiet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mrsimple07/stock-price-prediction")

print("Path to dataset files:", path)

# Load dataset
df = pd.read_csv(f"{path}/stock_data.csv")
df["Unnamed: 0"] = pd.to_datetime(df["Unnamed: 0"])
df.set_index("Unnamed: 0", inplace=True)

# Use one of the stock columns, for example 'Stock_1'
data = df[['Stock_1']]
dataset = data.values

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Training data
training_data_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[0:training_data_len, :]

# Create dataset
X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile and train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=5)

# Test data
test_data = scaled_data[training_data_len - 60:, :]
X_test, y_test = [], dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plotting
train = data[:training_data_len]
valid = data[training_data_len:].copy() # Use .copy() to avoid SettingWithCopyWarning
valid['Predictions'] = predictions

plt.figure(figsize=(16,6))
plt.title('Model - Stock Price Prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train['Stock_1'])
plt.plot(valid['Stock_1'], label='Actual') # Plot 'Stock_1' from valid
plt.plot(valid['Predictions'], label='Predicted') # Plot 'Predictions' from valid
plt.legend(['Train', 'Actual', 'Predicted'], loc='lower right')
plt.show()
