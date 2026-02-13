import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


def generate_mega_numbers_deep_ai():

    file_path = os.path.join(
        os.getcwd(),
        "lottnums",
        "Lottery_Mega_Millions_true.csv"
    )

    data = pd.read_csv(file_path)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    sequence_length = 10
    X = []
    y = []

    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True,
                   input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.6))
    model.add(Dense(units=6))

    model.compile(optimizer='adam', loss='mse')

    # Reduced epochs for API responsiveness
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_sequence = scaled_data[-sequence_length:]
    predicted_scaled = model.predict(
        np.array([last_sequence]), verbose=0
    )

    predicted_numbers = scaler.inverse_transform(predicted_scaled)
    predicted_numbers = np.round(predicted_numbers[0]).astype(int)

    return [str(num) for num in predicted_numbers]
