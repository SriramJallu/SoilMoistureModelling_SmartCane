import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


df = pd.read_csv("../../Data/Chemba_loc1_OpenMeteo_API_01012021_30122024_Daily.csv", skiprows=3)
df["time"] = pd.to_datetime(df["time"])

features = df.drop(columns=["soil_temperature_0_to_7cm_mean (°C)"])
target = df["soil_temperature_0_to_7cm_mean (°C)"]

features_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features_scaled = features_scaler.fit_transform(features)
target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))


def create_seq(x, y, time_steps=30, forecasts=3):
    xs, ys = [], []

    for i in range(len(x) - time_steps + forecasts):
        xs.append(x[i:i+time_steps])
        ys.append(y[i+time_steps:i+time_steps+forecasts].flatten())
    return np.array(xs), np.array(ys)


x, y = create_seq(features_scaled, target_scaled, time_steps=30, forecasts=3)

model = Sequential()
model.add(LSTM(64, input_shape=(x.shape[1], x.shape[2])))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(x, y, epochs=20, batch_size=32, validation_split=0.2)

last_30_days = features_scaled[-30:]
last_30_days = last_30_days.reshpae((1, 30, features_scaled.shape[1]))

predicted_sm = model.predict(last_30_days)
predicted_sm = target_scaler.inverse_transform(predicted_sm)

print("Next 3 days SM", predicted_sm)



