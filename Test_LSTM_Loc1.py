import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


print("Coolio!!")

df = pd.read_csv("../../Data/Chemba_loc2_OpenMeteo_API_01012021_30122024_Daily.csv", skiprows=3)
df["time"] = pd.to_datetime(df["time"])

features = df.drop(columns=["soil_temperature_0_to_7cm_mean (°C)", "time"])
target = df["soil_temperature_0_to_7cm_mean (°C)"]

features_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features_scaled = features_scaler.fit_transform(features)
target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))


def create_seq(x, y, time_steps=30, forecasts=3):
    xs, ys = [], []

    for i in range(len(x) - time_steps - forecasts + 1):
        xs.append(x[i:i+time_steps])
        ys.append(y[i+time_steps:i+time_steps+forecasts].flatten())
    return np.array(xs), np.array(ys)


x, y = create_seq(features_scaled, target_scaled, time_steps=30, forecasts=5)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

model = Sequential()
model.add(LSTM(64, input_shape=(x.shape[1], x.shape[2])))
model.add(Dense(5))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

last_30_days = features_scaled[-30:]
last_30_days = last_30_days.reshape((1, 30, features_scaled.shape[1]))

predicted_sm = model.predict(last_30_days)
predicted_sm = target_scaler.inverse_transform(predicted_sm)

print("Next 3 days SM", predicted_sm)


y_pred = model.predict(x_test)

for i in range(5):
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    rmse = mean_squared_error(y_test[:, i], y_pred[:, i], squared=False)
    print(f"Day {i+1} - R²: {r2:.3f}, RMSE: {rmse:.3f}")

# print("X test values")
# print(x_test)
#
# print("\n Y test values")
# print(y_test)

