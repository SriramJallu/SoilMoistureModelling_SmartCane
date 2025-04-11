import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)

print("Coolio!!")

df = pd.read_csv("../../Data/Chemba_loc1_OpenMeteo_API_01012019_30122024_Daily.csv", skiprows=3)
df["time"] = pd.to_datetime(df["time"])
# print(df.columns)

features = df.drop(columns=["soil_moisture_7_to_28cm_mean (m³/m³)", "time", "rain_sum (mm)"])
target = df["soil_moisture_7_to_28cm_mean (m³/m³)"]

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


x, y = create_seq(features_scaled, target_scaled, time_steps=30, forecasts=3)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=123)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
model.add(tf.keras.layers.LSTM(64, return_sequences=True))
model.add(tf.keras.layers.LSTM(32))
model.add(tf.keras.layers.Dense(3))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(x, y, epochs=20, batch_size=32, validation_split=0.2)

last_30_days = features_scaled[-30:]
last_30_days = last_30_days.reshape((1, 30, features_scaled.shape[1]))

predicted_sm = model.predict(last_30_days)
predicted_sm = target_scaler.inverse_transform(predicted_sm)

print("Next 3 days SM", predicted_sm)

test_df = pd.read_csv("../../Data/Chemba_loc1_OpenMeteo_API_01012017_30122018_Daily.csv", skiprows=3)
test_df["time"] = pd.to_datetime(test_df["time"])


test_features = test_df.drop(columns=["soil_moisture_7_to_28cm_mean (m³/m³)", "time", "rain_sum (mm)"])
test_target = test_df["soil_moisture_7_to_28cm_mean (m³/m³)"]

test_features_scaled = features_scaler.fit_transform(test_features)
test_target_scaled = target_scaler.fit_transform(test_target.values.reshape(-1, 1))

x_test, y_test = create_seq(test_features_scaled, test_target_scaled, time_steps=30, forecasts=3)

predicted_scaled = model.predict(x_test)

predicted = target_scaler.inverse_transform(predicted_scaled)

y_test_original = target_scaler.inverse_transform(y_test)

for i in range(3):
    r2_day = r2_score(y_test_original[:, i], predicted[:, i])
    rmse_day = np.sqrt(mean_squared_error(y_test_original[:, i], predicted[:, i]))
    print(f"Day {i+1} - R²: {r2_day:.4f}, RMSE: {rmse_day:.4f} m³/m³")

# for day in range(5):
#     plt.figure(figsize=(12, 8))
#
#     for i in range(len(y_test_original)):
#         true_value = y_test_original[i: day]
#
#         predicted_value = predicted[i: day]
#
#         forecast_days = test_df["time"].values[-len(y_test_original) + i: -len(y_test_original) + i + 1]
#
#         plt.plot(forecast_days, true_value, 'bo-', label=f'True Day {day + 1}' if i == 0 else "", alpha=0.6)
#         plt.plot(forecast_days, predicted_value, 'rx-', label=f'Predicted Day {day + 1}' if i == 0 else "", alpha=0.6)
#
#     plt.title(f"True vs Predicted Soil Moisture - Day {day + 1}")
#     plt.xlabel("Date")
#     plt.ylabel("Soil Moisture (m³/m³)")
#     plt.legend(loc="upper left")
#     plt.xticks(rotation=45)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


# y_pred = model.predict(x_test)
#
# for i in range(5):
#     r2 = r2_score(y_test[:, i], y_pred[:, i])
#     rmse = mean_squared_error(y_test[:, i], y_pred[:, i], squared=False)
#     print(f"Day {i+1} - R²: {r2:.3f}, RMSE: {rmse:.3f}")

# print("X test values")
# print(x_test)
#
# print("\n Y test values")
# print(y_test)

