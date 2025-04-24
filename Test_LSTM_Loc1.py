import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
import matplotlib.pyplot as plt
import statsmodels

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)

print("Coolio!!")

df = pd.read_csv("../../Data/Chemba_loc1_OpenMeteo_API_01012019_30122024_Daily.csv", skiprows=3)
df["time"] = pd.to_datetime(df["time"])
# print(df.head())

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


x, y = create_seq(features_scaled, target_scaled, time_steps=20, forecasts=7)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=123)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='same', input_shape=(x.shape[1], x.shape[2])))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.LSTM(32))
model.add(tf.keras.layers.Dense(7))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(x, y, epochs=20, batch_size=32, validation_split=0.3)

last_30_days = features_scaled[-20:]
last_30_days = last_30_days.reshape((1, 20, features_scaled.shape[1]))

predicted_sm = model.predict(last_30_days)
predicted_sm = target_scaler.inverse_transform(predicted_sm)

print("Next 3 days SM", predicted_sm)

test_df = pd.read_csv("../../Data/Chemba_loc1_OpenMeteo_API_01012017_30122018_Daily.csv", skiprows=3)
test_df["time"] = pd.to_datetime(test_df["time"])

# test_df = test_df[test_df["time"].dt.year == 2017]

test_features = test_df.drop(columns=["soil_moisture_7_to_28cm_mean (m³/m³)", "time", "rain_sum (mm)"])
test_target = test_df["soil_moisture_7_to_28cm_mean (m³/m³)"]

test_features_scaled = features_scaler.fit_transform(test_features)
test_target_scaled = target_scaler.fit_transform(test_target.values.reshape(-1, 1))

x_test, y_test = create_seq(test_features_scaled, test_target_scaled, time_steps=20, forecasts=7)

predicted_scaled = model.predict(x_test)

predicted = target_scaler.inverse_transform(predicted_scaled)

y_test_original = target_scaler.inverse_transform(y_test)

for i in range(7):
    r2_day = r2_score(y_test_original[:, i], predicted[:, i])
    rmse_day = np.sqrt(mean_squared_error(y_test_original[:, i], predicted[:, i]))
    print(f"Day {i+1} - R²: {r2_day:.4f}, RMSE: {rmse_day:.4f} m³/m³")


preds_day1, preds_day2, preds_day3 = predicted[:, 0], predicted[:, 1], predicted[:, 2]

true_day1, true_day2, true_day3 = y_test_original[:, 0], y_test_original[:, 1], y_test_original[:, 2]

fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex="all")

# print(true_day1[:5])
# print(true_day2[:5])

days = [1, 2, 3]

true_vals = [true_day1, true_day2, true_day3]
pred_vals = [preds_day1, preds_day2, preds_day3]

for i in range(3):
    axs[i].plot(true_vals[i], label='Actual', color='black')
    axs[i].plot(pred_vals[i], label='Predicted', color='green')
    axs[i].set_title(f'Soil Moisture Prediction - Day {days[i]}')
    axs[i].set_ylabel('Soil Moisture (m³/m³)')
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel('Time')

plt.tight_layout()
plt.show()

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

