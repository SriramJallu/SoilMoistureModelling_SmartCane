import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from pyproj import Transformer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)


def read_tif(tif):
    with rasterio.open(tif) as src:
        data = src.read().astype(np.float32)
    return data


def stack_weather(tif):
    feature_array = [read_tif(i) for i in tif]
    return np.stack(feature_array, axis=-1)


def create_sequences(weather_data, sm_data, past_days, forecast_days=7):
    T, H, W, F = weather_data.shape
    sequences_X, sequences_y, pixel_indices = [], [], []

    for i in range(H):
        for j in range(W):
            pixel_weather = weather_data[:, i, j, :]
            pixel_sm = sm_data[:, i, j]
            if np.any(np.isnan(pixel_weather)) or np.any(np.isnan(pixel_sm)):
                continue
            for t in range(T- past_days - forecast_days + 1):
                weather_X = pixel_weather[t:t+past_days, :]
                sm_X = pixel_sm[t:t + past_days].reshape(-1, 1)
                X = np.hstack([weather_X, sm_X])
                y = pixel_sm[t+past_days:t+past_days+forecast_days]
                sequences_X.append(X)
                sequences_y.append(y)
                pixel_indices.append((i, j))
    return  np.array(sequences_X), np.array(sequences_y), pixel_indices

era5_train_paths = [
    "../../Data/ERA5/ERA5_2015_2022_Precip_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_Temp_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_RH_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_WindSpeed_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_Radiation_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_ET_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_SoilTemp_Daily.tif"
]

era5_test_paths = [
    "../../Data/ERA5/ERA5_2023_2024_Precip_Daily.tif",
    "../../Data/ERA5/ERA5_2023_2024_Temp_Daily.tif",
    "../../Data/ERA5/ERA5_2023_2024_RH_Daily.tif",
    "../../Data/ERA5/ERA5_2023_2024_WindSpeed_Daily.tif",
    "../../Data/ERA5/ERA5_2023_2024_Radiation_Daily.tif",
    "../../Data/ERA5/ERA5_2023_2024_ET_Daily.tif",
    "../../Data/ERA5/ERA5_2023_2024_SoilTemp_Daily.tif"
]

smap_sm_train = "../../Data/SMAP/SMAP_2016_2022_SM_Daily.tif"
smap_sm_test = "../../Data/SMAP/SMAP_2023_2024_SM_Daily.tif"

weather_train = stack_weather(era5_train_paths)[365:]
weather_test = stack_weather(era5_test_paths)

sm_train = read_tif(smap_sm_train)
sm_test = read_tif(smap_sm_test)

print(weather_train.shape)
print(sm_train.shape)


feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

T, H, W, F = weather_train.shape
weather_flat_train = weather_train.reshape(T*H*W, F)
weather_flat_scaled = feature_scaler.fit_transform(weather_flat_train)
weather_train_scaled = weather_flat_scaled.reshape(T, H, W, F)

T2, H2, W2, F2 = weather_test.shape
weather_flat_test = weather_test.reshape(T2*H2*W2, F2)
weather_flat_test_scaled = feature_scaler.transform(weather_flat_test)
weather_test_scaled = weather_flat_test_scaled.reshape(T2, H2, W2, F2)

sm_flat = sm_train.reshape(-1, 1)
sm_train_scaled = target_scaler.fit_transform(sm_flat).reshape(sm_train.shape)

sm_flat_test = sm_test.reshape(-1, 1)
sm_test_scaled = target_scaler.transform(sm_flat_test).reshape(sm_test.shape)


past_days = 30
forecast_days = 7

X_train, y_train, pixel_indices_train = create_sequences(weather_train_scaled, sm_train_scaled, past_days, forecast_days)
X_test, y_test, pixel_indices_test = create_sequences(weather_test_scaled, sm_test_scaled, past_days, forecast_days)

print(X_train.shape, y_train.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(past_days, X_train.shape[-1])),
    tf.keras.layers.LSTM(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(forecast_days)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

y_preds = model.predict(X_test)


y_preds_inv = target_scaler.inverse_transform(y_preds.reshape(-1, 1)).reshape(y_preds.shape)
y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)


rmse_scores, r2_scores = [], []

for day in range(forecast_days):
    rmse = np.sqrt(mean_squared_error(y_test_inv[:, day], y_preds_inv[:, day]))
    r2 = r2_score(y_test_inv[:, day], y_preds_inv[:, day])
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    print(f"Day {day + 1} — RMSE: {rmse:.4f}, R²: {r2:.4f}")

print(np.unique(np.array(pixel_indices_test), axis=0))
target_pixel = (1, 0)
forecast_day = 0

matching_idx = [idx for idx, pix in enumerate(pixel_indices_test) if pix == target_pixel]
for day in range(forecast_days):
    true_series = y_test_inv[matching_idx, day]
    pred_series = y_preds_inv[matching_idx, day]
    rmse = np.sqrt(mean_squared_error(true_series, pred_series))
    r2 = r2_score(true_series, pred_series)
    print(f"Pixel {target_pixel}, Day {day + 1} — RMSE: {rmse:.4f}, R²: {r2:.4f}")


fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex="all")

for i in range(3):
    axs[i].plot(y_test_inv[matching_idx, i], label="Actual", color="black")
    axs[i].plot(y_preds_inv[matching_idx, i], label="Predicted (Mean)", color="green")
    axs[i].set_title(f"SM Prediction - Day {i+1}")
    axs[i].set_ylabel("SM")
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Time step")
plt.tight_layout()
plt.show()

# true_series = y_test_inv[matching_idx, forecast_day]
# pred_series = y_preds_inv[matching_idx, forecast_day]
#
# plt.figure(figsize=(14, 5))
# plt.plot(true_series, label="True SM", color='green')
# plt.plot(pred_series, label="Predicted SM", color='red', linestyle='--')
# plt.title(f"Soil Moisture Forecast — Pixel {target_pixel}, Day {forecast_day + 1}")
# plt.xlabel("Time Step (daily)")
# plt.ylabel("Soil Moisture")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


