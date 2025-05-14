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

era5_precip_train = "../../Data/ERA5/ERA5_2015_2022_Precip_Daily.tif"
era5_precip_test = "../../Data/ERA5/ERA5_2023_2024_Precip_Daily.tif"
era5_temp_train = "../../Data/ERA5/ERA5_2015_2022_Temp_Daily.tif"
era5_temp_test = "../../Data/ERA5/ERA5_2023_2024_Temp_Daily.tif"
era5_rh_train = "../../Data/ERA5/ERA5_2015_2022_RH_Daily.tif"
era5_rh_test = "../../Data/ERA5/ERA5_2023_2024_RH_Daily.tif"
era5_windspeed_train = "../../Data/ERA5/ERA5_2015_2022_WindSpeed_Daily.tif"
era5_windspeed_test = "../../Data/ERA5/ERA5_2023_2024_WindSpeed_Daily.tif"
era5_radiation_train = "../../Data/ERA5/ERA5_2015_2022_Radiation_Daily.tif"
era5_radiation_test = "../../Data/ERA5/ERA5_2023_2024_Radiation_Daily.tif"
era5_et_train = "../../Data/ERA5/ERA5_2015_2022_ET_Daily.tif"
era5_et_test = "../../Data/ERA5/ERA5_2023_2024_ET_Daily.tif"
era5_soiltemp_train = "../../Data/ERA5/ERA5_2015_2022_SoilTemp_Daily.tif"
era5_soiltemp_test = "../../Data/ERA5/ERA5_2023_2024_SoilTemp_Daily.tif"

smap_sm_train = "../../Data/SMAP/SMAP_2016_2022_SM_Daily.tif"
smap_sm_test = "../../Data/SMAP/SMAP_2023_2024_SM_Daily.tif"


def read_tif(tif):
    with rasterio.open(tif) as src:
        data = src.read().astype(np.float32)
    return data


def create_pixelwise_sequences(data, past_days, forecast_days=7):
    T, H, W = data.shape
    sequences_X, sequences_y = [], []
    pixel_indices = []

    for i in range(H):
        for j in range(W):
            pixel_series = data[:, i, j]
            if np.any(np.isnan(pixel_series)):
                continue
            for t in range(T - past_days - forecast_days + 1):
                X = pixel_series[t:t+past_days]
                y = pixel_series[t+past_days:t+past_days+forecast_days]
                sequences_X.append(X)
                sequences_y.append(y)
                pixel_indices.append((i, j))  # keep track
    return np.array(sequences_X), np.array(sequences_y), pixel_indices


train_data = read_tif(smap_sm_train)
test_data = read_tif(smap_sm_test)

past_days = 30
forecast_days = 7

X_train, y_train, pixel_indices_train = create_pixelwise_sequences(train_data, past_days, forecast_days)
X_test, y_test, pixel_indices_test = create_pixelwise_sequences(test_data, past_days, forecast_days)

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, activation='relu', input_shape=(past_days, 1)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(forecast_days)
])
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)


y_pred = model.predict(X_test)

rmse_scores = []
r2_scores = []

for day in range(forecast_days):
    rmse = np.sqrt(mean_squared_error(y_test[:, day], y_pred[:, day]))
    r2 = r2_score(y_test[:, day], y_pred[:, day])
    rmse_scores.append(rmse)
    r2_scores.append(r2)

for d in range(forecast_days):
    print(f"Day {d+1} — RMSE: {rmse_scores[d]:.4f}, R²: {r2_scores[d]:.4f}")


forecast_day = 0
pixel_index = 0

target_pixel = (1, 1)

matching_indices = [idx for idx, pix in enumerate(pixel_indices_test) if pix == target_pixel]

true_series = y_test[matching_indices, forecast_day]
pred_series = y_pred[matching_indices, forecast_day]

plt.figure(figsize=(14, 5))
plt.plot(true_series, label="True SM", color='green')
plt.plot(pred_series, label="Predicted SM", color='red', linestyle='--')
plt.title(f"Soil Moisture Forecast — Pixel {target_pixel}, Day {forecast_day + 1}")
plt.xlabel("Time Step (~daily)")
plt.ylabel("Soil Moisture")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# N = 1
# sample_indices = random.sample(range(X_test.shape[0]), N)
#
# forecast_days = 7
# days = list(range(1, forecast_days + 1))
#
# for idx in sample_indices:
#     input_sequence = X_test[idx].squeeze()  # shape: (30,)
#     true_future = y_test[idx]  # shape: (7,)
#     pred_future = y_pred[idx]  # shape: (7,)
#
#     plt.figure(figsize=(10, 4))
#     plt.plot(range(30), input_sequence, label='Past 30 Days (Input)', color='blue')
#     plt.plot(range(30, 37), true_future, label='True SM (Next 7 Days)', color='green')
#     plt.plot(range(30, 37), pred_future, label='Predicted SM', color='red', linestyle='--')
#     plt.title(f"Soil Moisture Forecast - Sample {idx}")
#     plt.xlabel("Time (Days)")
#     plt.ylabel("Soil Moisture")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
########################################################################################################################
# def read_stack(path):
#     with rasterio.open(path) as src:
#         return src.read().astype(np.float32)
#
#
# def stack_features(*arrays):
#     return np.stack(arrays, axis=-1)
#
#
# def create_sequences(X, Y, seq_len=20):
#     Xs, Ys = [], []
#     for i in range(len(X) - seq_len):
#         Xs.append(X[i:i+seq_len])
#         Ys.append(Y[i+seq_len])
#     return np.array(Xs), np.array(Ys)
#
#
# precip = read_stack(era5_precip_train)[365:]
# temp = read_stack(era5_temp_train)[365:]
# rh = read_stack(era5_rh_train)[365:]
# windspeed = read_stack(era5_windspeed_train)[365:]
# radiation = read_stack(era5_radiation_train)[365:]
# et = read_stack(era5_et_train)[365:]
# soiltemp = read_stack(era5_soiltemp_train)[365:]
#
# features_train = stack_features(precip, temp, rh, windspeed, radiation, et, soiltemp)
# print(features_train.shape)
#
# sm = read_stack(smap_sm_train)
# target_train = np.expand_dims(sm, axis=-1)
#
# print(target_train.shape)
#
#
# precip_t = read_stack(era5_precip_test)
# temp_t = read_stack(era5_temp_test)
# rh_t = read_stack(era5_rh_test)
# windspeed_t = read_stack(era5_windspeed_test)
# radiation_t = read_stack(era5_radiation_test)
# et_t = read_stack(era5_et_test)
# soiltemp_t = read_stack(era5_soiltemp_test)
#
# features_test = stack_features(precip_t, temp_t, rh_t, windspeed_t, radiation_t, et_t, soiltemp_t)
# print(features_test.shape)
#
#
# sm_t = read_stack(smap_sm_test)
# target_test = np.expand_dims(sm_t, axis=-1)
# print(target_test.shape)
#
#
# X_train, y_train = create_sequences(features_train, target_train, 30)
# X_test, y_test = create_sequences(features_test, target_test, 30)
#
# print(X_train.shape)
# print(y_train.shape)
#
# print(X_test.shape)
# print(y_test.shape)
#
# model = tf.keras.Sequential([
#     tf.keras.layers.ConvLSTM2D()
# ])

