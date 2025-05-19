import rasterio
from rasterio.transform import rowcol
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

smap_sm_train = "../../Data/GSSM/GSSM_2016_2020_SM_NL_Daily_1km.tif"
sm_test_path = "../../Data/dataverse_files/1_station_measurements/2_calibrated/ITCSM_05_cd.csv"


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
                pixel_indices.append((i, j))
    return np.array(sequences_X), np.array(sequences_y), pixel_indices


def create_seq(y, time_steps=30, forecasts=7):
    ys = []
    y = np.array(y)
    for i in range(len(y) - time_steps - forecasts + 1):
        ys.append(y[i+time_steps:i+time_steps+forecasts].flatten())
    return np.array(ys)


train_data = read_tif(smap_sm_train)[60:1461]
test_data = read_tif(smap_sm_train)[1461:1461+366]

past_days = 30
forecast_days = 7

X_train, y_train, pixel_indices_train = create_pixelwise_sequences(train_data, past_days, forecast_days)
X_test, y_test, pixel_indices_test = create_pixelwise_sequences(test_data, past_days, forecast_days)

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(X_test.shape)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, activation='relu', input_shape=(past_days, 1)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(forecast_days)
])
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2)


y_pred = model.predict(X_test)

rmse_scores = []
r2_scores = []


headers = pd.read_csv(sm_test_path, skiprows=18, nrows=0).columns.tolist()
sm_test = pd.read_csv(sm_test_path, skiprows=20, parse_dates=["Date time"], names=headers)

sm_test["Date time"] = pd.to_datetime(sm_test["Date time"], format='%d-%m-%Y %H:%M', errors='coerce')
sm_test = sm_test[sm_test["Date time"] >= '2020-01-01']

sm_test = sm_test.set_index("Date time")
sm_test = sm_test.resample("D").mean()

print(sm_test.head())
print(sm_test.shape)

sm_test_insitu = create_seq(sm_test[" 5 cm SM"], 30 , 7)

with rasterio.open(smap_sm_train) as src:
    transform = src.transform
    crs = src.crs

transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

lon, lat = 6.69944, 52.27333

x, y = transformer.transform(lon, lat)

row, col = rowcol(transform, x, y)
print(row, col)

# lat, lon
# 52.27333, 6.69944

forecast_day = 0
pixel_index = 0

target_pixel = (row, col)

matching_indices = [idx for idx, pix in enumerate(pixel_indices_test) if pix == target_pixel]

true_series = sm_test_insitu[:, forecast_day]
pred_series = y_pred[matching_indices, forecast_day]


for day in range(forecast_days):
    true_series = sm_test_insitu[:, day]
    pred_series = y_pred[matching_indices, day]
    rmse = np.sqrt(mean_squared_error(true_series, pred_series))
    r2 = r2_score(true_series, pred_series)
    print(f"Pixel {target_pixel}, Day {day + 1} — RMSE: {rmse:.4f}, R²: {r2:.4f}")

fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex="all")

for i in range(3):
    axs[i].plot(sm_test_insitu[:, i], label="Actual", color="black")
    axs[i].plot(y_pred[matching_indices, i], label="Predicted (Mean)", color="green")
    axs[i].set_title(f"SM Prediction - Day {i+1}")
    axs[i].set_ylabel("SM")
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Time step")
plt.tight_layout()
plt.show()


