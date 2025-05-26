import rasterio
from rasterio.transform import rowcol, xy
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


def map_10km_to_1km(r1km, c1km, sm_transform, weather_transform):
    x, y = xy(sm_transform, r1km, c1km)
    r10km, c10km = rowcol(weather_transform, x, y)
    return r10km, c10km


def create_sequences(sm_transform, weather_transform, weather_data, sm_data, past_days, forecast_days=7):
    T, H, W= sm_data.shape
    F = weather_data.shape[-1]
    sequences_X, sequences_y, pixel_indices = [], [], []

    for i in range(H):
        for j in range(W):
            r10km, c10km = map_10km_to_1km(i, j, sm_transform, weather_transform)
            if r10km < 0 or c10km < 0 or r10km >= weather_data.shape[1] or c10km >= weather_data.shape[2]:
                continue

            pixel_weather = weather_data[:, r10km, c10km, :]
            pixel_sm = sm_data[:, i, j]
            pixel_sm = pd.Series(pixel_sm).interpolate(method='linear', limit_direction='both').to_numpy()
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


def create_seq(y, time_steps=30, forecasts=7):
    ys = []
    y = np.array(y)
    for i in range(len(y) - time_steps - forecasts + 1):
        ys.append(y[i+time_steps:i+time_steps+forecasts].flatten())
    return np.array(ys)


era5_train_paths = [
    "../../Data/ERA5_NL/ERA5_2016_2022_Precip_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2016_2022_Temp_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2016_2022_RH_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2016_2022_WindSpeed_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2016_2022_Radiation_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2016_2022_ET_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2016_2022_SoilTemp_NL_Daily.tif"
]


smap_sm_am_train = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_AM_NL_Daily.tif"
smap_sm_pm_train = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_PM_NL_Daily.tif"
sm_test_path = "../../Data/dataverse_files/1_station_measurements/2_calibrated/ITCSM_05_cd.csv"
gssm_sm_train = "../../Data/GSSM/GSSM_2016_2020_SM_NL_Daily_1km.tif"

weather_train = stack_weather(era5_train_paths)[60:1461]
weather_test = stack_weather(era5_train_paths)[1461:1461+366]

sm_train = read_tif(gssm_sm_train)[51:1461-9]
sm_test = read_tif(gssm_sm_train)[1461-9:1461-9+366]

print(weather_train.shape)
print(sm_train.shape)

with rasterio.open(gssm_sm_train) as src_sm:
    sm_transform = src_sm.transform

with rasterio.open(era5_train_paths[0]) as src_weather:
    weather_transform = src_weather.transform


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

X_train, y_train, pixel_indices_train = create_sequences(sm_transform, weather_transform, weather_train_scaled, sm_train_scaled, past_days, forecast_days)
X_test, y_test, pixel_indices_test = create_sequences(sm_transform, weather_transform, weather_test_scaled, sm_test_scaled, past_days, forecast_days)

print(X_train.shape, y_train.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(past_days, X_train.shape[-1])),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.LSTM(128, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(forecast_days)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2)

model_name = f"sm_weather_conv_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
model.save(f"../../Models/{model_name}")

model = tf.keras.models.load_model("../../Models/sm_weather_conv_lstm_20250521_142831.h5")
y_preds = model.predict(X_test)


y_preds_inv = target_scaler.inverse_transform(y_preds.reshape(-1, 1)).reshape(y_preds.shape)
y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# print(np.unique(np.array(pixel_indices_test), axis=0))

headers = pd.read_csv(sm_test_path, skiprows=21, nrows=0).columns.tolist()
sm_test = pd.read_csv(sm_test_path, skiprows=23, parse_dates=["Date time"], names=headers)

sm_test["Date time"] = pd.to_datetime(sm_test["Date time"], format='%d-%m-%Y %H:%M', errors='coerce')
sm_test = sm_test[sm_test["Date time"] >= '2020-01-01']

# sm_test[" 5 cm SM"] = sm_test[" 5 cm SM"].apply(lambda x: np.nan if x < 0 else x)

# count_neg99 = (sm_test[" 5 cm SM"] == -99.999).sum()
# print(count_neg99)

sm_test = sm_test.set_index("Date time")
sm_test = sm_test.resample("D").mean()
# sm_test = sm_test[sm_test.index.time == pd.to_datetime("06:00:00").time()]
# print(sm_test.head())

sm_test_insitu = create_seq(sm_test[" 5 cm SM"], 30, 7)

with rasterio.open(smap_sm_am_train) as src:
    transform = src.transform
    crs = src.crs

transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

lon, lat = 6.69944, 52.27333

x, y = transformer.transform(lon, lat)

row, col = rowcol(transform, x, y)

# lat, lon
# 52.27333, 6.69944
target_pixel = (row, col)

matching_indices = [idx for idx, pix in enumerate(pixel_indices_test) if pix == target_pixel]

for day in range(forecast_days):
    true_series = sm_test_insitu[:, day]
    # true_series = y_test_inv[matching_indices, day]
    pred_series = y_preds_inv[matching_indices, day]
    rmse = np.sqrt(mean_squared_error(true_series, pred_series))
    r2 = r2_score(true_series, pred_series)
    print(f"Pixel {target_pixel}, Day {day + 1} — RMSE: {rmse:.4f}, R²: {r2:.4f}")

fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex="all")

for i in range(3):
    axs[i].plot(sm_test_insitu[:, i], label="Actual", color="black")
    # axs[i].plot(y_test_inv[matching_indices, i], label="Actual", color="black")
    axs[i].plot(y_preds_inv[matching_indices, i], label="Predicted (Mean)", color="green")
    axs[i].set_title(f"SM Prediction - Day {i+1}")
    axs[i].set_ylabel("SM")
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Time step")
plt.tight_layout()
plt.show()
