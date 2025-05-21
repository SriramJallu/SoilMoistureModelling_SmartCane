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

smap_sm_am_train = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_AM_NL_Daily.tif"
smap_sm_pm_train = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_PM_NL_Daily.tif"
sm_test_path = "../../Data/dataverse_files/1_station_measurements/2_calibrated/ITCSM_05_cd.csv"
gssm_sm_train = "../../Data/GSSM/GSSM_2016_2020_SM_NL_Daily_1km.tif"


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
            pixel_series = pd.Series(pixel_series).interpolate(method='linear', limit_direction='both').to_numpy()
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
    seq_ids = []
    for i in range(len(y) - time_steps - forecasts + 1):
        ys.append(y[i+time_steps:i+time_steps+forecasts].flatten())
        seq_ids.append(i)
    return np.array(ys), np.array(seq_ids)


train_data = read_tif(smap_sm_am_train)[:1461-9]
test_data = read_tif(smap_sm_am_train)[1461-9:1461-9+366]

past_days = 30
forecast_days = 7

X_train, y_train, pixel_indices_train = create_pixelwise_sequences(train_data, past_days, forecast_days)
X_test, y_test, pixel_indices_test = create_pixelwise_sequences(test_data, past_days, forecast_days)

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(X_test.shape)

X_train_2d = X_train.reshape((-1, past_days, 1, 1))
X_test_2d = X_test.reshape((-1, past_days, 1, 1))


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 1), activation='relu', padding='same', input_shape=(past_days, 1, 1)),
    tf.keras.layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
    tf.keras.layers.Reshape((past_days, 64)),
    tf.keras.layers.LSTM(128, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(forecast_days)
])
model.compile(optimizer='adam', loss='mse')

model.fit(X_train_2d, y_train, epochs=1, batch_size=32, validation_split=0.2)

model_name = f"sm_conv_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
model.save(f"../../Models/{model_name}")

# model = tf.keras.models.load_model("../../Models/sm_conv_lstm_20250520_143107.h5")
y_pred = model.predict(X_test_2d)

rmse_scores = []
r2_scores = []


headers = pd.read_csv(sm_test_path, skiprows=18, nrows=0).columns.tolist()
sm_test = pd.read_csv(sm_test_path, skiprows=20, parse_dates=["Date time"], names=headers)

sm_test["Date time"] = pd.to_datetime(sm_test["Date time"], format='%d-%m-%Y %H:%M', errors='coerce')
sm_test = sm_test[sm_test["Date time"] >= '2020-01-01']

sm_test = sm_test.set_index("Date time")
sm_test = sm_test.resample("D").mean()
# sm_test = sm_test[sm_test.index.time == pd.to_datetime("06:00:00").time()]

sm_test_insitu, insitu_seq = create_seq(sm_test[" 5 cm SM"], 30 , 7)
print(sm_test_insitu.shape)

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
print("Matching sequences for target pixel:", len(matching_indices))

for day in range(forecast_days):
    true_series = sm_test_insitu[:, day]
    # true_series = y_test[matching_indices, day]
    pred_series = y_pred[matching_indices, day]
    rmse = np.sqrt(mean_squared_error(true_series, pred_series))
    r2 = r2_score(true_series, pred_series)
    print(f"Pixel {target_pixel}, Day {day + 1} — RMSE: {rmse:.4f}, R²: {r2:.4f}")

fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex="all")

for i in range(3):
    axs[i].plot(sm_test_insitu[:, i], label="Actual", color="black")
    # axs[i].plot(y_test[matching_indices, i], label="Actual", color="black")
    axs[i].plot(y_pred[matching_indices, i], label="Predicted (Mean)", color="green")
    axs[i].set_title(f"SM Prediction - Day {i+1}")
    axs[i].set_ylabel("SM")
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Time step")
plt.tight_layout()
plt.show()



# am_train_data = read_tif(smap_sm_am_train)[:1461]
# am_test_data = read_tif(smap_sm_am_train)[1461:1461+366]
#
# pm_train_data = read_tif(smap_sm_pm_train)[:1461]
# pm_test_data = read_tif(smap_sm_pm_train)[1461:1461+366]
#
# train_data = np.nanmean(np.stack([am_train_data, pm_train_data], axis=0), axis=0)
# test_data = np.nanmean(np.stack([am_test_data, pm_test_data], axis=0), axis=0)