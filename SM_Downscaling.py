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
from skimage.transform import resize
from sklearn.ensemble import RandomForestRegressor
from rasterio.warp import reproject, Resampling

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)

gssm_path = "../../Data/GSSM/GSSM_2016_2020_SM_Daily_1km.tif"
smap_sm_path = "../../Data/SMAP/SMAP_2016_2022_SM_Daily.tif"
era5_paths = [
    "../../Data/ERA5/ERA5_2015_2022_Precip_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_Temp_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_RH_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_WindSpeed_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_Radiation_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_ET_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_SoilTemp_Daily.tif"
]


def read_tif(tif):
    with rasterio.open(tif) as src:
        data = src.read().astype(np.float32)
        meta = src.meta
    return data, meta


def stack_weather(tif):
    feature_array = [read_tif(i) for i in tif]
    return np.stack(feature_array, axis=-1)


def generate_dates(start_year, end_year):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    return [(start + timedelta(days=i)).strftime('%Y%m%d') for i in range((end - start).days + 1)]


def resample_to_match(source_array, source_meta, target_meta):
    T = source_array.shape[0]
    H_target, W_target = target_meta['height'], target_meta['width']
    resampled = np.empty((T, H_target, W_target), dtype=np.float32)
    for t in range(T):
        reproject(
            source=source_array[t],
            destination=resampled[t],
            src_transform=source_meta['transform'],
            src_crs=source_meta['crs'],
            dst_transform=target_meta['transform'],
            dst_crs=target_meta['crs'],
            resampling=Resampling.bilinear
        )
    return resampled


gssm_data, gssm_meta = read_tif(gssm_path)
smap_data, smap_meta = read_tif(smap_sm_path)
smap_data = smap_data[:gssm_data.shape[0]]
smap_1km = resample_to_match(smap_data, smap_meta, gssm_meta)

era5_bands = []
for path in era5_paths:
    era_data, era_meta = read_tif(path)
    era_data = era_data[365:]
    era_resampled = resample_to_match(era_data, era_meta, gssm_meta)
    era5_bands.append(era_resampled)

era5_1km = np.stack(era5_bands, axis=-1)
era5_1km = era5_1km[:gssm_data.shape[0]]

era5_dates = generate_dates(2016, 2022)
gssm_dates = generate_dates(2016, 2020)
smap_dates = generate_dates(2016, 2022)

T, H, W, F = era5_1km.shape
smap_res = smap_1km[:T]
X_raw = np.concatenate([era5_1km, smap_res[..., np.newaxis]], axis=-1)

gssm = gssm_data[:T]
gssm_missing = {'20160219','20160220','20160221','20160222','20160223','20160224','20160225','20160226','20160227'}
dates = era5_dates[:T]

train_idx = [i for i, d in enumerate(dates) if d[:4] in ['2016','2017','2018','2019'] and d not in gssm_missing]
test_idx = [i for i, d in enumerate(dates) if d[:4] == '2020' and d not in gssm_missing]

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = X_raw[train_idx].reshape(-1, F+1)
y_train = gssm[train_idx].reshape(-1, 1)
mask = ~np.isnan(y_train[:, 0]) & ~np.isnan(X_train).any(axis=1)
X_train, y_train = X_train[mask], y_train[mask]

X_train = scaler_x.fit_transform(X_train)
y_train = scaler_y.fit_transform(y_train)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)


X_test = X_raw[test_idx].reshape(-1, F+1)
y_test = gssm[test_idx].reshape(-1, 1)
mask = ~np.isnan(y_test[:, 0]) & ~np.isnan(X_test).any(axis=1)
X_test, y_test = X_test[mask], y_test[mask]

X_test = scaler_x.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

pred_scaled = model.predict(X_test)
pred = scaler_y.inverse_transform(pred_scaled)


rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²:   {r2:.4f}")

