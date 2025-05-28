import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from pyproj import Transformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from skimage.transform import resize
from rasterio.enums import Resampling
from rasterio.warp import reproject


np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)


def read_tif(tif):
    with rasterio.open(tif) as src:
        data = src.read().astype(np.float32)
        meta = src.meta
    return data, meta


def fill_missing_data(data, max_window=28):
    T, W, H = data.shape
    data_reshaped = data.reshape(T, -1)
    filled = np.empty_like(data_reshaped)

    for idx in range(data_reshaped.shape[1]):
        series = pd.Series(data_reshaped[:, idx])
        series_filled = series.fillna(method="ffill", limit=max_window)
        filled[:, idx] = series_filled.to_numpy()

    return filled.reshape(T, W, H)


def resampling_data(src_data, src_meta, target_shape, target_transform, resample=Resampling.bilinear):
    T, H, W = src_data.shape
    resampled = np.zeros((T, target_shape[0], target_shape[1]), dtype=np.float32)

    for t in range(T):
        reproject(
            source=src_data[t],
            destination=resampled[t],
            src_transform=src_meta["transform"],
            src_crs=src_meta["crs"],
            dst_transform=target_transform,
            dst_crs=src_meta["crs"],
            resampling=resample
        )
    return resampled


def resample_static_data(static_data, static_meta, smap_shape, resample=Resampling.bilinear):
    data = static_data[0]
    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=static_data.dtype,
            crs=static_meta["crs"],
            transform=static_meta["transform"]
        ) as dataset:
            dataset.write(data, 1)

        with memfile.open() as dataset:
            resampled = dataset.read(
                out_shape=(1, smap_shape[0], smap_shape[1]),
                resampling=resample
            )[0]
    return resampled


def flatten_inputs(dynamic_vars, static_vars):
    T, H, W, D = dynamic_vars.shape
    S = static_vars.shape[-1]

    X_dynamic = dynamic_vars.reshape(T, H*W, D)
    X_static = static_vars.reshape(1, H*W, S).repeat(T, axis=0)

    X = np.concatenate([X_dynamic, X_static], axis=-1)
    return X.reshape(-1, D+S)


smap_sm_am_path = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_AM_NL_Daily.tif"
smap_sm_pm_path = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_PM_NL_Daily.tif"
gssm_sm_path = "../../Data/GSSM/GSSM_2017_2020_SM_NL_Daily_1km.tif"
ndvi_path = "../../Data/VIIRS/VIIRS_NDVI_2016_2021_1km.tif"
lst_day_path = "../../Data/VIIRS/VIIRS_LST_Day_2017_2021_1km.tif"
lst_night_path = "../../Data/VIIRS/VIIRS_LST_Night_2017_2021_1km.tif"
dem_path = "../../Data/StaticVars/DEM_Map_90m.tif"
slope_path = "../../Data/StaticVars/Slope_Map_90m.tif"
soil_texture_path = "../../Data/StaticVars/SoilTexture_Map_250m.tif"

sm_test_path = "../../Data/dataverse_files/1_station_measurements/2_calibrated/ITCSM_05_cd.csv"


smap_sm_am, smap_sm_am_meta = read_tif(smap_sm_am_path)
smap_sm_pm, smap_sm_pm_meta = read_tif(smap_sm_am_path)
gssm, gssm_meta = read_tif(gssm_sm_path)
ndvi, ndvi_meta = read_tif(ndvi_path)
lst_day, lst_day_meta = read_tif(lst_day_path)
lst_night, lst_night_meta = read_tif(lst_night_path)
dem, dem_meta = read_tif(dem_path)
slope, slope_meta = read_tif(slope_path)
soil_texture, soil_texture_meta = read_tif(soil_texture_path)

smap = np.nanmean(np.stack([smap_sm_am, smap_sm_pm]), axis=0)
smap_filled = fill_missing_data(smap)[366:366+1461]

target_shape = gssm.shape[1:]
target_transform = gssm_meta["transform"]
smap_resampled = resampling_data(smap_filled, smap_sm_pm_meta, target_shape, target_transform)

ndvi_filled = fill_missing_data(ndvi)[366:366+1461]
lst_day_filled = fill_missing_data(lst_day)[:1461]
lst_night_filled = fill_missing_data(lst_night)[:1461]


smap_resampled_train, smap_resampled_test = smap_resampled[:1096], smap_resampled[1096:]
ndvi_train, ndvi_test = ndvi_filled[:1096], ndvi_filled[1096:]
lst_day_train, lst_day_test = lst_day_filled[:1096], lst_day_filled[1096:]
lst_night_train, lst_night_test = lst_night_filled[:1096], lst_night_filled[1096:]
gssm_train, gssm_test = gssm[:1096], gssm[1096:]

print(smap_resampled_train.shape)
print(ndvi_train.shape)
print(lst_day_train.shape)
print(lst_night_train.shape)

dynamic_stack_train = np.stack([smap_resampled_train, ndvi_train, lst_day_train, lst_night_train], axis=-1)
dynamic_stack_test = np.stack([smap_resampled_test, ndvi_test, lst_day_test, lst_night_test], axis=-1)


dem_1km = resample_static_data(dem, dem_meta, target_shape)
slope_1km = resample_static_data(slope, slope_meta, target_shape)
soil_texture_1km = resample_static_data(soil_texture, soil_texture_meta, resample=Resampling.nearest)

static_stack = np.stack([dem_1km, slope_1km, soil_texture_1km], axis=-1)

X_train_all = flatten_inputs(dynamic_stack_train, static_stack)
y_train_all = gssm_train.reshape(-1)

X_test_all = flatten_inputs(dynamic_stack_test, static_stack)
y_test_all = gssm_test.reshape(-1)


mask_train = (~np.isnan(X_train_all).any(axis=1) & (~np.isnan(y_train_all)))
X_train_clean = X_train_all[mask_train]
y_train_clean = y_train_all[mask_train]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1], )),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adma', loss='mse', metrics=['mae'])
model.fit(X_train_scaled, y_train_clean, epochs=1, batch_size=32, validation_split=0.2)

mask_test = ~np.isnan(X_test_all).any(axis=1)
X_test_valid = X_test_all[mask_test]
X_test_scaled = scaler.transform(X_test_valid)

predictions = np.full(X_test_all.shape[0], np.nan)
predictions[mask_test] = model.predict(X_test_scaled).flatten()

pred_map = predictions.reshape((366, *target_shape))


with rasterio.open("../../Data/SMAP/SMAP_downscaled_smap_test.tif", 'w', driver='GTiff',
        height=target_shape[0], width=target_shape[1], count=1,
        dtype='float32', crs=gssm_meta["crs"], transform=gssm_meta["transform"]) as dst:
    dst.write(pred_map[0], 1)


pred_flat = pred_map.reshape(366, -1)
obs_flat = gssm_test.reshape(366, -1)

valid_mask = (~np.isnan(pred_flat)) & (~np.isnan(obs_flat))
pred_valid = pred_flat[valid_mask]
obs_valid = obs_flat[valid_mask]

r2 = r2_score(obs_valid, pred_valid)
rmse = np.sqrt(mean_squared_error(obs_valid, pred_valid))

print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
