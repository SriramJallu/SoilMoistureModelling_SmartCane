import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from pyproj import Transformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from skimage.transform import resize
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform


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

    dynamic_flat = dynamic_vars.reshape(T * H * W, D)
    static_flat = np.tile(static_vars.reshape(H * W, S), (T, 1))

    return np.concatenate([dynamic_flat, static_flat], axis=1)


smap_sm_am_path = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_AM_NL_Daily.tif"
smap_sm_pm_path = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_PM_NL_Daily.tif"
gssm_sm_path = "../../Data/GSSM/GSSM_2017_2020_SM_NL_Daily_1km.tif"
ndvi_path = "../../Data/VIIRS/VIIRS_NDVI_4326_2016_2021_1km.tif"
lst_day_path = "../../Data/VIIRS/VIIRS_LST_Day_4326_2017_2021_1km.tif"
lst_night_path = "../../Data/VIIRS/VIIRS_LST_Night_4326_2017_2021_1km.tif"
dem_path = "../../Data/StaticVars/DEM_Map_90m.tif"
slope_path = "../../Data/StaticVars/Slope_Map_90m.tif"
soil_texture_path = "../../Data/StaticVars/SoilTexture_Map_250m.tif"

sm_test_path = "../../Data/dataverse_files/1_station_measurements/2_calibrated/ITCSM_05_cd.csv"


smap_sm_am, smap_sm_am_meta = read_tif(smap_sm_am_path)
smap_sm_pm, smap_sm_pm_meta = read_tif(smap_sm_pm_path)
gssm, gssm_meta = read_tif(gssm_sm_path)
ndvi, ndvi_meta = read_tif(ndvi_path)
lst_day, lst_day_meta = read_tif(lst_day_path)
lst_night, lst_night_meta = read_tif(lst_night_path)
dem, dem_meta = read_tif(dem_path)
slope, slope_meta = read_tif(slope_path)
soil_texture, soil_texture_meta = read_tif(soil_texture_path)

target_shape = ndvi.shape[1:]
target_transform = ndvi_meta["transform"]
target_crs = ndvi_meta["crs"]

smap = np.nanmean(np.stack([smap_sm_am, smap_sm_pm]), axis=0)
smap_filled = fill_missing_data(smap)[366:366+1461]

new_meta = {
    "crs": target_crs,
    "transform": target_transform
}

print(dem_meta["crs"])
print(target_crs)

smap_resampled = resampling_data(smap_filled, smap_sm_am_meta, target_shape, target_transform)
gssm_resampled = resampling_data(gssm, gssm_meta, target_shape, target_transform)

ndvi_filled = fill_missing_data(ndvi)[366:366+1461]
lst_day_filled = fill_missing_data(lst_day)[:1461]
lst_night_filled = fill_missing_data(lst_night)[:1461]

smap_resampled_train, smap_resampled_test = smap_resampled[:1095], smap_resampled[1095:]
ndvi_train, ndvi_test = ndvi_filled[:1095], ndvi_filled[1095:]
lst_day_train, lst_day_test = lst_day_filled[:1095], lst_day_filled[1095:]
lst_night_train, lst_night_test = lst_night_filled[:1095], lst_night_filled[1095:]
gssm_train, gssm_test = gssm_resampled[:1095], gssm_resampled[1095:]

dynamic_stack_train = np.stack([smap_resampled_train, ndvi_train, lst_day_train, lst_night_train], axis=-1)
dynamic_stack_test = np.stack([smap_resampled_test, ndvi_test, lst_day_test, lst_night_test], axis=-1)


dem_1km = resample_static_data(dem, new_meta, target_shape)
slope_1km = resample_static_data(slope, new_meta, target_shape)
soil_texture_1km = resample_static_data(soil_texture, new_meta, target_shape, resample=Resampling.nearest)

static_stack = np.stack([dem_1km, slope_1km, soil_texture_1km], axis=-1)

X_train_all = flatten_inputs(dynamic_stack_train, static_stack)
y_train_all = gssm_train.reshape(-1)

X_test_all = flatten_inputs(dynamic_stack_test, static_stack)
y_test_all = gssm_test.reshape(-1)

print("dynamic_stack_train shape:", dynamic_stack_train.shape)
print("static_stack shape:", static_stack.shape)
print("gssm_train shape:", gssm_train.shape)

print(X_test_all.shape)
print(y_train_all.shape)

mask_train = (~np.isnan(X_train_all).any(axis=1) & (~np.isnan(y_train_all)))
X_train_clean = X_train_all[mask_train]
y_train_clean = y_train_all[mask_train]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train_clean.reshape(-1, 1))

print("GSSM train min/max:", np.nanmin(gssm_train), np.nanmax(gssm_train))
print("y_train_clean min/max:", np.min(y_train_clean), np.max(y_train_clean))


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1], )),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train_scaled, y_train_scaled, epochs=2, batch_size=32, validation_split=0.2)

model_name = f"sm_downscaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
model.save(f"../../Models/{model_name}")

# model = tf.keras.models.load_model("../../Models/sm_downscaling_lstm_20250530_085224.h5")

mask_test = ~np.isnan(X_test_all).any(axis=1)
X_test_valid = X_test_all[mask_test]

X_test_scaled = scaler.transform(X_test_valid)

predictions_valid_scaled = model.predict(X_test_scaled, batch_size=32).flatten()
predictions_valid = y_scaler.inverse_transform(predictions_valid_scaled.reshape(-1, 1)).flatten()

pred_map_flat = np.full(X_test_all.shape[0], np.nan, dtype=np.float32)
pred_map_flat[mask_test] = predictions_valid

pred_map = pred_map_flat.reshape((366, target_shape[0], target_shape[1]))

with rasterio.open("../../Data/SMAP/SMAP_downscaled_smap_test.tif", 'w', driver='GTiff',
    height=target_shape[0], width=target_shape[1], count=pred_map.shape[0],
    dtype='float32', crs=new_meta["crs"], transform=new_meta["transform"]) as dst:
    for i in range(pred_map.shape[0]):
        dst.write(pred_map[i], i + 1)


pred_flat = pred_map.reshape(366, -1)
obs_flat = gssm_test.reshape(366, -1)

valid_mask = (~np.isnan(pred_flat)) & (~np.isnan(obs_flat))
pred_valid = pred_flat[valid_mask]
obs_valid = obs_flat[valid_mask]

r2 = r2_score(obs_valid, pred_valid)
rmse = np.sqrt(mean_squared_error(obs_valid, pred_valid))

print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")


# rf_model = RandomForestRegressor(
#     n_estimators=100,
#     max_depth=None,
#     random_state=42,
#     n_jobs=-1
# )
#
# rf_model.fit(X_train_scaled, y_train_scaled.ravel())
#
#
# X_test_scaled = scaler.transform(X_test_all)
#
#
# mask_test = (~np.isnan(X_test_scaled).any(axis=1) & (~np.isnan(y_test_all)))
# X_test_clean = X_test_scaled[mask_test]
# y_test_clean = y_test_all[mask_test]
#
#
# y_pred_scaled = rf_model.predict(X_test_clean)
#
#
# y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
#
# rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred))
# r2 = r2_score(y_test_clean, y_pred)
#
# print(f"Test RMSE: {rmse:.4f}")
# print(f"Test R²: {r2:.4f}")

# def plot_variable_stack(stack, title, days=[25, 45, 60]):
#     fig, axs = plt.subplots(1, len(days), figsize=(15, 5))
#     for i, day in enumerate(days):
#         axs[i].imshow(stack[day], cmap='viridis',
#                       vmin=np.nanpercentile(stack, 5),
#                       vmax=np.nanpercentile(stack, 95))
#         axs[i].set_title(f"{title} - Day {day}")
#         axs[i].axis('off')
#
#     plt.tight_layout()
#     plt.suptitle(title, y=1.05)
#     plt.show()

# plot_variable_stack(smap_sm_am, "SMAP am (Train)")
# plot_variable_stack(smap, "SMAP (Train)")
# plot_variable_stack(smap_filled, "SMAP Filled (Train)")
# plot_variable_stack(smap_resampled, "SMAP Resampled (Train)")
# plot_variable_stack(gssm, "GSSM (Train)")
# plot_variable_stack(gssm_resampled, "GSSM Resampled (Train)")
# plot_variable_stack(gssm_train, "GSSM (Train)")
#
#
# plot_variable_stack(ndvi, "NDVI")
# plot_variable_stack(ndvi_filled, "NDVI Filled")
# plot_variable_stack(ndvi_train, "NDVI (Train)")
#
# plot_variable_stack(lst_day, "LST Day")
# plot_variable_stack(lst_day_filled, "LST Day Filled")
# plot_variable_stack(lst_day_train, "LST Day (Train)")
#
# plot_variable_stack(lst_night, "LST Night")
# plot_variable_stack(lst_night_filled, "LST Night Filled")
# plot_variable_stack(lst_night_train, "LST Night (Train)")

# static_vars = ["DEM", "slope", "soil_texture"]
# resampled_vars = [dem_1km, slope_1km, soil_texture_1km]
#
# plt.figure(figsize=(15, 10))
#
# for i, (var, data) in enumerate(zip(static_vars, resampled_vars)):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(data, cmap='terrain' if var in ["DEM", "slope"] else 'viridis')
#     plt.title(f"{var} (resampled)")
#     plt.colorbar()
#     plt.axis("off")
#
# plt.tight_layout()
# plt.show()