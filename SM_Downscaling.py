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
from scipy.signal import savgol_filter
from pyproj import Transformer
from rasterio.transform import rowcol, xy

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)


smap_sm_path = "../../Data/SMAP/SMAP_2016_2022_SM_NL_Daily.tif"
gssm_sm_path = "../../Data/GSSM/GSSM_2016_2020_SM_NL_Daily_1km.tif"
ndvi_path = "../../Data/VIIRS/VIIRS_NDVI_2016_2021_1km.tif"
lst_day_path = "../../Data/VIIRS/VIIRS_LST_Day_2017_2021_1km.tif"
lst_night_path = "../../Data/VIIRS/VIIRS_LST_Night_2017_2021_1km.tif"
dem_path = "../../Data/StaticVars/DEM_Map_90m.tif"
slope_path = "../../Data/StaticVars/Slope_Map_90m.tif"
soil_texture_path = "../../Data/StaticVars/SoilTexture_Map_250m.tif"
smap_sm_am_path = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_AM_NL_Daily.tif"
smap_sm_pm_path = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_PM_NL_Daily.tif"
sm_test_path = "../../Data/dataverse_files/1_station_measurements/2_calibrated/ITCSM_05_cd.csv"


def read_tif(tif):
    with rasterio.open(tif) as src:
        data = src.read().astype(np.float32)
        meta = src.meta
        bands = src.descriptions
    return data, meta, bands


def fill_missing_data(data, max_window=28):
    T, W, H = data.shape
    data_reshaped = data.reshape(T, -1)
    filled = np.empty_like(data_reshaped)

    for idx in range(data_reshaped.shape[1]):
        series = pd.Series(data_reshaped[:, idx])
        series_filled = series.fillna(method="ffill", limit=max_window)
        filled[:, idx] = series_filled.to_numpy()

    return filled.reshape(T, W, H)


def resampling_data(src_data, src_meta, target_shape, resample=Resampling.bilinear):
    T, H, W = src_data.shape
    resampled = np.zeros((T, target_shape[0], target_shape[1]), dtype=np.float32)

    for t in range(T):
        with rasterio.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=H,
                width=W,
                count=1,
                dtype=src_data.dtype,
                crs=src_meta["crs"],
                transform=src_meta["transform"]
            ) as dataset:
                dataset.write(src_data[t, :, :], 1)

            with memfile.open() as dataset:
                resampled[t] = dataset.read(
                    out_shape=(1, target_shape[0], target_shape[1]),
                    resampling=resample
                )[0]
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


smap_sm_am_data, smap_sm_am_meta, smap_sm_am_bands = read_tif(smap_sm_am_path)
smap_sm_pm_data, smap_sm_pm_meta, smap_sm_pm_bands = read_tif(smap_sm_pm_path)

smap_sm_shape = (smap_sm_am_meta["height"], smap_sm_am_meta["width"])

smap_sm_combined_data = np.where(
    np.isnan(smap_sm_am_data) & np.isnan(smap_sm_pm_data),
    np.nan,
    np.nanmean(np.stack([smap_sm_am_data, smap_sm_pm_data]), axis=0)
)
smap_sm_filled = fill_missing_data(smap_sm_combined_data)[366:366+1826]

ndvi_data, ndvi_meta, ndvi_bands = read_tif(ndvi_path)
ndvi_data = ndvi_data[366:]
ndvi_bands = ndvi_bands[366:]

lst_day_data, lst_day_meta, lst_day_bands = read_tif(lst_day_path)
lst_night_data, lst_night_meta, lst_night_bands = read_tif(lst_night_path)

dem_data, dem_meta, dem_band = read_tif(dem_path)
slope_data, slope_meta, slope_band = read_tif(slope_path)
soil_texture_data, soil_texture_meta, soil_texture_band = read_tif(soil_texture_path)


ndvi_filled = fill_missing_data(ndvi_data)
lst_day_filled = fill_missing_data(lst_day_data)
lst_night_filled = fill_missing_data(lst_night_data)

ndvi_resampled = resampling_data(ndvi_filled, ndvi_meta, smap_sm_shape)
lst_day_resampled = resampling_data(lst_day_filled, lst_day_meta, smap_sm_shape)
lst_night_resampled = resampling_data(lst_night_filled, lst_night_meta, smap_sm_shape)

dem_resampled = resample_static_data(dem_data, dem_meta, smap_sm_shape)
slope_resampled = resample_static_data(slope_data, slope_meta, smap_sm_shape)
soil_texture_resampled = resample_static_data(soil_texture_data, soil_texture_meta, smap_sm_shape, resample=Resampling.nearest)


dynamic_inputs_stack = np.stack([ndvi_resampled, lst_day_resampled, lst_night_resampled], axis=-1)
static_inputs_stack = np.stack([dem_resampled, slope_resampled, soil_texture_resampled], axis=-1)

X_data = flatten_inputs(dynamic_inputs_stack, static_inputs_stack)
y_data = smap_sm_filled.reshape(-1)

combined_mask = (~np.isnan(y_data)) & (~np.isnan(X_data).any(axis=1))

X_data = X_data[combined_mask]
y_data = y_data[combined_mask]

print(X_data.shape)
print(y_data.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_scaled, y_data, epochs=1, batch_size=32, validation_split=0.2)

target_shape_1km = (lst_day_meta["height"], lst_day_meta["width"])

ndvi_1km = ndvi_filled
lst_day_1km = lst_day_filled
lst_night_1km = lst_night_filled

ndvi_1km_filled = fill_missing_data(ndvi_1km)
lst_day_1km_filled = fill_missing_data(lst_day_1km)
lst_night_1km_filled = fill_missing_data(lst_night_1km)


dynamic_inputs_1km = np.stack([ndvi_1km_filled, lst_day_1km_filled, lst_night_1km_filled], axis=-1)

dem_1km = resample_static_data(dem_data, dem_meta, target_shape_1km)
slope_1km = resample_static_data(slope_data, slope_meta, target_shape_1km)
soil_texture_1km = resample_static_data(soil_texture_data, soil_texture_meta, target_shape_1km, resample=Resampling.nearest)

static_inputs_1km = np.stack([dem_1km, slope_1km, soil_texture_1km], axis=-1)

X_1km = flatten_inputs(dynamic_inputs_1km, static_inputs_1km)


valid_mask = ~np.isnan(X_1km).any(axis=1)
X_1km_valid = X_1km[valid_mask]

X_1km_scaled = scaler.transform(X_1km_valid)
y_pred_1km = model.predict(X_1km_scaled)

pred_grid = np.full(X_1km.shape[0], np.nan)
pred_grid[valid_mask] = y_pred_1km.flatten()
pred_map = pred_grid.reshape((1826, lst_day_meta["height"], lst_day_meta["width"]))


# output_path = "../../Data/GSSM/GSSM_Predicted_SoilMoisture_2017_2021_1km.tif"
# T, H, W = pred_map.shape
#
# new_meta = lst_day_meta.copy()
# new_meta.update({
#     "count": T,
#     "dtype": "float32"
# })
#
#
# with rasterio.open(output_path, "w", **new_meta) as dst:
#     for i in range(T):
#         dst.write(pred_map[i, :, :].astype("float32"), i + 1)

print("Done")
########################################################################################################################
# print("Starting!")
# param_grid = {
#     'hidden_layer_sizes': [(21,), (42,), (21, 21)],
#     'activation': ['relu'],
#     'solver': ['adam'],
#     'alpha': [0.0001, 0.001],
#     'learning_rate': ['adaptive']
# }
#
#
# grid_search = GridSearchCV(
#     MLPRegressor(max_iter=500),
#     param_grid,
#     cv=10,
#     scoring='r2',
#     verbose=1
# )
#
# grid_search.fit(X_scaled, y_data)
# best_model = grid_search.best_estimator_
#
# print(best_model)

# cv_results_df = pd.DataFrame(grid_search.cv_results_)
# print(cv_results_df[[
#     'params',
#     'mean_test_score',
#     'std_test_score',
#     'rank_test_score'
# ]].sort_values('rank_test_score'))
#
#
# for alpha in cv_results_df['param_alpha'].unique():
#     subset = cv_results_df[cv_results_df['param_alpha'] == alpha]
#     plt.plot(subset['param_hidden_layer_sizes'].astype(str), subset['mean_test_score'], label=f'alpha={alpha}')
#
# plt.ylabel('Mean R² (CV)')
# plt.xlabel('Hidden layer sizes')
# plt.legend()
# plt.title('Grid Search CV Results')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# smap_sm_avg_path = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_AM_PM_NL_Daily.tif"
# with rasterio.open(smap_sm_avg_path, 'w', **smap_sm_am_meta) as dst:
#     dst.write(smap_combined.astype(np.float32))
# print("Done!")

# original_series = ndvi_data[:, 1, 1]
# filled_series = ndvi_filled[:, 1, 1]
#
# days = np.arange(len(original_series))
#
# plt.figure(figsize=(12, 5))
# plt.plot(days, original_series, label='Original', color='red', linestyle='--', marker='*', markersize=5, alpha=0.6)
# plt.plot(days, filled_series, label='Filled (last 28 days)', color='blue', marker='o', markersize=3, alpha=0.6)
#
# plt.xlabel('Days since 2017-01-01')
# plt.ylabel('NDVI')
# plt.title(f'NDVI at Pixel ({1}, {1})')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# def resample_to_match(source_array, source_meta, target_meta):
#     T = source_array.shape[0]
#     H_target, W_target = target_meta['height'], target_meta['width']
#     resampled = np.empty((T, H_target, W_target), dtype=np.float32)
#     for t in range(T):
#         reproject(
#             source=source_array[t],
#             destination=resampled[t],
#             src_transform=source_meta['transform'],
#             src_crs=source_meta['crs'],
#             dst_transform=target_meta['transform'],
#             dst_crs=target_meta['crs'],
#             resampling=Resampling.bilinear
#         )
#     return resampled

# out_path = "../../Data/StaticVars/SoilTexture_Map_Resampled_10km.tif"
# H, W = soil_texture_resampled.shape
# new_meta = smap_sm_am_meta.copy()
# new_meta.update({
#     "count":1,
#     "height":H,
#     "width":W,
#     "dtype":"float32"
# })
#
# with rasterio.open(out_path, "w", **new_meta) as src:
#     src.write(soil_texture_resampled, 1)
#
# output_path = "../../Data/VIIRS/VIIRS_NDVI_Day_Resampled_2017_2021_10km.tif"
# T, H, W = ndvi_resampled.shape
#
# new_meta = smap_sm_am_meta.copy()
# new_meta.update({
#     "count": T,
#     "dtype": "float32"
# })
#
# with rasterio.open(output_path, "w", **new_meta) as dst:
#     for i in range(T):
#         dst.write(ndvi_resampled[i, :, :], i + 1)
#
# print("Done")


# ndvi_path = "../../Data/VIIRS/VIIRS_NDVI_Resampled_2017_2021_10km.tif"
# lst_day_path = "../../Data/VIIRS/VIIRS_LST_Day_Resampled_2017_2021_10km.tif"
# lst_night_path = "../../Data/VIIRS/VIIRS_LST_Night_Resampled_2017_2021_10km.tif"
# T, H, W = ndvi_resampled.shape
#
# new_meta = smap_sm_am_meta.copy()
# new_meta.update({
#     "count": T,
#     "dtype": "float32"
# })
#
# with rasterio.open(ndvi_path, "w", **new_meta) as dst:
#     for i in range(T):
#         dst.write(ndvi_resampled[i, :, :], i + 1)
#
# with rasterio.open(lst_day_path, "w", **new_meta) as dst:
#     for i in range(T):
#         dst.write(lst_day_resampled[i, :, :], i + 1)
#
# with rasterio.open(lst_night_path, "w", **new_meta) as dst:
#     for i in range(T):
#         dst.write(lst_night_resampled[i, :, :], i + 1)
