import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from pyproj import Transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
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
import xgboost as xgb

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)


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
        # series_filled = series.fillna(method="ffill", limit=max_window)
        # series_filled = series.interpolate(limit=max_window, limit_direction="both")
        series_filled = pd.Series(
            savgol_filter(series.interpolate().fillna(method='bfill'), window_length=7, polyorder=2))
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


# Setting all the paths for tif files
smap_sm_path = "../../Data/SMAP/SMAP_2016_2022_SM_NL_Daily.tif"
gssm_sm_path = "../../Data/GSSM/GSSM_2016_2020_SM_NL_Daily_1km.tif"
ndvi_path = "../../Data/VIIRS/VIIRS_NDVI_4326_2016_2021_1km.tif"
lst_day_path = "../../Data/VIIRS/VIIRS_LST_Day_4326_2017_2021_1km.tif"
lst_night_path = "../../Data/VIIRS/VIIRS_LST_Night_4326_2017_2021_1km.tif"
dem_path = "../../Data/StaticVars/DEM_Map_90m.tif"
slope_path = "../../Data/StaticVars/Slope_Map_90m.tif"
soil_texture_path = "../../Data/StaticVars/SoilTexture_Map_250m.tif"
smap_sm_am_path = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_AM_NL_Daily.tif"
smap_sm_pm_path = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_PM_NL_Daily.tif"
modis_terra_lst_day_path = "../../Data/MODIS/MODIS_TERRA_LST_Day_2017_2021_1km.tif"
modis_terra_lst_night_path = "../../Data/MODIS/MODIS_TERRA_LST_Night_2017_2021_1km.tif"
modis_aqua_lst_day_path = "../../Data/MODIS/MODIS_AQUA_LST_Day_2017_2021_1km.tif"
modis_aqua_lst_night_path = "../../Data/MODIS/MODIS_AQUA_LST_Night_2017_2021_1km.tif"

sm_test_path = "../../Data/dataverse_files/1_station_measurements/2_calibrated/ITCSM_09_cd.csv"


# Load SMAP Data
smap_sm_am_data, smap_sm_am_meta, smap_sm_am_bands = read_tif(smap_sm_am_path)
smap_sm_pm_data, smap_sm_pm_meta, smap_sm_pm_bands = read_tif(smap_sm_pm_path)

smap_sm_shape = (smap_sm_am_meta["height"], smap_sm_am_meta["width"])       # SMAP Shape

smap_sm_combined_data = np.where(                                           # Get the average of SMAP AM and PM Products
    np.isnan(smap_sm_am_data) & np.isnan(smap_sm_pm_data),
    np.nan,
    np.nanmean(np.stack([smap_sm_am_data, smap_sm_pm_data]), axis=0)
)[366:366+1461]

# smap_sm_combined_data, smap_sm_combined_meta, smap_sm_combined_bands = read_tif(smap_sm_path)
# smap_sm_shape = (smap_sm_combined_meta["height"], smap_sm_combined_meta["width"])
# smap_sm_combined_data = smap_sm_combined_data[366:366+1461]

# Load dynamic variables
ndvi_data, ndvi_meta, ndvi_bands = read_tif(ndvi_path)
lst_day_data, lst_day_meta, lst_day_bands = read_tif(lst_day_path)
lst_night_data, lst_night_meta, lst_night_bands = read_tif(lst_night_path)

# Filter dynamic variables between 2017 and 2020
ndvi_data = ndvi_data[366:366+1461]
lst_day_data = lst_day_data[:1461]
lst_night_data = lst_night_data[:1461]

# Fill missing values in dynamic variables
smap_sm_filled = fill_missing_data(smap_sm_combined_data)
ndvi_filled = fill_missing_data(ndvi_data)
lst_day_filled = fill_missing_data(lst_day_data)
lst_night_filled = fill_missing_data(lst_night_data)

# Resample dynamic variables from fine to SMAP resolution
ndvi_resampled = resampling_data(ndvi_filled, ndvi_meta, smap_sm_shape)
lst_day_resampled = resampling_data(lst_day_filled, lst_day_meta, smap_sm_shape)
lst_night_resampled = resampling_data(lst_night_filled, lst_night_meta, smap_sm_shape)

print(ndvi_resampled.shape)
print(lst_day_resampled.shape)
print(lst_night_resampled.shape)

# Load Static variables
dem_data, dem_meta, dem_band = read_tif(dem_path)
slope_data, slope_meta, slope_band = read_tif(slope_path)
soil_texture_data, soil_texture_meta, soil_texture_band = read_tif(soil_texture_path)

# Resample static variables from fine to SMAP resolution
dem_resampled = resample_static_data(dem_data, dem_meta, smap_sm_shape)
slope_resampled = resample_static_data(slope_data, slope_meta, smap_sm_shape)
soil_texture_resampled = resample_static_data(soil_texture_data, soil_texture_meta, smap_sm_shape, resample=Resampling.nearest)

# Stack dynamic and static variables
dynamic_inputs_stack = np.stack([ndvi_resampled, lst_day_resampled, lst_night_resampled], axis=-1)
static_inputs_stack = np.stack([dem_resampled, slope_resampled, soil_texture_resampled], axis=-1)
static_inputs_expanded = np.expand_dims(static_inputs_stack, axis=0).repeat(smap_sm_filled.shape[0], axis=0)

# Flatten and concatenate dynamic & static variables to shape (T*H*W, D+S)
# X_data = flatten_inputs(dynamic_inputs_stack, static_inputs_stack)
X_data = np.concatenate([dynamic_inputs_stack, static_inputs_expanded], axis=-1)
print("New X_data shape", X_data.shape)

# Flatten target variable to shape (T*H*W,)
y_data = smap_sm_filled.reshape(smap_sm_filled.shape[0], smap_sm_filled.shape[1], smap_sm_filled.shape[2], 1)
print("New y_data shape", y_data.shape)

# Create a mask to non-NAN values in X and y data
y_mask = ~np.isnan(y_data).squeeze(-1)
x_mask = ~np.isnan(X_data).any(axis=-1)

combined_mask = y_mask & x_mask

# Filter out NAN values
X_data = X_data[combined_mask]
y_data = y_data[combined_mask]

print(X_data.shape)
print(y_data.shape)

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(1, activation='relu')
# ])
#
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.fit(X_scaled, y_data, epochs=200, batch_size=32, validation_split=0.2)

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=1,
    random_state=123,
    n_jobs=-1
)
model.fit(X_scaled, y_data.ravel())

# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=123, n_jobs=-1)
#
# params = {
#     "n_estimators": [100, 150, 200],
#     "learning_rate": [0.1, 0.01, 0.001],
#     "max_depth": [3, 5, 7],
#     "subsample": [0.4, 0.6, 0.8],
#     'colsample_bytree': [0.6, 0.8, 1.0]
# }
#
# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=params,
#     scoring='neg_mean_squared_error',
#     cv=3,
#     n_jobs=-1,
#     verbose=1
# )
#
# grid_search.fit(X_scaled, y_data.ravel())
# model = grid_search.best_estimator_
# print("Best parameters found: ", grid_search.best_params_)
# print("Best RMSE (neg): ", grid_search.best_score_)

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
static_inputs_1km_expanded = np.expand_dims(static_inputs_1km, axis=0).repeat(ndvi_1km_filled.shape[0], axis=0)


# X_1km = flatten_inputs(dynamic_inputs_1km, static_inputs_1km)
X_1km = np.concatenate([dynamic_inputs_1km, static_inputs_1km_expanded], axis=-1)


valid_mask = ~np.isnan(X_1km).any(axis=-1)
X_1km_valid = X_1km[valid_mask]

X_1km_scaled = scaler.transform(X_1km_valid)
y_pred_1km = model.predict(X_1km_scaled)

pred_grid = np.full(X_1km.shape[:-1], np.nan)
pred_grid[valid_mask] = y_pred_1km.flatten()
pred_map = pred_grid.reshape((1461, lst_day_meta["height"], lst_day_meta["width"]))
pred_map = pred_map[1095:1461]


gssm_data, gssm_meta, gssm_bands = read_tif(gssm_sm_path)
gssm_data = gssm_data[1461-9:1461-9+366]


lat, lon = 52.14639, 6.84306

transform = gssm_meta["transform"]
crs = gssm_meta["crs"]
transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)


x, y = transformer.transform(lon, lat)
row, col = rowcol(transform, x, y)
print("Pixel location:", row, col)

# gssm_series = gssm_data[:, row, col]
pred_series = pred_map[:, row, col]


# pred_flat = pred_map.reshape(1461, -1)
# gssm_flat = gssm_data.reshape(1461, -1)

# valid_mask = ~np.isnan(pred_series) & ~np.isnan(gssm_series)
valid_mask = ~np.isnan(pred_series)
pred_valid = pred_series[valid_mask]


# gssm_valid = gssm_series[valid_mask]
#
# rmse = mean_squared_error(gssm_valid, pred_valid, squared=False)
# r2 = r2_score(gssm_valid, pred_valid)
#
# print(f"RMSE (2020): {rmse:.4f}")
# print(f"R² (2020): {r2:.4f}")


headers = pd.read_csv(sm_test_path, skiprows=18, nrows=0).columns.tolist()
sm_test = pd.read_csv(sm_test_path, skiprows=20, parse_dates=["Date time"], names=headers)

sm_test["Date time"] = pd.to_datetime(sm_test["Date time"], format='%d-%m-%Y %H:%M', errors='coerce')
sm_test = sm_test[sm_test["Date time"] >= '2020-01-01']
# sm_test = sm_test[sm_test[" 10 cm SM"] >= 0]
sm_test = sm_test.set_index("Date time")
sm_test = sm_test.resample("D").mean()

date_range_2020 = pd.date_range(start="2020-01-01", periods=366, freq="D")
pred_2020_series = pd.Series(pred_series, index=date_range_2020)

combined_df = pd.DataFrame({
    "pred": pred_2020_series,
    "insitu": sm_test[" 5 cm SM"]
})

combined_df = combined_df.dropna()
rmse_insitu = mean_squared_error(combined_df["insitu"], combined_df["pred"], squared=False)
r2_insitu = r2_score(combined_df["insitu"], combined_df["pred"])

print(f"RMSE vs in-situ (2020): {rmse_insitu:.4f}")
print(f"R² vs in-situ (2020): {r2_insitu:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(combined_df.index, combined_df["insitu"], label="In-situ SM", linewidth=2)
plt.plot(combined_df.index, combined_df["pred"], label="Predicted SM", linewidth=2)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Soil Moisture", fontsize=12)
plt.title("Soil Moisture Predictions vs In-situ Measurements (2020)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

output_path = "../../Data/SMAP/SMAP_downscaled_appraoch1_smap_test.tif"
T, H, W = pred_map.shape

new_meta = lst_day_meta.copy()
new_meta.update({
    "count": T,
    "dtype": "float32"
})


with rasterio.open(output_path, "w", **new_meta) as dst:
    for i in range(T):
        dst.write(pred_map[i, :, :].astype("float32"), i + 1)

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
