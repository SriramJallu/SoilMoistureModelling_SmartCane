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
from scipy.signal import savgol_filter
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from skimage.transform import resize
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from rasterio.transform import rowcol
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
        series_filled = pd.Series(
            savgol_filter(series.interpolate().fillna(method='bfill'), window_length=7, polyorder=2))
        filled[:, idx] = series_filled.to_numpy()
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


def get_valid_dates(bands, suffix):
    return set(
        b.replace(f"_{suffix}", "") for b in bands if b.endswith(f"_{suffix}")
    )


def filter_by_dates(data, bands, suffix, common_dates):
    filtered_indices = []
    filtered_band_names = []

    for i, b in enumerate(bands):
        if b.endswith(f"_{suffix}"):
            date = b.replace(f"_{suffix}", "")
            if date in common_dates:
                filtered_indices.append(i)
                filtered_band_names.append(b)

    return data[filtered_indices], filtered_band_names


smap_sm_am_path = "../../Data/SMAP/SMAP_L3_2015_2020_SoilMoisture_AM_NL_Daily.tif"
smap_sm_pm_path = "../../Data/SMAP/SMAP_L3_2015_2020_SoilMoisture_PM_NL_Daily.tif"
gssm_sm_path = "../../Data/GSSM/GSSM_2015_2020_SM_Daily_1km.tif"
ndvi_path = "../../Data/VIIRS/VIIRS_NDVI_4326_2015_2020_1km.tif"
lst_day_path = "../../Data/VIIRS/VIIRS_LST_Day_4326_2015_2020_1km.tif"
lst_night_path = "../../Data/VIIRS/VIIRS_LST_Night_4326_2015_2020_1km.tif"
modis_terra_lst_day_path = "../../Data/MODIS/MODIS_TERRA_LST_Day_2015_2020_1km.tif"
modis_terra_lst_night_path = "../../Data/MODIS/MODIS_TERRA_LST_Night_2015_2020_1km.tif"
modis_et_path = "../../Data/MODIS/MODIS_ET_8Day_2015_2020_500m.tif"
dem_path = "../../Data/StaticVars/DEM_Map_90m.tif"
slope_path = "../../Data/StaticVars/Slope_Map_90m.tif"
soil_texture_path = "../../Data/StaticVars/SoilTexture_Map_250m.tif"

sm_test_path = "../../Data/dataverse_files/1_station_measurements/2_calibrated/ITCSM_10_cd.csv"
lat, lon = 52.2, 6.65944

smap_sm_am_data, smap_sm_am_meta, smap_sm_am_bands = read_tif(smap_sm_am_path)
smap_sm_pm_data, smap_sm_pm_meta, smap_sm_pm_bands = read_tif(smap_sm_pm_path)
gssm_data, gssm_meta, gssm_bands = read_tif(gssm_sm_path)
ndvi_data, ndvi_meta, ndvi_bands = read_tif(ndvi_path)
lst_day_data, lst_day_meta, lst_day_bands = read_tif(modis_terra_lst_day_path)
lst_night_data, lst_night_meta, lst_night_bands = read_tif(modis_terra_lst_night_path)
dem, dem_meta, dem_band = read_tif(dem_path)
slope, slope_meta, slope_band = read_tif(slope_path)
soil_texture, soil_texture_meta, soil_texture_band = read_tif(soil_texture_path)
et_data, et_meta, et_bands = read_tif(modis_et_path)


gssm_dates = get_valid_dates(gssm_bands, "GSSM")
smap_am_dates = get_valid_dates(smap_sm_am_bands, "AM")
smap_pm_dates = get_valid_dates(smap_sm_pm_bands, "PM")
ndvi_dates = get_valid_dates(ndvi_bands, "NDVI")
lst_day_dates = get_valid_dates(lst_day_bands, "LST_Day")
lst_night_dates = get_valid_dates(lst_night_bands, "LST_Night")
et_dates = get_valid_dates(et_bands, "ET")

common_dates = sorted(ndvi_dates & lst_day_dates & lst_night_dates & smap_am_dates & smap_pm_dates & gssm_dates)
print(common_dates[:5])

gssm_data, gssm_bands_filt = filter_by_dates(gssm_data, gssm_bands, "GSSM", common_dates)
smap_sm_am_data, smap_am_bands_filt = filter_by_dates(smap_sm_am_data, smap_sm_am_bands, "AM", common_dates)
smap_sm_pm_data, smap_pm_bands_filt = filter_by_dates(smap_sm_pm_data, smap_sm_pm_bands, "PM", common_dates)
ndvi_data, ndvi_bands_filt = filter_by_dates(ndvi_data, ndvi_bands, "NDVI", common_dates)
lst_day_data, lst_day_bands_filt = filter_by_dates(lst_day_data, lst_day_bands, "LST_Day", common_dates)
lst_night_data, lst_night_bands_filt = filter_by_dates(lst_night_data, lst_night_bands, "LST_Night", common_dates)
# et_data, et_bands_filt = filter_by_dates(et_data, et_bands, "ET", common_dates)

smap = np.nanmean(np.stack([smap_sm_am_data, smap_sm_pm_data]), axis=0)
smap_filled = fill_missing_data(smap)

target_shape = ndvi_data.shape[1:]
target_transform = ndvi_meta["transform"]
target_crs = ndvi_meta["crs"]

new_meta = {
    "crs": target_crs,
    "transform": target_transform
}

print(dem_meta["crs"])
print(target_crs)
print(ndvi_data.shape)

smap_resampled = resampling_data(smap_filled, smap_sm_am_meta, target_shape, target_transform)
gssm_resampled = resampling_data(gssm_data, gssm_meta, target_shape, target_transform)
# et_resampled = resampling_data(et_data, et_meta, target_shape, target_transform)

ndvi_filled = fill_missing_data(ndvi_data)
lst_day_filled = fill_missing_data(lst_day_data)
lst_night_filled = fill_missing_data(lst_night_data)

smap_resampled_train, smap_resampled_test = smap_resampled, smap_resampled[-366:]
ndvi_train, ndvi_test = ndvi_filled, ndvi_filled[-366:]
lst_day_train, lst_day_test = lst_day_filled, lst_day_filled[-366:]
lst_night_train, lst_night_test = lst_night_filled, lst_night_filled[-366:]
gssm_train, gssm_test = gssm_resampled, gssm_resampled[-366:]
# et_train, et_test = et_resampled/8, et_resampled[-366:]/8

print("SMAP Shape:", smap_resampled_train.shape)
print("GSSM Shape:", gssm_train.shape)
print("NDVI Shape:", ndvi_train.shape)
print("LST Day Shape:", lst_day_train.shape)
print("LST Night Shape:", lst_night_train.shape)

dynamic_stack_train = np.stack([smap_resampled_train, ndvi_train, lst_day_train, lst_night_train], axis=-1)
dynamic_stack_test = np.stack([smap_resampled_test, ndvi_test, lst_day_test, lst_night_test], axis=-1)


dem_1km = resample_static_data(dem, new_meta, target_shape)
slope_1km = resample_static_data(slope, new_meta, target_shape)
soil_texture_1km = resample_static_data(soil_texture, new_meta, target_shape, resample=Resampling.nearest)

static_stack = np.stack([dem_1km, slope_1km, soil_texture_1km], axis=-1)
static_stack_train_expanded = np.expand_dims(static_stack, axis=0).repeat(gssm_train.shape[0], axis=0)
static_stack_test_expanded = np.expand_dims(static_stack, axis=0).repeat(gssm_test.shape[0], axis=0)

# X_train_all = flatten_inputs(dynamic_stack_train, static_stack)
X_train_all = np.concatenate([dynamic_stack_train, static_stack_train_expanded], axis=-1)
y_train_all = gssm_train.reshape(gssm_train.shape[0], gssm_train.shape[1], gssm_train.shape[2], 1)

y_mask_train = ~np.isnan(y_train_all).squeeze(-1)
x_mask_train = ~np.isnan(X_train_all).any(axis=-1)
combined_mask_train = y_mask_train & x_mask_train

X_train_clean = X_train_all[combined_mask_train]
y_train_clean = y_train_all[combined_mask_train]

# X_test_all = flatten_inputs(dynamic_stack_test, static_stack)
X_test_all = np.concatenate([dynamic_stack_test, static_stack_test_expanded], axis=-1)
y_test_all = gssm_test.reshape(gssm_test.shape[0], gssm_test.shape[1], gssm_test.shape[2], 1)

y_mask_test = ~np.isnan(y_test_all).squeeze(-1)
x_mask_test = ~np.isnan(X_test_all).any(axis=-1)
combined_mask_test = y_mask_test & x_mask_test

X_test_clean = X_test_all[x_mask_test]
y_test_clean = y_test_all[y_mask_test]

print("dynamic_stack_train shape:", dynamic_stack_train.shape)
print("static_stack shape:", static_stack.shape)
print("gssm_train shape:", gssm_train.shape)

print(X_test_all.shape)
print(y_train_all.shape)

# mask_train = (~np.isnan(X_train_all).any(axis=1) & (~np.isnan(y_train_all)))
# X_train_clean = X_train_all[mask_train]
# y_train_clean = y_train_all[mask_train]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)

# y_scaler = StandardScaler()
# y_train_scaled = y_scaler.fit_transform(y_train_clean.reshape(-1, 1))

print("GSSM train min/max:", np.nanmin(gssm_train), np.nanmax(gssm_train))
print("y_train_clean min/max:", np.min(y_train_clean), np.max(y_train_clean))


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1], )),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(1, activation='relu')
# ])
#
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.fit(X_train_scaled, y_train_clean, epochs=1, batch_size=32, validation_split=0.2)
#
# model_name = f"sm_downscaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
# model.save(f"../../Models/{model_name}")

# model = tf.keras.models.load_model("../../Models/sm_downscaling_20250610_104655.h5")

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.6,
    colsample_bytree=1,
    random_state=123,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train_clean.ravel())


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
# grid_search.fit(X_train_scaled, y_train_clean.ravel())
# model = grid_search.best_estimator_
# print("Best parameters found: ", grid_search.best_params_)
# print("Best RMSE (neg): ", grid_search.best_score_)

# mask_test = ~np.isnan(X_test_all).any(axis=1)
# X_test_valid = X_test_all[mask_test]

X_test_scaled = scaler.transform(X_test_clean)

predictions_valid = model.predict(X_test_scaled)
# predictions_valid = y_scaler.inverse_transform(predictions_valid_scaled.reshape(-1, 1))

pred_map_flat = np.full(X_test_all.shape[:-1], np.nan, dtype=np.float32)
pred_map_flat[x_mask_test] = predictions_valid.flatten()
pred_map = pred_map_flat.reshape((366, target_shape[0], target_shape[1]))

with rasterio.open("../../Data/SMAP/SMAP_downscaled_appraoch2_smap_test.tif", 'w', driver='GTiff',
    height=target_shape[0], width=target_shape[1], count=pred_map.shape[0],
    dtype='float32', crs=new_meta["crs"], transform=new_meta["transform"]) as dst:
    for i in range(pred_map.shape[0]):
        dst.write(pred_map[i], i + 1)


transform = gssm_meta["transform"]
crs = gssm_meta["crs"]
transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

x, y = transformer.transform(lon, lat)
row, col = rowcol(transform, x, y)
print("Pixel location:", row, col)

pred_series = pred_map[:, row, col]

valid_mask = ~np.isnan(pred_series)
pred_valid = pred_series[valid_mask]


common_dates_dt = pd.to_datetime(common_dates[-366:], format="%Y_%m_%d").normalize()
# date_range_2020 = pd.date_range(start="2020-01-01", periods=366, freq="D")
pred_series = pd.Series(pred_series, index=common_dates_dt)

headers = pd.read_csv(sm_test_path, skiprows=18, nrows=0).columns.tolist()
sm_test = pd.read_csv(sm_test_path, skiprows=20, parse_dates=["Date time"], names=headers)

sm_test["Date time"] = pd.to_datetime(sm_test["Date time"], format='%d-%m-%Y %H:%M', errors='coerce')
sm_test = sm_test[sm_test["Date time"] >= '2015-04-01']
sm_test = sm_test[sm_test[" 5 cm SM"] >= 0]
sm_test = sm_test.set_index("Date time")
sm_test.index = sm_test.index.normalize()
sm_test = sm_test.resample("D").mean()
sm_test_common = sm_test.loc[common_dates_dt]

combined_df = pd.DataFrame({
    "pred": pred_series,
    "insitu": sm_test_common[" 5 cm SM"]
})

combined_df = combined_df[combined_df.index >= '2020-01-01']
combined_df = combined_df.dropna()
rmse_insitu = mean_squared_error(combined_df["insitu"], combined_df["pred"], squared=False)
r2_insitu = r2_score(combined_df["insitu"], combined_df["pred"])

print(f"RMSE vs in-situ: {rmse_insitu:.4f}")
print(f"R² vs in-situ: {r2_insitu:.4f}")

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

# pred_flat = pred_map.reshape(366, -1)
# obs_flat = gssm_test.reshape(366, -1)
#
#
# y_true_mask = ~np.isnan(obs_flat).squeeze(-1)
# y_pred_mask = ~np.isnan(pred_flat).squeeze(-1)
#
# valid_mask = (~np.isnan(y_true_mask)) & (~np.isnan(y_pred_mask))
# pred_valid = pred_flat[valid_mask]
# obs_valid = obs_flat[valid_mask]
#
# r2 = r2_score(obs_valid, pred_valid)
# rmse = np.sqrt(mean_squared_error(obs_valid, pred_valid))
#
# print(f"R²: {r2:.4f}")
# print(f"RMSE: {rmse:.4f}")


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