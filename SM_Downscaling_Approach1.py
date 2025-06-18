import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
from rasterio.enums import Resampling
from scipy.signal import savgol_filter
from pyproj import Transformer
from rasterio.transform import rowcol
import xgboost as xgb

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)


def read_tif(tif):
    """ Function to read the rasters, returns the data, metadata and band names"""
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
    """ Function for resampling dynamic data"""
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
    """ Function for resampling static data"""
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


def get_valid_dates(bands, suffix):
    """ Function to extract dates from each raster"""
    return set(
        b.replace(f"_{suffix}", "") for b in bands if b.endswith(f"_{suffix}")
    )


def filter_by_dates(data, bands, suffix, common_dates):
    """ Function to filter the data using common dates"""
    filtered_indices = []
    filtered_band_names = []

    for i, b in enumerate(bands):
        if b.endswith(f"_{suffix}"):
            date = b.replace(f"_{suffix}", "")
            if date in common_dates:
                filtered_indices.append(i)
                filtered_band_names.append(b)

    return data[filtered_indices], filtered_band_names


# Setting all the paths for tif files
smap_sm_path = "../../Data/SMAP/SMAP_2016_2022_SM_NL_Daily.tif"
gssm_sm_path = "../../Data/GSSM/GSSM_2016_2020_SM_NL_Daily_1km.tif"
ndvi_path = "../../Data/VIIRS/VIIRS_NDVI_4326_2015_2020_1km.tif"
lst_day_path = "../../Data/VIIRS/VIIRS_LST_Day_4326_2015_2020_1km.tif"
lst_night_path = "../../Data/VIIRS/VIIRS_LST_Night_4326_2015_2020_1km.tif"
dem_path = "../../Data/StaticVars/DEM_Map_90m.tif"
slope_path = "../../Data/StaticVars/Slope_Map_90m.tif"
soil_texture_path = "../../Data/StaticVars/SoilTexture_Map_250m.tif"
smap_sm_am_path = "../../Data/SMAP/SMAP_L3_2015_2020_SoilMoisture_AM_NL_Daily.tif"
smap_sm_pm_path = "../../Data/SMAP/SMAP_L3_2015_2020_SoilMoisture_PM_NL_Daily.tif"
modis_terra_lst_day_path = "../../Data/MODIS/MODIS_TERRA_LST_Day_2015_2020_1km.tif"
modis_terra_lst_night_path = "../../Data/MODIS/MODIS_TERRA_LST_Night_2015_2020_1km.tif"
modis_aqua_lst_day_path = "../../Data/MODIS/MODIS_AQUA_LST_Day_2017_2021_1km.tif"
modis_aqua_lst_night_path = "../../Data/MODIS/MODIS_AQUA_LST_Night_2017_2021_1km.tif"
modis_et_path = "../../Data/MODIS/MODIS_ET_8Day_2015_2020_500m.tif"
modis_daily_et_path = "../../Data/MODIS/MODIS_ET_Daily_2015_2020.tif"
precip_era5_path = "../../Data/ERA5_NL/ERA5_2015_2020_Precip_NL_Daily.tif"
era5_sm_path = "../../Data/ERA5_NL/ERA5_2015_2020_SM_NL_Daily.tif"

# Path for insitu data and it corresponding lat, lon
sm_test_path = "../../Data/dataverse_files/1_station_measurements/2_calibrated/ITCSM_10_cd.csv"
lat, lon = 52.2, 6.65944

sm_api_path = "../../Data/dataverse_files/Loc_10_API.csv"

# Load SMAP Data
smap_sm_am_data, smap_sm_am_meta, smap_sm_am_bands = read_tif(smap_sm_am_path)
smap_sm_pm_data, smap_sm_pm_meta, smap_sm_pm_bands = read_tif(smap_sm_pm_path)

smap_sm_shape = (smap_sm_am_meta["height"], smap_sm_am_meta["width"])       # SMAP Shape

smap_am_dates = get_valid_dates(smap_sm_am_bands, "AM")
smap_pm_dates = get_valid_dates(smap_sm_pm_bands, "PM")

era5_sm_data, era5_sm_meta, era5_sm_bands = read_tif(era5_sm_path)
# smap_sm_shape = (era5_sm_meta["height"], era5_sm_meta["width"])

# Load dynamic variables
ndvi_data, ndvi_meta, ndvi_bands = read_tif(ndvi_path)
lst_day_data, lst_day_meta, lst_day_bands = read_tif(lst_day_path)
lst_night_data, lst_night_meta, lst_night_bands = read_tif(lst_night_path)
et_data, et_meta_dummy, et_bands = read_tif(modis_daily_et_path)
et_data = np.where(et_data == 0, np.nan, et_data)
et_data_dummy, et_meta, et_bands_dummy = read_tif(modis_et_path)
precip_data, precip_meta, precip_bands = read_tif(precip_era5_path)

# Get the dates of each dynamic variable
ndvi_dates = get_valid_dates(ndvi_bands, "NDVI")
lst_day_dates = get_valid_dates(lst_day_bands, "LST_Day")
lst_night_dates = get_valid_dates(lst_night_bands, "LST_Night")
et_dates = get_valid_dates(et_bands, "ET")
precip_dates = get_valid_dates(precip_bands, "Precip")
era5_sm_dates = get_valid_dates(era5_sm_bands, "SM")

# Filter to keep only the common dates between all the dynamic variables
common_dates = sorted(ndvi_dates & lst_day_dates & lst_night_dates & smap_am_dates & smap_pm_dates & et_dates & precip_dates)
# common_dates = sorted(ndvi_dates & lst_day_dates & lst_night_dates & era5_sm_dates & et_dates & precip_dates)

smap_am_data, smap_am_bands_filt = filter_by_dates(smap_sm_am_data, smap_sm_am_bands, "AM", common_dates)
smap_pm_data, smap_pm_bands_filt = filter_by_dates(smap_sm_pm_data, smap_sm_pm_bands, "PM", common_dates)
ndvi_data, ndvi_bands_filt = filter_by_dates(ndvi_data, ndvi_bands, "NDVI", common_dates)
lst_day_data, lst_day_bands_filt = filter_by_dates(lst_day_data, lst_day_bands, "LST_Day", common_dates)
lst_night_data, lst_night_bands_filt = filter_by_dates(lst_night_data, lst_night_bands, "LST_Night", common_dates)
et_data, et_bands_filt = filter_by_dates(et_data, et_bands, "ET", common_dates)
et_data = et_data * 0.1
precip_data, precip_bands_filt = filter_by_dates(precip_data, precip_bands, "Precip", common_dates)
precip_data = precip_data * 1000
era5_sm_data, era5_sm_bands_filt = filter_by_dates(era5_sm_data, era5_sm_bands, "SM", common_dates)


# Get the average of AM and PM SMAP products
smap_sm_combined_data = np.where(                                           # Get the average of SMAP AM and PM Products
    np.isnan(smap_am_data) & np.isnan(smap_pm_data),
    np.nan,
    np.nanmean(np.stack([smap_am_data, smap_pm_data]), axis=0)
)

# Fill missing values in dynamic variables
smap_sm_filled = fill_missing_data(smap_sm_combined_data)
ndvi_filled = fill_missing_data(ndvi_data)
lst_day_filled = fill_missing_data(lst_day_data)
lst_night_filled = fill_missing_data(lst_night_data)
era5_sm_filled = fill_missing_data(era5_sm_data)
# et_filled = fill_missing_data(et_data) / 8

# Resample dynamic variables from fine to SMAP resolution
ndvi_resampled = resampling_data(ndvi_filled, ndvi_meta, smap_sm_shape)
lst_day_resampled = resampling_data(lst_day_filled, lst_day_meta, smap_sm_shape)
lst_night_resampled = resampling_data(lst_night_filled, lst_night_meta, smap_sm_shape)
et_resampled = resampling_data(et_data, et_meta, smap_sm_shape)

print(ndvi_resampled.shape)
print(lst_day_resampled.shape)
print(lst_night_resampled.shape)
print(et_resampled.shape)

# Load Static variables
dem_data, dem_meta, dem_band = read_tif(dem_path)
slope_data, slope_meta, slope_band = read_tif(slope_path)
soil_texture_data, soil_texture_meta, soil_texture_band = read_tif(soil_texture_path)
soil_texture_data = np.where(soil_texture_data == 0, np.nan, soil_texture_data)

# Resample static variables from fine to SMAP resolution
dem_resampled = resample_static_data(dem_data, dem_meta, smap_sm_shape)
slope_resampled = resample_static_data(slope_data, slope_meta, smap_sm_shape)
soil_texture_resampled = resample_static_data(soil_texture_data, soil_texture_meta, smap_sm_shape, resample=Resampling.nearest)

# Stack dynamic and static variables
dynamic_inputs_stack = np.stack([ndvi_resampled, lst_day_resampled, lst_night_resampled, et_resampled, precip_data], axis=-1)
static_inputs_stack = np.stack([dem_resampled, slope_resampled, soil_texture_resampled], axis=-1)
static_inputs_expanded = np.expand_dims(static_inputs_stack, axis=0).repeat(smap_sm_filled.shape[0], axis=0)
# static_inputs_expanded = np.expand_dims(static_inputs_stack, axis=0).repeat(era5_sm_filled.shape[0], axis=0)

# Flatten and concatenate dynamic & static variables to shape (T*H*W, D+S)
# X_data = flatten_inputs(dynamic_inputs_stack, static_inputs_stack)
X_data = np.concatenate([dynamic_inputs_stack, static_inputs_expanded], axis=-1)
print("New X_data shape", X_data.shape)

# Flatten target variable to shape (T*H*W,)
y_data = smap_sm_filled.reshape(smap_sm_filled.shape[0], smap_sm_filled.shape[1], smap_sm_filled.shape[2], 1)
# y_data = era5_sm_filled.reshape(era5_sm_filled.shape[0], era5_sm_filled.shape[1], era5_sm_filled.shape[2], 1)
print("New y_data shape", y_data.shape)

# Create a mask to non-NAN values in X and y data
y_mask = ~np.isnan(y_data).squeeze(-1)
x_mask = ~np.isnan(X_data).any(axis=-1)

combined_mask = y_mask & x_mask

# Filter out NAN values
X_data = X_data[y_mask]
y_data = y_data[y_mask]

print(X_data.shape)
print(y_data.shape)

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

# Training an MLP model, does not give the best results
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(1, activation='relu')
# ])
#
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.fit(X_scaled, y_data, epochs=200, batch_size=32, validation_split=0.2)

# Train the model, these are the best hyperparameters after performing a grid search
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.6,
    colsample_bytree=1,
    random_state=123,
    n_jobs=-1
)
model.fit(X_scaled, y_data.ravel())

# Grid search for finding best hyperparameters
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

# Feature Importance
feature_names = ["NDVI", "LST Day", "LST Night", "ET", "Precip", "DEM", "Slope", "Soil Texture"]
print("Feature Names:", feature_names)
importances = model.feature_importances_
print("Feature importances:", importances * 100)

# Getting the shape at 1km resolution
target_shape_1km = (lst_day_meta["height"], lst_day_meta["width"])

# Filling missing data at 1km resolution
ndvi_1km_filled = ndvi_filled
lst_day_1km_filled = lst_day_filled
lst_night_1km_filled = lst_night_filled
et_1km_resampled = resampling_data(et_data, et_meta, target_shape_1km)
precip_1km_resmapled = resampling_data(precip_data, precip_meta, target_shape_1km)


# Stacking the dynamic inputs at 1km
dynamic_inputs_1km = np.stack([ndvi_1km_filled, lst_day_1km_filled, lst_night_1km_filled, et_1km_resampled, precip_1km_resmapled], axis=-1)

# Resampling static variables to 1km
dem_1km = resample_static_data(dem_data, dem_meta, target_shape_1km)
slope_1km = resample_static_data(slope_data, slope_meta, target_shape_1km)
soil_texture_1km = resample_static_data(soil_texture_data, soil_texture_meta, target_shape_1km, resample=Resampling.nearest)


# output_path = "../../Data/StaticVars/SoilTexture_Map_Resampled_1km.tif"
# soil_texture_meta.update({
#     "height": soil_texture_1km.shape[0],
#     "width": soil_texture_1km.shape[1],
#     "count": 1,
#     "dtype": soil_texture_1km.dtype,
#     "transform": ndvi_meta["transform"],
#     "driver": "GTiff"
# })
# with rasterio.open(output_path, "w", **soil_texture_meta) as dst:
#     dst.write(soil_texture_1km, 1)

# Stacking static inputs
static_inputs_1km = np.stack([dem_1km, slope_1km, soil_texture_1km], axis=-1)
static_inputs_1km_expanded = np.expand_dims(static_inputs_1km, axis=0).repeat(ndvi_1km_filled.shape[0], axis=0)

# X_1km = flatten_inputs(dynamic_inputs_1km, static_inputs_1km)
X_1km = np.concatenate([dynamic_inputs_1km, static_inputs_1km_expanded], axis=-1)   # Stacking dynamic inputs at 1km

valid_mask = ~np.isnan(X_1km).any(axis=-1)                                          # Masking nan values from inputs
X_1km_valid = X_1km[valid_mask]

X_1km_scaled = scaler.transform(X_1km_valid)                                        # Scaling the inputs at 1km
y_pred_1km = model.predict(X_1km_scaled)                                            # Predicting on 1km inputs

pred_grid = np.full(X_1km.shape[:-1], np.nan)                                       # Grid to store the 1km predictions
pred_grid[valid_mask] = y_pred_1km.flatten()
pred_map = pred_grid.reshape((smap_sm_filled.shape[0], lst_day_meta["height"], lst_day_meta["width"]))
# pred_map = pred_grid.reshape((era5_sm_filled.shape[0], lst_day_meta["height"], lst_day_meta["width"]))


# Getting the corresponding pixel coordinates and the timeseries at that coordinates, given latitude and longitude
transform = ndvi_meta["transform"]
crs = ndvi_meta["crs"]
transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
x, y = transformer.transform(lon, lat)
row, col = rowcol(transform, x, y)
print("Pixel location:", row, col)
pred_series = pred_map[:, row, col]
ndvi_series = ndvi_1km_filled[:, row, col]


# Maksing the nan values
valid_mask = ~np.isnan(pred_series)
pred_valid = pred_series[valid_mask]

# precip_data, precip_meta, precip_bands = read_tif(precip_era5_path)

transform2 = precip_meta["transform"]
crs = precip_meta["crs"]
transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
x, y = transformer.transform(lon, lat)
row2, col2 = rowcol(transform2, x, y)
print("Pixel location:", row2, col2)

# precip_era5_path = "../../Data/ERA5_NL/ERA5_2015_2020_Precip_NL_Daily.tif"
# precip_data, precip_meta, precip_bands = read_tif(precip_era5_path)
precip_data = precip_data[:, row2, col2]
era5_sm_data = era5_sm_data[:, row2, col2]

# Converting common dates to datetime format
common_dates_dt = pd.to_datetime(common_dates, format="%Y_%m_%d").normalize()
pred_series = pd.Series(pred_series, index=common_dates_dt)
ndvi_series = pd.Series(ndvi_series, index=common_dates_dt)
precip_series = pd.Series(precip_data, index=common_dates_dt)
era5_series = pd.Series(era5_sm_data, index=common_dates_dt)
# pred_series = pred_series[pred_series.index < "2017-01-01"]
# common_dates_dt = common_dates_dt[common_dates_dt < "2017-01-01"]


# Reading and filtering the insitu data (csv) to match the common_dates of predictions
headers = pd.read_csv(sm_test_path, skiprows=18, nrows=0).columns.tolist()
sm_test = pd.read_csv(sm_test_path, skiprows=20, parse_dates=["Date time"], names=headers)

sm_test["Date time"] = pd.to_datetime(sm_test["Date time"], format='%d-%m-%Y %H:%M', errors='coerce')
sm_test = sm_test[sm_test["Date time"] >= '2015-01-01']
sm_test = sm_test[sm_test[" 5 cm SM"] >= 0]
sm_test = sm_test.set_index("Date time")
sm_test.index = sm_test.index.normalize()
sm_test = sm_test.resample("D").mean()
sm_test_common = sm_test.loc[common_dates_dt]


headers_api = pd.read_csv(sm_api_path, skiprows=3, nrows=0).columns.tolist()
sm_api = pd.read_csv(sm_api_path, skiprows=4, parse_dates=["time"], names=headers_api)
sm_api["time"] = pd.to_datetime(sm_api["time"], format='%Y-%m-%d', errors='coerce')
sm_api = sm_api.set_index("time")
sm_api_common = sm_api.loc[common_dates_dt]

# Df with insitu and predictions, with datetime
combined_df = pd.DataFrame({
    "pred": pred_series,
    "ndvi": ndvi_series,
    "precip": precip_series,
    "era5_SM": era5_series,
    "insitu": sm_test_common[" 5 cm SM"],
    "SM_API": sm_api_common["soil_moisture_0_to_7cm_mean (m³/m³)"]
})


# Filtering out 2020 (Can filter any range) and Validation metrics calculations
combined_df = combined_df[combined_df.index >= '2015-01-01']
# combined_df["precip"] = precip_data
combined_df = combined_df.dropna()
rmse_insitu = mean_squared_error(combined_df["insitu"], combined_df["pred"], squared=False)
r2_insitu = r2_score(combined_df["insitu"], combined_df["pred"])
bias_insitu = (combined_df["pred"] - combined_df["insitu"]).mean()
unbiased_rmse_insitu = np.sqrt(rmse_insitu**2 - bias_insitu**2)
mae_insitu = mean_absolute_error(combined_df["insitu"], combined_df["pred"])

print(f"RMSE Insitu: {rmse_insitu:.4f}")
print(f"Unbiased RMSE Insitu: {unbiased_rmse_insitu:.4f}")
print(f"Bias Insitu: {bias_insitu:.4f}")
print(f"R² Insitu: {r2_insitu:.4f}")
print(f"MAE Insitu: {mae_insitu:.4f}")

rmse_era5 = mean_squared_error(combined_df["era5_SM"], combined_df["pred"], squared=False)
r2_era5 = r2_score(combined_df["era5_SM"], combined_df["pred"])
bias_ear5 = (combined_df["pred"] - combined_df["era5_SM"]).mean()
unbiased_rmse_era5 = np.sqrt(rmse_era5**2 - bias_ear5**2)
mae_era5 = mean_absolute_error(combined_df["era5_SM"], combined_df["pred"])

print(f"RMSE ERA5: {rmse_era5:.4f}")
print(f"Unbiased RMSE ERA5: {unbiased_rmse_era5:.4f}")
print(f"Bias ERA5: {bias_ear5:.4f}")
print(f"R² ERA5: {r2_era5:.4f}")
print(f"MAE ERA5: {mae_era5:.4f}")


rmse_api = mean_squared_error(combined_df["SM_API"], combined_df["pred"], squared=False)
r2_api = r2_score(combined_df["SM_API"], combined_df["pred"])
bias_api = (combined_df["pred"] - combined_df["SM_API"]).mean()
unbiased_rmse_api = np.sqrt(rmse_api**2 - bias_api**2)
mae_api = mean_absolute_error(combined_df["SM_API"], combined_df["pred"])

print(f"RMSE API: {rmse_api:.4f}")
print(f"Unbiased RMSE API: {unbiased_rmse_api:.4f}")
print(f"Bias API: {bias_api:.4f}")
print(f"R² API: {r2_api:.4f}")
print(f"MAE API: {mae_api:.4f}")

# Plotting insitu vs predictions
# plt.figure(figsize=(12, 5))
# plt.plot(combined_df.index, combined_df["insitu"], label="In-situ SM", linewidth=2)
# plt.plot(combined_df.index, combined_df["pred"], label="Predicted SM", linewidth=2)
# plt.xlabel("Date", fontsize=12)
# plt.ylabel("Soil Moisture", fontsize=12)
# plt.title("Soil Moisture Predictions vs In-situ Measurements (2020)", fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Classification - Loamy Sand
# SM >= FC = 0.16 -> Wet
# 0.5 * TAW + PWP < SM < FC -> 0.5*0.9 + 0.07 <= SM < 0.16 -> 0.115 <= SM < 0.16 -> Moist
# SM <= 0.115 -> Dry

combined_df["SM_pred_class"] = np.where(combined_df["pred"] >= 0.225, "Wet", np.where((combined_df["pred"] >= 0.125) & (combined_df["pred"] < 0.225), "Moist", "Dry"))
combined_df["SM_insitu_class"] = np.where(combined_df["insitu"] >= 0.225, "Wet", np.where((combined_df["insitu"] >= 0.125) & (combined_df["insitu"] < 0.225), "Moist", "Dry"))

accuracy = accuracy_score(combined_df["SM_pred_class"], combined_df["SM_insitu_class"])
classification_report = classification_report(combined_df["SM_insitu_class"], combined_df["SM_pred_class"])
conf_mat = confusion_matrix(combined_df["SM_insitu_class"], combined_df["SM_pred_class"], labels=["Wet", "Moist", "Dry"])
conf_df = pd.DataFrame(conf_mat, index=["True_Wet", "True_Moist", "True_Dry"], columns=["Pred_Wet", "Pred_Moist", "Pred_Dry"])

print("Accuracy: ", accuracy)
print("Classification Report")
print(classification_report)
print("Confusion Matrix")
print(conf_df)

class_colors = {"Wet" : "red", "Moist" : "green", "Dry" : "blue"}
plt.plot(combined_df.index, combined_df["pred"], label="Predicted SM", color="black", linewidth=2)
plt.plot(combined_df.index, combined_df["insitu"], label="In-situ SM", color="gray", linestyle="--")

for cls, color in class_colors.items():
    pred_mask = combined_df["SM_pred_class"] == cls
    insitu_mask = combined_df["SM_insitu_class"] == cls
    plt.scatter(combined_df.index[pred_mask], combined_df["pred"][pred_mask], color=color, s=10, label=f"{cls} (pred)")
    plt.scatter(combined_df.index[insitu_mask], combined_df["insitu"][insitu_mask], edgecolor=color, facecolor='none',
                s=30, label=f"{cls} (insitu)")
plt.title("Time Series of Predicted Soil Moisture with Classification")
plt.xlabel("Date")
plt.ylabel("Soil Moisture")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

axes[0].plot(combined_df.index, combined_df["insitu"], label="In-situ SM", linewidth=2)
axes[0].plot(combined_df.index, combined_df["pred"], label="Predicted SM", linewidth=2)
axes[0].plot(combined_df.index, combined_df["era5_SM"], label="ERA5 SM", linewidth=2)
axes[0].plot(combined_df.index, combined_df["SM_API"], label="API SM", linewidth=2)
# axes[0].plot(combined_df.index, combined_df["ndvi"], label="NDVI", linewidth=2)
axes[0].set_ylabel("Soil Moisture", fontsize=12)
axes[0].set_title("Soil Moisture Predictions vs In-situ Measurements (2020)", fontsize=14)
axes[0].legend()
axes[0].grid(True)

axes[1].plot(combined_df.index, combined_df["precip"], label="Precipitation", linewidth=2)
axes[1].set_xlabel("Date", fontsize=12)
axes[1].set_ylabel("Precipitation (mm)", fontsize=12)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()


# Writing the predictions to map, where each band corresponds to a date
# output_path = "../../Data/SMAP/SMAP_downscaled_appraoch1_smap_test.tif"
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
#         dst.set_band_description(i + 1, common_dates_dt[i].strftime("%Y_%m_%d") + "_SM")

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
