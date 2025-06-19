import rasterio
from rasterio.transform import rowcol, xy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pyproj import Transformer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)


def read_tif(path):
    """
    Reads a tif file to return its data as a NumPy array.

    Parameters
    ----------
    path : str
        The location of the tif file.

    Returns
    -------
    data : np.ndarray
        A NumPy array containing the raster data with shape (bands, height, width).
    bands : tuple or None
        A tuple of band descriptions.
    """
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        bands = src.descriptions
    return data, bands


def map_10km_to_1km(r1km, c1km, sm_transform, weather_transform):
    """
    This function is used as a helper function to map the 1km soil moisture cell (row, col) to the corresponding 10km
    weather pixels.

    Parameters
    ----------
    r1km : int
        row index of the pixel in 1km soil moisture grid.
    c1km : int
        column index of the pixel in 1km soil moisture grid.
    sm_transform : Affine
        Affine transformation for the 1km soil moisture raster.
    weather_transform : Affine
        Affine transformation for the 10km weather raster.

    Returns
    -------
    r10km : int
        row index of the pixel in 10km weather grid.
    c10km : int
        column index of the pixel in 10km weather grid.
    """
    x, y = xy(sm_transform, r1km, c1km)
    r10km, c10km = rowcol(weather_transform, x, y)
    return r10km, c10km


def create_sequences(sm_transform, sm_data, weather_transform, weather_data, past_days=30, forecast_days=7):
    """
    Generates data sequences for deep learning model using soil moisture and weather information.

    This function extracts time series sequences for each pixel in the 1km soil moisture dataset, along with
    corresponding 10km weather data, and constructs input (past days)-output (forecast days) pairs.

    Parameters
    ----------
    sm_transform : Affine
        Affine transformation for the 1km soil moisture raster.
    sm_data : np.ndarray
        Soil moisture data array of shape (Time, H_sm, W_sm).
    weather_transform : Affine
        Affine transformation for the 10km weather raster.
    weather_data : np.ndarray
        Weather data array of shape (Time, H_weather, W_weather, num_features).
    past_days : int
        Number of past days to create one input sequence (default=30).
    forecast_days : int
        Number of future days to forecast ahead (default=7).

    Returns
    -------
    sequences_x : np.ndarray
        Array of shape (num_sequences, past_days, num_features + 1).
    sequences_y : np.ndarray
        Array of shape (num_sequences, forecast_days).
    pixel_indices : list of tuple
        List of tuple (row, col) indicating the pixel location in the 1km soil moisture grid.
    """
    T, H, W= sm_data.shape
    sequences_x, sequences_y, pixel_indices = [], [], []

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
            for t in range(T - past_days - forecast_days + 1):
                weather_x = pixel_weather[t:t+past_days, :]
                sm_x = pixel_sm[t:t + past_days].reshape(-1, 1)
                X = np.hstack([weather_x, sm_x])
                y = pixel_sm[t+past_days:t+past_days+forecast_days]
                sequences_x.append(X)
                sequences_y.append(y)
                pixel_indices.append((i, j))
    return np.array(sequences_x), np.array(sequences_y), pixel_indices


def create_seq(y, time_steps=30, forecasts=7):
    """
    Generate forecast sequences for insitu data.

    Parameters
    ----------
    y : array like
        Univariate time series data.
    time_steps : int
        Number of past days in the input sequences (default=30).
    forecasts : int
        Number of future days to forecast (default=7).

    Return
    ------
    np.ndarray
        A 2D NumPy array of shape (num_sequences, forecasts).
    """
    ys = []
    y = np.array(y)
    for i in range(len(y) - time_steps - forecasts + 1):
        ys.append(y[i+time_steps:i+time_steps+forecasts].flatten())
    return np.array(ys)


def get_valid_dates(bands, suffix):
    """
    Extract dates from a raster, given each band has descriptions of the format "YYYY_MM_DD_suffix".

    Parameters
    ----------
    bands : tuple
        A tuple of band descriptions.
    suffix : str
        String of the variable (eg: "SM", "Precip").

    Return
    ------
    set of str
        Set of unique dates (as strings of format "YYY_MM_DD").
    """
    return set(
        b.replace(f"_{suffix}", "") for b in bands if b.endswith(f"_{suffix}")
    )


def filter_by_dates(data, bands, suffix, common_dates):
    """
    Filter the raster data to align all the features.

    Parameters
    ----------
    data : np.ndarray
        NumPy array containing the raster data with shape (bands, height, width).
    bands : tuple
        A tuple of band descriptions.
    suffix : str
        String of the variable (eg: "SM", "Precip").
    common_dates : list
        List of unique and common dates (as strings of format "YYY_MM_DD") between all the features.

    Returns
    -------
    data : np.ndarray
        NumPy array containing the raster data with shape (bands, height, width), filtered to common dates.
    bands : tuple
        A tuple of band descriptions, filtered to common dates.
    """
    filtered_indices = []
    filtered_band_names = []

    for i, b in enumerate(bands):
        if b.endswith(f"_{suffix}"):
            date = b.replace(f"_{suffix}", "")
            if date in common_dates:
                filtered_indices.append(i)
                filtered_band_names.append(b)
    return data[filtered_indices], filtered_band_names


# List of paths for weather variables
era5_train_paths = [
    "../../Data/ERA5_NL/ERA5_2015_2020_Precip_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2015_2020_Temp_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2015_2020_RH_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2015_2020_WindSpeed_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2015_2020_Radiation_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2015_2020_ET_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2015_2020_SoilTemp_NL_Daily.tif"
]

# Paths for SMAP 10km SM product, not used in the code anymore.
smap_sm_am_train = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_AM_NL_Daily.tif"
smap_sm_pm_train = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_PM_NL_Daily.tif"

# Path for GSSM 1km SM product, not used in the code anymore.
gssm_sm_train = "../../Data/GSSM/GSSM_2016_2020_SM_NL_Daily_1km.tif"

# Path for SMAP 1km downscaled SM product.
smap_sm_downscaled_path = "../../Data/SMAP/SMAP_downscaled_appraoch1_smap_test.tif"

# Path for insitu measurements, csv format and the corresponding lat, lon information.
sm_test_path = "../../Data/dataverse_files/1_station_measurements/2_calibrated/ITCSM_10_cd.csv"
lat, lon = 52.2, 6.65944


# Read SM and weather data.
sm_data, sm_bands = read_tif(smap_sm_downscaled_path)
era5_precip, era5_precip_bands = read_tif(era5_train_paths[0])
era5_temp, era5_temp_bands = read_tif(era5_train_paths[1])
era5_rh, era5_rh_bands = read_tif(era5_train_paths[2])
era5_wind, era5_wind_bands = read_tif(era5_train_paths[3])
era5_rad, era5_rad_bands = read_tif(era5_train_paths[4])
era5_et, era5_et_bands = read_tif(era5_train_paths[5])
era5_soiltemp, era5_soiltemp_bands = read_tif(era5_train_paths[6])


# Get the valid dates for each feature.
sm_dates = get_valid_dates(sm_bands, "SM")
era5_precip_dates = get_valid_dates(era5_precip_bands, "Precip")
era5_temp_dates = get_valid_dates(era5_temp_bands, "Temp")
era5_rh_dates = get_valid_dates(era5_rh_bands, "RH")
era5_wind_dates = get_valid_dates(era5_wind_bands, "Wind")
era5_rad_dates = get_valid_dates(era5_rad_bands, "Radiation")
era5_et_dates = get_valid_dates(era5_et_bands, "ET")
era5_soiltemp_dates = get_valid_dates(era5_soiltemp_bands, "SoilTemp")

# Get the common dates between all the features.
common_dates = sorted(sm_dates & era5_precip_dates & era5_temp_dates & era5_rh_dates & era5_wind_dates & era5_rad_dates
                      & era5_et_dates & era5_soiltemp_dates)

# Filter all the features to common dates.
era5_precip, era5_precip_bands_filt = filter_by_dates(era5_precip, era5_precip_bands, "Precip", common_dates)
era5_temp, era5_temp_bands_filt = filter_by_dates(era5_temp, era5_temp_bands, "Temp", common_dates)
era5_rh, era5_rh_bands_filt = filter_by_dates(era5_rh, era5_rh_bands, "RH", common_dates)
era5_wind, era5_wind_bands_filt = filter_by_dates(era5_wind, era5_wind_bands, "Wind", common_dates)
era5_rad, era5_rad_bands_filt = filter_by_dates(era5_rad, era5_rad_bands, "Radiation", common_dates)
era5_et, era5_et_bands_filt = filter_by_dates(era5_et, era5_et_bands, "ET", common_dates)
era5_soiltemp, era5_soiltemp_bands_filt = filter_by_dates(era5_soiltemp, era5_soiltemp_bands, "SoilTemp", common_dates)

# sm_new_dates = [pd.to_datetime(b.replace('_', '-')) for b in common_dates]
# mask = [
#     (date >= pd.to_datetime("2017-01-01")) and (date <= pd.to_datetime("2019-05-31"))
#     for date in sm_new_dates
# ]

# Stack the weather features, with shape (Time, H, W, num_features).
weather_data_stack = np.stack([era5_precip, era5_temp, era5_rh, era5_wind, era5_rad, era5_et, era5_soiltemp], axis=-1)

# weather_train = weather_data_stack[mask]
# Split train and test weather data, currently using the last year as test dataset and remaining as training dataset.
weather_train = weather_data_stack[:-366]
weather_test = weather_data_stack[-366:]

# sm_train = sm_data[mask]
# Split train and test SM data, currently using the last year as test dataset and remaining as training dataset.
sm_train = sm_data[:-366]
sm_test = sm_data[-366:]

# Get the Affine transformation of SM and weather data, needed for creating training sequences.
with rasterio.open(smap_sm_downscaled_path) as src_sm:
    sm_transform = src_sm.transform

with rasterio.open(era5_train_paths[0]) as src_weather:
    weather_transform = src_weather.transform

# Initialize scalars for input features and target.
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Scale the weather train and test datasets. Before scaling, reshape the data to 2D array of shape (T*H*W, F).
T, H, W, F = weather_train.shape
weather_flat_train = weather_train.reshape(T*H*W, F)
weather_flat_scaled = feature_scaler.fit_transform(weather_flat_train)
weather_train_scaled = weather_flat_scaled.reshape(T, H, W, F)

T2, H2, W2, F2 = weather_test.shape
weather_flat_test = weather_test.reshape(T2*H2*W2, F2)
weather_flat_test_scaled = feature_scaler.transform(weather_flat_test)
weather_test_scaled = weather_flat_test_scaled.reshape(T2, H2, W2, F2)

# Scale the SM train and test datasets. Before scaling, reshape the data to 2D array of shape (T*H*W, 1).
sm_flat = sm_train.reshape(-1, 1)
sm_train_scaled = target_scaler.fit_transform(sm_flat).reshape(sm_train.shape)

sm_flat_test = sm_test.reshape(-1, 1)
sm_test_scaled = target_scaler.transform(sm_flat_test).reshape(sm_test.shape)

# Define past and future steps parameters.
past_days = 30
forecast_days = 7

# Create training and testing sequences for the model.
X_train, y_train, pixel_indices_train = create_sequences(sm_transform, sm_train_scaled, weather_transform,
                                                         weather_train_scaled, past_days, forecast_days)
X_test, y_test, pixel_indices_test = create_sequences(sm_transform, sm_test_scaled, weather_transform,
                                                      weather_test_scaled, past_days, forecast_days)

# Build a sequential architecture.
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(past_days, X_train.shape[-1])),
#     # tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
#     tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
#     # tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
#     tf.keras.layers.LSTM(64, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(forecast_days)
# ])
#
# model.compile(optimizer='adam', loss='mse')
# model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2)
#
# # Save the model
# model_name = f"sm_smap_weather_downscaled_conv_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
# model.save(f"../../Models/{model_name}")

model = tf.keras.models.load_model("../../Models/sm_smap_weather_downscaled_conv_lstm_20250618_120259.h5")

# Predict on test dataset and inverse transform the predictions to original scale.
y_preds = model.predict(X_test)
y_preds_inv = target_scaler.inverse_transform(y_preds.reshape(-1, 1)).reshape(y_preds.shape)
# y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# Read the insitu sm data, csv. Skipping unnecessary rows.
headers = pd.read_csv(sm_test_path, skiprows=18, nrows=0).columns.tolist()
sm_test = pd.read_csv(sm_test_path, skiprows=20, parse_dates=["Date time"], names=headers)

sm_test["Date time"] = pd.to_datetime(sm_test["Date time"], format='%d-%m-%Y %H:%M', errors='coerce')
sm_test = sm_test[sm_test["Date time"] >= '2020-01-01']
sm_test = sm_test.set_index("Date time")
sm_test = sm_test.resample("D").mean()
sm_test_insitu = create_seq(sm_test[" 5 cm SM"], 30, 7)

# Get the pixel index in the SM grid, corresponding to the location (lat, lon) of insitu data.
with rasterio.open(smap_sm_downscaled_path) as src:
    transform = src.transform
    crs = src.crs

transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
x, y = transformer.transform(lon, lat)
row, col = rowcol(transform, x, y)
target_pixel = (row, col)

# Get the prediction indices for the insitu location.
matching_indices = [idx for idx, pix in enumerate(pixel_indices_test) if pix == target_pixel]

# Compute validation metrics for each day.
for day in range(forecast_days):
    true_series = sm_test_insitu[:, day]
    # true_series = y_test_inv[matching_indices, day]
    pred_series = y_preds_inv[matching_indices, day]
    rmse = np.sqrt(mean_squared_error(true_series, pred_series))
    r2 = r2_score(true_series, pred_series)
    print(f"Pixel {target_pixel}, Day {day + 1} — RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Plot the 3 days' predictions vs ture values.
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


########################################################################################################################
# weather_train = stack_weather(era5_train_paths)[60:1461]
# weather_test = stack_weather(era5_train_paths)[1461:1461+366]
#
# sm_train = read_tif(gssm_sm_train)[51:1461-9]
# sm_test = read_tif(gssm_sm_train)[1461-9:1461-9+366]
#
# smap_sm_am_data = read_tif(smap_sm_am_train)
# smap_sm_pm_data = read_tif(smap_sm_pm_train)
#
# smap_sm_avg_data = np.where(
#     np.isnan(smap_sm_am_data) & np.isnan(smap_sm_pm_data),
#     np.nan,
#     np.nanmean(np.stack([smap_sm_am_data, smap_sm_pm_data]), axis=0)
# )
#
# sm_train = smap_sm_avg_data[60:1461]
# sm_test = smap_sm_avg_data[1461:1461+366]
