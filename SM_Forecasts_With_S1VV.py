import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import random
from pyproj import Transformer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)


def read_tif(filename):
    with rasterio.open(filename) as f:
        data = f.read()
        return data, f.transform


def get_pixels_values(lat, lon, transform):
    row, col = ~transform*(lon, lat)
    return int(row), int(col)


def get_pixels_values_s1(lat, lon, transform):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32736", always_xy=True)
    x, y = transformer.transform(lon, lat)
    col, row = ~transform*(x, y)
    return int(row), int(col)

# Chemba
pt_lat, pt_lon = -17.33091, 34.96124

s1_vv_train = "../../Data/Sentinel-1/S1_VV_2016_2022_sorted.tif"
s1_vv_test = "../../Data/Sentinel-1/S1_VV_2023_2024_sorted.tif"

s1_vh_train = "../../Data/Sentinel-1/S1_VH_2017_2022_sorted.tif"
s1_vh_test = "../../Data/Sentinel-1/S1_VH_2023_2024_sorted.tif"

era5_precip_train = "../../Data/ERA5/ERA5_2015_2022_Precip_Daily.tif"
era5_precip_test = "../../Data/ERA5/ERA5_2023_2024_Precip_Daily.tif"
era5_temp_train = "../../Data/ERA5/ERA5_2015_2022_Temp_Daily.tif"
era5_temp_test = "../../Data/ERA5/ERA5_2023_2024_Temp_Daily.tif"
era5_rh_train = "../../Data/ERA5/ERA5_2015_2022_RH_Daily.tif"
era5_rh_test = "../../Data/ERA5/ERA5_2023_2024_RH_Daily.tif"
era5_windspeed_train = "../../Data/ERA5/ERA5_2015_2022_WindSpeed_Daily.tif"
era5_windspeed_test = "../../Data/ERA5/ERA5_2023_2024_WindSpeed_Daily.tif"
era5_radiation_train = "../../Data/ERA5/ERA5_2015_2022_Radiation_Daily.tif"
era5_radiation_test = "../../Data/ERA5/ERA5_2023_2024_Radiation_Daily.tif"
era5_et_train = "../../Data/ERA5/ERA5_2015_2022_ET_Daily.tif"
era5_et_test = "../../Data/ERA5/ERA5_2023_2024_ET_Daily.tif"
era5_soiltemp_train = "../../Data/ERA5/ERA5_2015_2022_SoilTemp_Daily.tif"
era5_soiltemp_test = "../../Data/ERA5/ERA5_2023_2024_SoilTemp_Daily.tif"

smap_sm_train = "../../Data/SMAP/SMAP_2016_2022_SM_Daily.tif"
smap_sm_test = "../../Data/SMAP/SMAP_2023_2024_SM_Daily.tif"

########################################################################################################################
era5_precip_train_data, era5_precip_train_transform = read_tif(era5_precip_train)
era5_precip_test_data, era5_precip_test_transform = read_tif(era5_precip_test)

era5_temp_train_data, era5_temp_train_transform = read_tif(era5_temp_train)
era5_temp_test_data, era5_temp_test_transform = read_tif(era5_temp_test)

era5_rh_train_data, era5_rh_train_transform = read_tif(era5_rh_train)
era5_rh_test_data, era5_rh_test_transform = read_tif(era5_rh_test)

era5_windspeed_train_data, era5_windspeed_train_transform = read_tif(era5_windspeed_train)
era5_windspeed_test_data, era5_windspeed_test_transform = read_tif(era5_windspeed_test)

era5_radiation_train_data, era5_radiation_train_transform = read_tif(era5_radiation_train)
era5_radiation_test_data, era5_radiation_test_transform = read_tif(era5_radiation_test)

era5_et_train_data, era5_et_train_transform = read_tif(era5_et_train)
era5_et_test_data, era5_et_test_transform = read_tif(era5_et_test)

era5_soiltemp_train_data, era5_soiltemp_train_transform = read_tif(era5_soiltemp_train)
era5_soiltemp_test_data, era5_soiltemp_test_transform = read_tif(era5_soiltemp_test)

smap_sm_train_data, smap_sm_train_transform = read_tif(smap_sm_train)
smap_sm_test_data, smap_sm_test_transform = read_tif(smap_sm_test)

s1_vv_train_data, s1_vv_train_transform = read_tif(s1_vv_train)
s1_vv_test_data, s1_vv_test_transform = read_tif(s1_vv_test)

s1_vh_train_data, s1_vh_train_transform = read_tif(s1_vh_train)
s1_vh_test_data, s1_vh_test_transform = read_tif(s1_vh_test)

era5_row, era5_col = get_pixels_values(pt_lat, pt_lon, era5_precip_train_transform)
s1_row, s1_col = get_pixels_values_s1(pt_lat, pt_lon, s1_vv_train_transform)

########################################################################################################################
# Filtering Train Data

era5_train_precip = era5_precip_train_data[:, era5_row, era5_col]
era5_train_temp = era5_temp_train_data[:, era5_row, era5_col]
era5_train_rh = era5_rh_train_data[:, era5_row, era5_col]
era5_train_windspeed = era5_windspeed_train_data[:, era5_row, era5_col]
era5_train_radiation = era5_radiation_train_data[:, era5_row, era5_col]
era5_train_et = era5_et_train_data[:, era5_row, era5_col]
era5_train_soiltemp = era5_soiltemp_train_data[:, era5_row, era5_col]
smap_train_sm = smap_sm_train_data[:, era5_row, era5_col]
s1_train_vv = s1_vv_train_data[:, s1_row, s1_col]
s1_train_vh = s1_vh_train_data[:, s1_row, s1_col]

########################################################################################################################
# Filtering Test Data

era5_test_precip = era5_precip_test_data[:, era5_row, era5_col]
era5_test_temp = era5_temp_test_data[:, era5_row, era5_col]
era5_test_rh = era5_rh_test_data[:, era5_row, era5_col]
era5_test_windspeed = era5_windspeed_test_data[:, era5_row, era5_col]
era5_test_radiation = era5_radiation_test_data[:, era5_row, era5_col]
era5_test_et = era5_et_test_data[:, era5_row, era5_col]
era5_test_soiltemp = era5_soiltemp_test_data[:, era5_row, era5_col]
smap_test_sm = smap_sm_test_data[:, era5_row, era5_col]
s1_test_vv = s1_vv_test_data[:, s1_row, s1_col]
s1_test_vh = s1_vh_test_data[:, s1_row, s1_col]

train_dates = pd.date_range(start="2015-01-01", end="2022-12-31", freq="D")

train_df = pd.DataFrame({
    "Date" : train_dates,
    "Precip" : era5_train_precip,
    "Temp" : era5_train_temp,
    "RH" : era5_train_rh,
    "WindSpeed" : era5_train_windspeed,
    "Radiation" : era5_train_radiation,
    "ET" : era5_train_et,
    "SoilTemp" : era5_train_soiltemp
})

train_df = train_df[train_df["Date"] >= '2016-01-01']
train_df["SM"] = smap_train_sm

train_df = train_df[train_df["Date"] >= '2017-01-01']

with rasterio.open(s1_vv_train) as src:
    data = src.read()
    original_band_names = src.descriptions
    transform = src.transform
    profile = src.profile
    s1_crs = src.crs

s1_train_dates = pd.to_datetime(original_band_names)


vv_series = pd.Series(s1_train_vv, index=s1_train_dates)
vv_daily = vv_series.reindex(train_df["Date"])
train_df["VV_lag"] = vv_daily.fillna(0).values
train_df["VV_missing_flag"] = vv_daily.isna().astype(int).values
# train_df["DoY"] = train_df["Date"].dt.dayofyear

with rasterio.open(s1_vh_train) as src:
    data_vh = src.read()
    original_band_names_vh = src.descriptions
    transform_vh = src.transform
    profile_vh = src.profile
    s1_crs_vh = src.crs

s1_vh_train_dates = pd.to_datetime(original_band_names_vh)
vh_series = pd.Series(s1_train_vh, index=s1_vh_train_dates)
vh_daily = vh_series.reindex(train_df["Date"])
train_df["VH_lag"] = vh_daily.fillna(0).values
train_df["VH_missing_flag"] = vh_daily.isna().astype(int).values


print(train_df.head(20))
print(train_df.describe())
pd.set_option('display.max_columns', None)
print(train_df.drop(columns=["Date"]).corr())
pd.reset_option('display.max_columns')


test_dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")

test_df = pd.DataFrame({
    "Date" : test_dates,
    "Precip" : era5_test_precip,
    "Temp" : era5_test_temp,
    "RH" : era5_test_rh,
    "WindSpeed" : era5_test_windspeed,
    "Radiation" : era5_test_radiation,
    "ET" : era5_test_et,
    "SoilTemp" : era5_test_soiltemp,
    "SM" : smap_test_sm
})


with rasterio.open(s1_vv_test) as src:
    data1 = src.read()
    original_band_names1 = src.descriptions
    transform1 = src.transform
    profile1 = src.profile
    s1_crs1 = src.crs

s1_test_dates = pd.to_datetime(original_band_names1)
# print(s1_test_dates)

# test_df = test_df[test_df["Date"] >= '2023-01-04']

vv_series = pd.Series(s1_test_vv, index=s1_test_dates)
vv_daily = vv_series.reindex(test_df["Date"])
test_df["VV_lag"] = vv_daily.fillna(0).values
test_df["VV_missing_flag"] = vv_daily.isna().astype(int).values
# test_df["DoY"] = test_df["Date"].dt.dayofyear

with rasterio.open(s1_vh_test) as src:
    data1_vh = src.read()
    original_band_names1_vh = src.descriptions
    transform1_vh = src.transform
    profile1_vh = src.profile
    s1_crs1_vh = src.crs

s1_vh_test_dates = pd.to_datetime(original_band_names1_vh)
# print(s1_test_dates)

# test_df = test_df[test_df["Date"] >= '2023-01-04']

vh_series = pd.Series(s1_test_vh, index=s1_vh_test_dates)
vh_daily = vv_series.reindex(test_df["Date"])
test_df["VH_lag"] = vh_daily.fillna(0).values
test_df["VH_missing_flag"] = vh_daily.isna().astype(int).values


print(test_df.head(20))
print(test_df.describe())


train_features = train_df.drop(columns=["Date", "SM"])
train_target = train_df["SM"]

test_features = test_df.drop(columns=["Date", "SM"])
test_target = test_df["SM"]

features_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features_scaled = features_scaler.fit_transform(train_features)
target_scaled = target_scaler.fit_transform(train_target.values.reshape(-1, 1))

test_features_scaled = features_scaler.transform(test_features)
test_target_scaled = target_scaler.transform(test_target.values.reshape(-1, 1))


def create_seq(x, y, time_steps=20, forecasts=7):
    xs, ys = [], []
    for i in range(len(x) - time_steps - forecasts + 1):
        xs.append(x[i:i+time_steps])
        ys.append(y[i+time_steps:i+time_steps+forecasts].flatten())
    return np.array(xs), np.array(ys)


x_train, y_train = create_seq(features_scaled, target_scaled, time_steps=60, forecasts=7)
x_test, y_test = create_seq(test_features_scaled, test_target_scaled, time_steps=60, forecasts=7)


# model = tf.keras.Sequential([
#     tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='same', input_shape=(x_train.shape[1], x_train.shape[2])),
#     tf.keras.layers.LSTM(128, return_sequences=True),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.LSTM(128, return_sequences=True),
#     tf.keras.layers.LSTM(32),
#     tf.keras.layers.Dense(7)
# ])
#
# model.compile(optimizer='adam', loss='mse')
# model.summary()

input_seq = tf.keras.Input(shape=(x_train.shape[1], x_train.shape[2]))

cnn = tf.keras.layers.Conv1D(64, kernel_size=7, activation='relu')(input_seq)
cnn = tf.keras.layers.Conv1D(64, kernel_size=7, activation='relu')(cnn)
cnn = tf.keras.layers.Dropout(0.3)(cnn)
cnn = tf.keras.layers.Flatten()(cnn)
cnn = tf.keras.layers.Dense(64, activation='relu')(cnn)

lstm = tf.keras.layers.LSTM(128, return_sequences=True)(input_seq)
lstm = tf.keras.layers.Dropout(0.3)(lstm)
lstm = tf.keras.layers.LSTM(128)(lstm)
lstm = tf.keras.layers.Dense(64)(lstm)

merged = tf.keras.layers.Concatenate()([cnn, lstm])
merged = tf.keras.layers.Dense(10, activation='relu')(merged)
output = tf.keras.layers.Dense(7, activation='relu')(merged)

model = tf.keras.models.Model(inputs=input_seq, outputs=output)

model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(x_train, y_train, epochs=40, batch_size=32, validation_split=0.3)


def monte_carlo_predict(model, X, num_samples=100):
    predictions = np.zeros((num_samples, X.shape[0], 7))
    for i in range(num_samples):
        predictions[i] = model(X, training=True)
    return predictions


last_20_days = features_scaled[-60:]
last_20_days = last_20_days.reshape((1, 60, features_scaled.shape[1]))

predictions_mc = monte_carlo_predict(model, last_20_days, num_samples=200)

mean_preds = np.mean(predictions_mc, axis=0)
lower_bound = np.percentile(predictions_mc, 2.5, axis=0)
upper_bound = np.percentile(predictions_mc, 97.5, axis=0)

mean_preds = target_scaler.inverse_transform(mean_preds)[0]
lower_bound = target_scaler.inverse_transform(lower_bound)[0]
upper_bound = target_scaler.inverse_transform(upper_bound)[0]

print(f"Predicted SM (Mean): {mean_preds}")
print(f"Lower Bound (2.5%): {lower_bound}")
print(f"Upper Bound (97.5%): {upper_bound}")


predictions_mc_test = monte_carlo_predict(model, x_test, num_samples=200)

mean_preds_test = np.mean(predictions_mc_test, axis=0)
lower_bound_test = np.percentile(predictions_mc_test, 2.5, axis=0)
upper_bound_test = np.percentile(predictions_mc_test, 97.5, axis=0)

mean_preds_test = target_scaler.inverse_transform(mean_preds_test)
lower_bound_test = target_scaler.inverse_transform(lower_bound_test)
upper_bound_test = target_scaler.inverse_transform(upper_bound_test)

y_true = target_scaler.inverse_transform(y_test)

for i in range(7):
    r2 = r2_score(y_true[:, i], mean_preds_test[:, i])
    rmse = np.sqrt(mean_squared_error(y_true[:, i], mean_preds_test[:, i]))
    print(f"Day {i+1} - R²: {r2:.4f}, RMSE: {rmse:.4f}")


fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex="all")

for i in range(3):
    axs[i].plot(y_true[:, i], label="Actual", color="black")
    axs[i].plot(mean_preds_test[:, i], label="Predicted (Mean)", color="green")
    axs[i].fill_between(
        np.arange(len(mean_preds_test)),
        lower_bound_test[:, i],
        upper_bound_test[:, i],
        color="gray",
        alpha=0.5,
        label="95% CI"
    )
    axs[i].set_title(f"SM Prediction - Day {i+1}")
    axs[i].set_ylabel("SM")
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Time step")
plt.tight_layout()
plt.show()

########################################################################################################################

# with rasterio.open(s1_vh_test) as src:
#     data = src.read()
#     original_band_names = src.descriptions
#     transform = src.transform
#     profile = src.profile
#     s1_crs = src.crs
#
# # s1_train_dates = pd.to_datetime(original_band_names)
#
# print(s1_crs)
#
# dates = [name.split('_')[-1] for name in original_band_names]
# dates = pd.to_datetime(dates, errors='coerce')
#
# # print(dates)
#
# valid_indices = [i for i, d in enumerate(dates) if pd.notnull(d)]
# dates = [dates[i] for i in valid_indices]
# data = data[valid_indices, :, :]
#
# sorted_indices = np.argsort(dates)
# dates_sorted = [dates[i] for i in sorted_indices]
# data_sorted = data[sorted_indices, :, :]
# print(str(dates_sorted[1].strftime("%Y-%m-%d")))
# # data_sorted = data_sorted[1:, :, :]
# # dates_sorted = dates_sorted[1:]
#
#
# profile.update({
#     "count": data_sorted.shape[0],
#     "dtype": str(data_sorted.dtype),
#     "driver": "GTiff"
# })
#
#
# output_path = "../../Data/Sentinel-1/S1_VH_2023_2024_sorted.tif"
#
#
# with rasterio.open(output_path, "w", **profile) as dst:
#     for i in range(data_sorted.shape[0]):
#         dst.write(data_sorted[i, :, :], i + 1)
#         dst.set_band_description(i + 1, dates_sorted[i].strftime("%Y-%m-%d"))
#
# print("Saved sorted GeoTIFF to:", output_path)


########################################################################################################################
# Aggregation

# precip_agg_train = []
# temp_agg_train = []
# rh_agg_train = []
# windspeed_agg_train = []
# radiation_agg_train = []
# et_agg_train = []
# soiltemp_agg_train = []
# sm_agg_train = []
#
# for i in range(len(s1_train_dates)):
#     if i == 0:
#         start_date = train_df["Date"].min()
#     else:
#         start_date = s1_train_dates[i - 1] + pd.Timedelta(days=1)
#
#     end_date = s1_train_dates[i]
#
#     mask = (train_df["Date"] >= start_date) & (train_df["Date"] <= end_date)
#     precip_agg = train_df.loc[mask, "Precip"].sum()
#     temp_agg = train_df.loc[mask, "Temp"].mean()
#     rh_agg = train_df.loc[mask, "RH"].mean()
#     windspeed_agg = train_df.loc[mask, "WindSpeed"].mean()
#     radiation_agg = train_df.loc[mask, "Radiation"].mean()
#     et_agg = train_df.loc[mask, "ET"].mean()
#     soiltemp_agg = train_df.loc[mask, "SoilTemp"].mean()
#     sm_agg = train_df.loc[mask, "SM"].mean()
#
#     precip_agg_train.append(precip_agg)
#     temp_agg_train.append(temp_agg)
#     rh_agg_train.append(rh_agg)
#     windspeed_agg_train.append(windspeed_agg)
#     radiation_agg_train.append(radiation_agg)
#     et_agg_train.append(et_agg)
#     soiltemp_agg_train.append(soiltemp_agg)
#     sm_agg_train.append(sm_agg)
#
# train_agg_df = pd.DataFrame({
#     "Date": s1_train_dates,
#     "VV": s1_train_vv,
#     "Precip": precip_agg_train,
#     "Temp" : temp_agg_train,
#     "RH" : rh_agg_train,
#     "WindSpeed" : windspeed_agg_train,
#     "Radiation" : radiation_agg_train,
#     "ET" : et_agg_train,
#     "SoilTemp" : soiltemp_agg_train,
#     "SM" : sm_agg_train
# })

# print(train_agg_df.head())


# precip_agg_test = []
# temp_agg_test = []
# rh_agg_test = []
# windspeed_agg_test = []
# radiation_agg_test = []
# et_agg_test = []
# soiltemp_agg_test = []
# sm_agg_test = []
#
# for i in range(len(s1_test_dates)):
#     if i == 0:
#         start_date = test_df["Date"].min()
#     else:
#         start_date = s1_test_dates[i - 1] + pd.Timedelta(days=1)
#
#     end_date = s1_test_dates[i]
#
#     mask = (test_df["Date"] >= start_date) & (test_df["Date"] <= end_date)
#     precip_agg_1 = test_df.loc[mask, "Precip"].sum()
#     temp_agg_1 = test_df.loc[mask, "Temp"].mean()
#     rh_agg_1 = test_df.loc[mask, "RH"].mean()
#     windspeed_agg_1 = test_df.loc[mask, "WindSpeed"].mean()
#     radiation_agg_1 = test_df.loc[mask, "Radiation"].mean()
#     et_agg_1 = test_df.loc[mask, "ET"].mean()
#     soiltemp_agg_1 = test_df.loc[mask, "SoilTemp"].mean()
#     sm_agg_1 = test_df.loc[mask, "SM"].mean()
#
#     precip_agg_test.append(precip_agg_1)
#     temp_agg_test.append(temp_agg_1)
#     rh_agg_test.append(rh_agg_1)
#     windspeed_agg_test.append(windspeed_agg_1)
#     radiation_agg_test.append(radiation_agg_1)
#     et_agg_test.append(et_agg_1)
#     soiltemp_agg_test.append(soiltemp_agg_1)
#     sm_agg_test.append(sm_agg_1)
#
#
# test_agg_df = pd.DataFrame({
#     "Date": s1_test_dates,
#     "VV": s1_test_vv,
#     "Precip": precip_agg_test,
#     "Temp" : temp_agg_test,
#     "RH" : rh_agg_test,
#     "WindSpeed" : windspeed_agg_test,
#     "Radiation" : radiation_agg_test,
#     "ET" : et_agg_test,
#     "SoilTemp" : soiltemp_agg_test,
#     "SM" : sm_agg_test
# })
