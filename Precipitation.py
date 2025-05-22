import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import Ridge
from sklearn.utils import resample
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import random
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


era5 = "../../Data/ERA5/ERA5_2021_2022_Precip_Daily.tif"
imerg = "../../Data/IMERG/IMERG_2021_2022_Precip_Daily.tif"
chirps = "../../Data/CHIRPS/CHIRPS_2021_2022_Precip_Daily.tif"
gsmap = "../../Data/GSMaP/GSMaP_2021_2022_Precip_Daily.tif"


def read_tif(filename):
    with rasterio.open(filename) as f:
        data = f.read()
        dates = [str(band) for band in f.descriptions]
        return data, dates, f.transform, f.crs, f.bounds


era5_data, era5_dates, era5_transform, era5_crs, era5_bounds = read_tif(era5)
imerg_data, imerg_dates, imerg_transform, imerg_crs, imerg_bounds = read_tif(imerg)
chirps_data, chirps_dates, chirps_transform, chirps_crs, chirps_bounds = read_tif(chirps)
gsmap_data, gsmap_dates, gsmap_transform, gsmap_crs, gsmap_bounds = read_tif(gsmap)


def get_pixels_values(lat, lon, transform):
    row, col = ~transform*(lon, lat)
    return int(row), int(col)

# Chemba
pt_lat, pt_lon = -17.331524, 34.954147

# Transmara, Kenya
# pt_lat, pt_lon = -1.023509, 34.740671

era5_row, era5_col = get_pixels_values(pt_lat, pt_lon, era5_transform)
era5_precip = era5_data[:, era5_row, era5_col]


imerg_row, imerg_col = get_pixels_values(pt_lat, pt_lon, imerg_transform)
imerg_precip = imerg_data[:, imerg_row, imerg_col]


chirps_row, chirps_col = get_pixels_values(pt_lat, pt_lon, chirps_transform)
chirps_precip = chirps_data[:, chirps_row, chirps_col]


gsmap_row, gsmap_col = get_pixels_values(pt_lat, pt_lon, gsmap_transform)
gsmap_precip = gsmap_data[:, gsmap_row, gsmap_col]


dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")

precip_df = pd.DataFrame({
    "Date" : dates,
    "ERA5" : era5_precip,
    "IMERG" : imerg_precip,
    "CHIRPS" : chirps_precip,
    "GSMaP" : gsmap_precip,
})


df = pd.read_csv("../../Data/Chemba_loc1_OpenMeteo_API_01012019_30122024_Daily.csv", skiprows=3)
df["time"] = pd.to_datetime(df["time"])

openmeteo = df[df["time"].dt.year.isin([2021, 2022])].copy()

openmeteo.rename(columns={"precipitation_sum (mm)": "OpenMeteo"}, inplace=True)

# precip_df = precip_df[precip_df["Date"] < "2024-12-31"]

precip_df["OpenMeteo"] = openmeteo["OpenMeteo"].values


vc_daily = pd.read_csv("../../Data/Chemba_loc1_VisualCrossing_01012021_31122022_Daily.csv")
vc_daily = vc_daily.drop(columns="name")
vc_daily["datetime"] = pd.to_datetime(vc_daily["datetime"])

vc_daily.rename(columns={"precip": "VisualCrossing"}, inplace=True)

precip_df["VisualCrossing"] = vc_daily["VisualCrossing"].values

precip_df["Date"] = pd.to_datetime(precip_df["Date"])

corr_mat = precip_df.drop(columns=["Date"]).corr()

r2_mat = corr_mat**2

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("#######################")
print("R2 Daily\n", r2_mat)
print("#######################")

products = ['ERA5', 'IMERG', 'CHIRPS', 'GSMaP', 'OpenMeteo', 'VisualCrossing']
rmse_df = pd.DataFrame(index=products, columns=products)
mae_df = pd.DataFrame(index=products, columns=products)

for p1, p2 in itertools.combinations(products, 2):
    rmse = np.sqrt(mean_squared_error(precip_df[p1], precip_df[p2]))
    mae = mean_absolute_error(precip_df[p1], precip_df[p2])
    rmse_df.loc[p1, p2] = rmse
    rmse_df.loc[p2, p1] = rmse
    mae_df.loc[p1, p2] = mae
    mae_df.loc[p2, p1] = mae


print("RSME Daily\n", rmse_df)
print("#######################")
print("MAE Daily\n", mae_df)
print("#######################")

#
# plt.figure(figsize=(14, 8))
# plt.plot(precip_df["Date"], precip_df["ERA5"], label="ERA5", linewidth=1.5)
# plt.plot(precip_df["Date"], precip_df["IMERG"], label="IMERG", linewidth=1.5)
# plt.plot(precip_df["Date"], precip_df["CHIRPS"], label="CHIRPS", linewidth=1.5)
# plt.plot(precip_df["Date"], precip_df["GSMaP"], label="GSMaP", linewidth=1.5)
# plt.plot(precip_df["Date"], precip_df["OpenMeteo"], label="OpenMeteo", linewidth=1.5)
# plt.plot(precip_df["Date"], precip_df["VisualCrossing"], label="VisualCrossing", linewidth=1.5)
#
# plt.title("Daily Comparision - Precipitation")
# plt.xlabel("Date")
# plt.ylabel("Precipitation (mm)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# sns.scatterplot(x=precip_df["IMERG"], y=precip_df["GSMaP"])
# plt.xlabel("IMERG")
# plt.ylabel("GSMaP")
# plt.show()


##### 7-Day comparisions
# precip_df = precip_df.set_index("Date")
# precip_df_7d = precip_df.resample("7D").sum().reset_index()
# # print(precip_df_7d.head())
#
# corr_mat_7d = precip_df_7d.drop(columns=["Date"]).corr()
# # print(corr_mat_7d)
#
# r2_mat_7d = corr_mat_7d**2
#
# print("R2 7-Day\n", r2_mat_7d)
# print("#######################")
#
# rmse_df_7d = pd.DataFrame(index=products, columns=products)
# mae_df_7d = pd.DataFrame(index=products, columns=products)
#
# for p1, p2 in itertools.combinations(products, 2):
#     rmse_7d = np.sqrt(mean_squared_error(precip_df_7d[p1], precip_df_7d[p2]))
#     mae_7d = mean_absolute_error(precip_df_7d[p1], precip_df_7d[p2])
#     rmse_df_7d.loc[p1, p2] = rmse_7d/7
#     rmse_df_7d.loc[p2, p1] = rmse_7d/7
#     mae_df_7d.loc[p1, p2] = mae_7d/7
#     mae_df_7d.loc[p2, p1] = mae_7d/7
#
# print("RSME 7-Day\n", rmse_df_7d)
# print("#######################")
# print("MAE 7-Day\n", mae_df_7d)
# print("#######################")

#
# plt.figure(figsize=(14, 8))
# plt.plot(precip_df_7d["Date"], precip_df_7d["ERA5"], label="ERA5", linewidth=1.5)
# plt.plot(precip_df_7d["Date"], precip_df_7d["IMERG"], label="IMERG", linewidth=1.5)
# plt.plot(precip_df_7d["Date"], precip_df_7d["CHIRPS"], label="CHIRPS", linewidth=1.5)
# plt.plot(precip_df_7d["Date"], precip_df_7d["GSMaP"], label="GSMaP", linewidth=1.5)
# plt.plot(precip_df_7d["Date"], precip_df_7d["OpenMeteo"], label="OpenMeteo", linewidth=1.5)
# plt.plot(precip_df_7d["Date"], precip_df_7d["VisualCrossing"], label="VisualCrossing", linewidth=1.5)
#
# plt.title("Comparision 7-Day aggregates - Precipitation")
# plt.xlabel("Date")
# plt.ylabel("Precipitation (mm)")
# plt.legend()
# plt.tight_layout()
# plt.show()



# Time Series Decomposition
#
# def time_series_decompose(df, cols, period = 365):
#     fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
#
#     for col in cols:
#         series = df[col]
#         time = df.index
#
#         result = seasonal_decompose(series, period=period, model="additive")
#
#         axs[0].plot(time, series, label=col)
#         axs[1].plot(time, result.trend, label=col)
#         axs[2].plot(time, result.seasonal, label=col)
#         axs[3].plot(time, result.resid, label=col)
#
#     axs[0].set_title("Original Series")
#     axs[1].set_title("Trend")
#     axs[2].set_title("Seasonality")
#     axs[3].set_title("Residuals")
#
#     for ax in axs:
#         ax.legend()
#         ax.tick_params(axis='x', rotation=45)
#
#     plt.tight_layout()
#     plt.show()
#
#
# time_series_decompose(precip_df, cols=["ERA5", "OpenMeteo", "VisualCrossing", "IMERG"], period=365)

## Precipitation forecasting

# np.random.seed(123)
# random.seed(123)
# tf.random.set_seed(123)
# import statsmodels.api as sm
#
# era5_train_10_22 = "../../Data/ERA5/ERA5_2010_2022_Precip_Daily.tif"
# era5_train_00_09 = "../../Data/ERA5/ERA5_2000_2009_Precip_Daily.tif"
# era5_test = "../../Data/ERA5/ERA5_2023_2024_Precip_Daily.tif"
#
# era5_train_10_22_data, era5_train_dates, era5_train_transform, era5_train_crs, era5_train_bounds = read_tif(era5_train_10_22)
# era5_train_00_09_data, era5_train_dates_00_09, era5_train_transform_00_09, era5_train_crs_00_09, era5_train_bounds_00_09 = read_tif(era5_train_00_09)
#
# era5_test_data, era5_test_dates, era5_test_transform, era5_test_crs, era5_test_bounds = read_tif(era5_test)
#
# era5_train_row, era5_train_col = get_pixels_values(pt_lat, pt_lon, era5_train_transform)
#
# era5_train_10_22_precip = era5_train_10_22_data[:, era5_train_row, era5_train_col]
# era5_train_00_09_precip = era5_train_00_09_data[:, era5_train_row, era5_train_col]
# era5_test_precip = era5_test_data[:, era5_train_row, era5_train_row]
#
# train_dates_10_22 = pd.date_range(start="2010-01-01", end="2022-12-31", freq="D")
# train_dates_00_09 = pd.date_range(start="2000-01-01", end="2009-12-31", freq="D")
# test_dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
#
# precip_train_10_22_df = pd.DataFrame({
#     "Date" : train_dates_10_22,
#     "ERA5" : era5_train_10_22_precip
# })
#
# precip_train_00_09_df = pd.DataFrame({
#     "Date" : train_dates_00_09,
#     "ERA5" : era5_train_00_09_precip
# })
#
# precip_train_df = pd.concat([precip_train_00_09_df, precip_train_10_22_df], ignore_index=True)
#
# precip_test_df = pd.DataFrame({
#     "Date" : test_dates,
#     "ERA5" : era5_test_precip
# })
#
# precip_train_df = precip_train_df.dropna(subset=["ERA5"])
#
# precip_train_df["Date"] = pd.to_datetime(precip_train_df["Date"])
# precip_train_df.set_index("Date", inplace=True)
#
# precip_train_df["ERA5_Roll7"] = precip_train_df['ERA5'].rolling(7).sum()
#
# precip_train_df["DoY"] = precip_train_df.index.dayofyear
#
# precip_train_df['sin_DoY'] = np.sin(2 * np.pi * precip_train_df["DoY"]/365)
# precip_train_df['cos_DoY'] = np.cos(2 * np.pi * precip_train_df["DoY"]/365)
#
# precip_train_df = precip_train_df.dropna()
#
# features = ["ERA5", "ERA5_Roll7", "sin_DoY", "cos_DoY"]
# target = "ERA5"
#
# data = precip_train_df[features].values
#
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(data)
#
#
# def create_sequences_multivariate(data, input_len=30, output_len=7, target_index=0):
#     X, y = [], []
#     for i in range(len(data) - input_len - output_len + 1):
#         X.append(data[i:i+input_len])
#         y.append(data[i+input_len:i+input_len+output_len, target_index])
#
#     return np.array(X), np.array(y)
#
#
# X, y = create_sequences_multivariate(scaled_data, input_len=30, output_len=7, target_index=0)
#
# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.LSTM(64),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(7, activation='relu')
# ])
#
#
# model.compile(optimizer='adam', loss='mse')
# model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
#
#
# last_30 = scaled_data[-30:].reshape(1, 30, len(features))
#
#
# def monte_carlo_predict(model, X, num_samples=100):
#     predictions = np.zeros((num_samples, X.shape[0], 7))
#     for i in range(num_samples):
#         predictions[i] = model(X, training=True)
#     return predictions
#
#
# predictions_mc = monte_carlo_predict(model, last_30, num_samples=100)
#
# mean_preds = np.mean(predictions_mc, axis=0)[0]
# lower_bound = np.percentile(predictions_mc, 2.5, axis=0)[0]
# upper_bound = np.percentile(predictions_mc, 97.5, axis=0)[0]
#
#
# precip_scaler = MinMaxScaler()
# precip_scaler.fit(precip_train_df[[target]])
# mean_preds = precip_scaler.inverse_transform(mean_preds.reshape(-1, 1)).flatten()
# lower_bound = precip_scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
# upper_bound = precip_scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()
#
# print("7-day Forecast:")
# print("Mean:", mean_preds)
# print("Lower 2.5%:", lower_bound)
# print("Upper 97.5%:", upper_bound)
#
#
# precip_test_df = precip_test_df.dropna(subset=["ERA5"])
# precip_test_df["Date"] = pd.to_datetime(precip_test_df["Date"])
# precip_test_df.set_index("Date", inplace=True)
#
# precip_test_df["ERA5_Roll7"] = precip_test_df["ERA5"].rolling(7).mean()
# precip_test_df["DoY"] = precip_test_df.index.dayofyear
# precip_test_df["sin_DoY"] = np.sin(2 * np.pi * precip_test_df["DoY"] / 365)
# precip_test_df["cos_DoY"] = np.cos(2 * np.pi * precip_test_df["DoY"] / 365)
#
# precip_test_df = precip_test_df.dropna()
#
# test_data = precip_test_df[features].values
# scaled_test_data = scaler.transform(test_data)
#
#
# X_test, y_test = create_sequences_multivariate(scaled_test_data, input_len=30, output_len=7, target_index=0)
#
#
# predictions_mc_test = monte_carlo_predict(model, X_test, num_samples=100)
#
# mean_preds_test = np.mean(predictions_mc_test, axis=0)
# lower_bound_test = np.percentile(predictions_mc_test, 2.5, axis=0)
# upper_bound_test = np.percentile(predictions_mc_test, 97.5, axis=0)
#
#
# mean_preds_test_inv = precip_scaler.inverse_transform(mean_preds_test)
# lower_bound_test_inv = precip_scaler.inverse_transform(lower_bound_test)
# upper_bound_test_inv = precip_scaler.inverse_transform(upper_bound_test)
# y_test_inv = precip_scaler.inverse_transform(y_test)
#
#
# for i in range(7):
#     r2 = r2_score(y_test_inv[:, i], mean_preds_test_inv[:, i])
#     rmse = np.sqrt(mean_squared_error(y_test_inv[:, i], mean_preds_test_inv[:, i]))
#     print(f"Day {i+1} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
#
#
#
# fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex="all")
#
# for i in range(3):
#     axs[i].plot(y_test_inv[:, i+1], label="Actual", color="black")
#     axs[i].plot(mean_preds_test_inv[:, i+1], label="Predicted (Mean)", color="green")
#     axs[i].fill_between(
#         np.arange(len(mean_preds_test_inv)),
#         lower_bound_test_inv[:, i+1],
#         upper_bound_test_inv[:, i+1],
#         color="gray",
#         alpha=0.5,
#         label="97.5% CI"
#     )
#     axs[i].set_title(f"ERA5 Precipitation Prediction - Day {i+1}")
#     axs[i].set_ylabel("Precipitation (mm)")
#     axs[i].legend()
#     axs[i].grid(True)
#
# axs[-1].set_xlabel("Time step")
# plt.tight_layout()
# plt.show()

# data = precip_train_df["ERA5"].values.reshape(-1, 1)
#
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(data)
#
#
# def create_sequences(data, input_len=30, output_len=7):
#     X, y = [], []
#     for i in range(len(data) - input_len - output_len + 1):
#         X.append(data[i:i+input_len])
#         y.append(data[i+input_len:i+input_len+output_len].flatten())
#     return np.array(X), np.array(y)
#
#
# X, y = create_sequences(scaled_data, input_len=30, output_len=7)
#
# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(30, 1)),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.LSTM(32),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(7, activation='relu')
# ])
#
# model.compile(optimizer='adam', loss='mse')
# model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
#
#
# def monte_carlo_predict(model, X, num_samples=100):
#     predictions = np.zeros((num_samples, X.shape[0], 7))
#     for i in range(num_samples):
#         predictions[i] = model(X, training=True)
#     return predictions
#
#
# last_30 = scaled_data[-30:].reshape(1, 30, 1)
# predictions_mc = monte_carlo_predict(model, last_30, num_samples=100)
# #
# # last_30 = scaled_data[-30:].reshape(1, 30, 1)
# # forecast_scaled = model.predict(last_30)
# # forecast = scaler.inverse_transform(forecast_scaled)[0]
#
# mean_preds = np.mean(predictions_mc, axis=0)
# lower_bound = np.percentile(predictions_mc, 2.5, axis=0)
# upper_bound = np.percentile(predictions_mc, 97.5, axis=0)
#
# mean_preds = scaler.inverse_transform(mean_preds)[0]
# lower_bound = scaler.inverse_transform(lower_bound)[0]
# upper_bound = scaler.inverse_transform(upper_bound)[0]
#
# print(f"Predicted Precipitation (Mean): {mean_preds}")
# print(f"Lower Bound (2.5%): {lower_bound}")
# print(f"Upper Bound (97.5%): {upper_bound}")
#
#
# precip_test_df["Date"] = pd.to_datetime(precip_test_df["Date"])
# precip_test_df = precip_test_df.dropna(subset=["ERA5"])
# test_series = precip_test_df["ERA5"].values.reshape(-1, 1)
# test_scaled = scaler.transform(test_series)
#
# X_test, y_test = create_sequences(test_scaled, input_len=30, output_len=7)
#
# predictions_mc_test = monte_carlo_predict(model, X_test, num_samples=100)
#
# mean_preds_test = np.mean(predictions_mc_test, axis=0)
# lower_bound_test = np.percentile(predictions_mc_test, 2.5, axis=0)
# upper_bound_test = np.percentile(predictions_mc_test, 97.5, axis=0)
#
# mean_preds_test = scaler.inverse_transform(mean_preds_test)
# lower_bound_test = scaler.inverse_transform(lower_bound_test)
# upper_bound_test = scaler.inverse_transform(upper_bound_test)
# # y_pred_scaled = model.predict(X_test)
# # y_pred = scaler.inverse_transform(y_pred_scaled)
# y_true = scaler.inverse_transform(y_test)
#
# for i in range(7):
#     r2 = r2_score(y_true[:, i], mean_preds_test[:, i])
#     rmse = np.sqrt(mean_squared_error(y_true[:, i], mean_preds_test[:, i]))
#     print(f"Day {i+1} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
#
# fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex="all")
#
# for i in range(3):
#     axs[i].plot(y_true[:, i+1], label="Actual", color="black")
#     axs[i].plot(mean_preds_test[:, i+1], label="Predicted (Mean)", color="green")
#     axs[i].fill_between(
#         np.arange(len(mean_preds_test)),
#         lower_bound_test[:, i+1],
#         upper_bound_test[:, i+1],
#         color="gray",
#         alpha=0.5,
#         label="97.5% CI"
#     )
#     axs[i].set_title(f"ERA5 Precipitation Prediction - Day {i+1}")
#     axs[i].set_ylabel("Precipitation (mm)")
#     axs[i].legend()
#     axs[i].grid(True)
#
# axs[-1].set_xlabel("Time step")
# plt.tight_layout()
# plt.show()
