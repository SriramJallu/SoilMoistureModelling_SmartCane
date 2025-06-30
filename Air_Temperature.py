import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import random


era5 = "../../Data/ERA5/ERA5_2021_2022_Temp_Daily.tif"
gldas = "../../Data/GLDAS/GLDAS_2021_2022_Temp_Daily.tif"
cfs = "../../Data/CFSV2/CFSV2_2021_2022_Temp_Daily.tif"


era5_train = "../../Data/ERA5/ERA5_2015_2022_Temp_Daily.tif"
era5_test = "../../Data/ERA5/ERA5_2023_2024_Temp_Daily.tif"

def read_tif(filename):
    with rasterio.open(filename) as f:
        data = f.read()
        dates = [str(band) for band in f.descriptions]
        return data, dates, f.transform, f.crs, f.bounds


era5_data, era5_dates, era5_transform, era5_crs, era5_bounds = read_tif(era5)
gldas_data, gldas_dates, gldas_transform, gldas_crs, gldas_bounds = read_tif(gldas)
cfs_data, cfs_dates, cfs_transform, cfs_crs, cfs_bounds = read_tif(cfs)


print(f"ERA5 data shape: {era5_data.shape}")
print(f"GLDAS data shape: {gldas_data.shape}")
print(f"CFS data shape: {cfs_data.shape}")


def get_pixels_values(lat, lon, transform):
    row, col = ~transform*(lon, lat)
    return int(row), int(col)

# Chemba
pt_lat, pt_lon = -17.331524, 34.954147

# Transmara, Kenya
# pt_lat, pt_lon = -1.023509, 34.740671

era5_row, era5_col = get_pixels_values(pt_lat, pt_lon, era5_transform)
era5_temp = era5_data[:, era5_row, era5_col]


gldas_row, gldas_col = get_pixels_values(pt_lat, pt_lon, gldas_transform)
gldas_temp = gldas_data[:, gldas_row, gldas_col]


cfs_row, cfs_col = get_pixels_values(pt_lat, pt_lon, cfs_transform)
cfs_temp = cfs_data[:, cfs_row, cfs_col]


dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")

temp_df = pd.DataFrame({
    "Date" : dates,
    "ERA5" : era5_temp,
    "GLDAS" : gldas_temp,
    "CFS" : cfs_temp
})


df = pd.read_csv("../../Data/Chemba_loc1_OpenMeteo_API_01012019_30122024_Daily.csv", skiprows=3)
df["time"] = pd.to_datetime(df["time"])

openmeteo = df[df["time"].dt.year.isin([2021, 2022])].copy()

openmeteo.rename(columns={"temperature_2m_mean (°C)": "OpenMeteo"}, inplace=True)
temp_df["OpenMeteo"] = openmeteo["OpenMeteo"].values

vc_daily = pd.read_csv("../../Data/Chemba_loc1_VisualCrossing_01012021_31122022_Daily.csv")
vc_daily = vc_daily.drop(columns="name")
vc_daily["datetime"] = pd.to_datetime(vc_daily["datetime"])

vc_daily.rename(columns={"temp": "VisualCrossing"}, inplace=True)

temp_df["VisualCrossing"] = vc_daily["VisualCrossing"].values

print(temp_df.head())

print(temp_df.shape)


temp_df["Date"] = pd.to_datetime(temp_df["Date"])

corr_mat = temp_df.drop(columns=["Date"]).corr()

r2_mat = corr_mat**2

print("#######################")
print("R2 Daily\n", r2_mat)
print("#######################")

products = ['ERA5', 'GLDAS', 'CFS', 'OpenMeteo', 'VisualCrossing']
rmse_df = pd.DataFrame(index=products, columns=products)
mae_df = pd.DataFrame(index=products, columns=products)

for p1, p2 in itertools.combinations(products, 2):
    rmse = np.sqrt(mean_squared_error(temp_df[p1], temp_df[p2]))
    mae = mean_absolute_error(temp_df[p1], temp_df[p2])
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
# plt.plot(temp_df["Date"], temp_df["ERA5"], label="ERA5", linewidth=1.5)
# plt.plot(temp_df["Date"], temp_df["GLDAS"], label="GLDAS", linewidth=1.5)
# plt.plot(temp_df["Date"], temp_df["CFS"], label="CFS", linewidth=1.5)
# plt.plot(temp_df["Date"], temp_df["OpenMeteo"], label="OpenMeteo", linewidth=1.5)
# plt.plot(temp_df["Date"], temp_df["VisualCrossing"], label="VisualCrossing", linewidth=1.5)
#
# plt.title("Daily Comparision - Temperature")
# plt.xlabel("Date")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# sns.scatterplot(x=precip_df["IMERG"], y=precip_df["GSMaP"])
# plt.xlabel("IMERG")
# plt.ylabel("GSMaP")
# plt.show()


temp_df = temp_df.set_index("Date")
temp_df_7d = temp_df.resample("7D").mean().reset_index()
# print(precip_df_7d.head())

corr_mat_7d = temp_df_7d.drop(columns=["Date"]).corr()
# print(corr_mat_7d)

r2_mat_7d = corr_mat_7d**2

print("R2 7-Day\n", r2_mat_7d)
print("#######################")

rmse_df_7d = pd.DataFrame(index=products, columns=products)
mae_df_7d = pd.DataFrame(index=products, columns=products)

for p1, p2 in itertools.combinations(products, 2):
    rmse_7d = np.sqrt(mean_squared_error(temp_df_7d[p1], temp_df_7d[p2]))
    mae_7d = mean_absolute_error(temp_df_7d[p1], temp_df_7d[p2])
    rmse_df_7d.loc[p1, p2] = rmse_7d/7
    rmse_df_7d.loc[p2, p1] = rmse_7d/7
    mae_df_7d.loc[p1, p2] = mae_7d/7
    mae_df_7d.loc[p2, p1] = mae_7d/7

print("RSME 7-Day\n", rmse_df_7d)
print("#######################")
print("MAE 7-Day\n", mae_df_7d)
print("#######################")
#
#
# plt.figure(figsize=(14, 8))
# plt.plot(temp_df_7d["Date"], temp_df_7d["ERA5"], label="ERA5", linewidth=1.5)
# plt.plot(temp_df_7d["Date"], temp_df_7d["GLDAS"], label="GLDAS", linewidth=1.5)
# plt.plot(temp_df_7d["Date"], temp_df_7d["CFS"], label="CFS", linewidth=1.5)
# plt.plot(temp_df_7d["Date"], temp_df_7d["OpenMeteo"], label="OpenMeteo", linewidth=1.5)
# plt.plot(temp_df_7d["Date"], temp_df_7d["VisualCrossing"], label="VisualCrossing", linewidth=1.5)
#
# plt.title("Comparision 7-Day aggregates - Temperature")
# plt.xlabel("Date")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.tight_layout()
# plt.show()


# from sktime.forecasting.theta import ThetaForecaster
# from sktime.forecasting.base import ForecastingHorizon
# from sktime.utils.plotting import plot_series
#
# era5_series = temp_df["ERA5"].copy()
#
# era5_series.index = pd.to_datetime(era5_series.index)
# era5_series = era5_series.asfreq('D')
#
# # print(era5_series)
# fh = ForecastingHorizon(pd.date_range(start=era5_series.index[-1] + pd.Timedelta(days=1), periods=7), is_relative=False)
#
# forecaster = ThetaForecaster(sp=365)
# forecaster.fit(era5_series, fh=fh)
#
# coverage = 0.95
# y_pred_ints = forecaster.predict_interval(fh=fh, coverage=coverage)
# y_pred = forecaster.predict(fh=fh)
#
#
# fig, ax = plot_series(era5_series[-30:], y_pred, labels=["Observed", "Forecast"], pred_interval=y_pred_ints)
# ax.set_title("ERA5 Temperature: Forecast with 90% Prediction Interval")
# ax.legend(loc='lower left', bbox_to_anchor=(0, -0.15), ncol=1)
# plt.tight_layout()
# plt.show()


np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)

### Temperature Forecasting

# era5_train_data, era5_train_dates, era5_train_transform, era5_train_crs, era5_train_bounds = read_tif(era5_train)
# era5_test_data, era5_test_dates, era5_test_transform, era5_test_crs, era5_test_bounds = read_tif(era5_test)
#
# era5_train_row, era5_train_col = get_pixels_values(pt_lat, pt_lon, era5_train_transform)
# era5_train_temp = era5_train_data[:, era5_train_row, era5_train_col]
#
# era5_test_row, era5_test_col = get_pixels_values(pt_lat, pt_lon, era5_test_transform)
# era5_test_temp = era5_test_data[:, era5_test_row, era5_test_col]
#
# train_dates = pd.date_range(start="2015-01-01", end="2022-12-31", freq="D")
# test_dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
#
# temp_train_df = pd.DataFrame({
#     "Date" : train_dates,
#     "ERA5" : era5_train_temp
# })
#
# temp_test_df = pd.DataFrame({
#     "Date" : test_dates,
#     "ERA5" : era5_test_temp
# })
#
# temp_train_df = temp_train_df.dropna(subset=["ERA5"])
# data = temp_train_df["ERA5"].values.reshape(-1, 1)
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
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(30, 1))),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(7)
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
# predictions_mc = monte_carlo_predict(model, last_30, num_samples=200)
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
# print(f"Predicted Temperature (Mean): {mean_preds}")
# print(f"Lower Bound (2.5%): {lower_bound}")
# print(f"Upper Bound (97.5%): {upper_bound}")
#
#
# temp_test_df["Date"] = pd.to_datetime(temp_test_df["Date"])
# precip_test_df = temp_test_df.dropna(subset=["ERA5"])
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
#     axs[i].plot(y_true[:, i], label="Actual", color="black")
#     axs[i].plot(mean_preds_test[:, i], label="Predicted (Mean)", color="green")
#     axs[i].fill_between(
#         np.arange(len(mean_preds_test)),
#         lower_bound_test[:, i],
#         upper_bound_test[:, i],
#         color="gray",
#         alpha=0.5,
#         label="95% CI"
#     )
#     axs[i].set_title(f"ERA5 Temperature Prediction - Day {i+1}")
#     axs[i].set_ylabel("Temperature")
#     axs[i].legend()
#     axs[i].grid(True)
#
# axs[-1].set_xlabel("Time step")
# plt.tight_layout()
# plt.show()
