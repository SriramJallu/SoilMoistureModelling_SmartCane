import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import itertools
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge
from sklearn.utils import resample
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


era5 = "../../Data/ERA5/ERA5_2021_2022_Precip_Daily.tif"
imerg = "../../Data/IMERG/IMERG_2021_2022_Precip_Daily.tif"
chirps = "../../Data/CHIRPS/CHIRPS_2021_2022_Precip_Daily.tif"
gsmap = "../../Data/GSMaP/GSMaP_2021_2022_Precip_Daily.tif"

era5_train = "../../Data/ERA5/ERA5_2019_2024_Precip_Daily.tif"
era5_test = "../../Data/ERA5/ERA5_2017_2018_Precip_Daily.tif"


def read_tif(filename):
    with rasterio.open(filename) as f:
        data = f.read()
        dates = [str(band) for band in f.descriptions]
        return data, dates, f.transform, f.crs, f.bounds


era5_data, era5_dates, era5_transform, era5_crs, era5_bounds = read_tif(era5)
imerg_data, imerg_dates, imerg_transform, imerg_crs, imerg_bounds = read_tif(imerg)
chirps_data, chirps_dates, chirps_transform, chirps_crs, chirps_bounds = read_tif(chirps)
gsmap_data, gsmap_dates, gsmap_transform, gsmap_crs, gsmap_bounds = read_tif(gsmap)

era5_train_data, era5_train_dates, era5_train_transform, era5_train_crs, era5_train_bounds = read_tif(era5_train)
era5_test_data, era5_test_dates, era5_test_transform, era5_test_crs, era5_test_bounds = read_tif(era5_test)


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


precip_df = precip_df.set_index("Date")
precip_df_7d = precip_df.resample("7D").sum().reset_index()
# print(precip_df_7d.head())

corr_mat_7d = precip_df_7d.drop(columns=["Date"]).corr()
# print(corr_mat_7d)

r2_mat_7d = corr_mat_7d**2

print("R2 7-Day\n", r2_mat_7d)
print("#######################")

rmse_df_7d = pd.DataFrame(index=products, columns=products)
mae_df_7d = pd.DataFrame(index=products, columns=products)

for p1, p2 in itertools.combinations(products, 2):
    rmse_7d = np.sqrt(mean_squared_error(precip_df_7d[p1], precip_df_7d[p2]))
    mae_7d = mean_absolute_error(precip_df_7d[p1], precip_df_7d[p2])
    rmse_df_7d.loc[p1, p2] = rmse_7d/7
    rmse_df_7d.loc[p2, p1] = rmse_7d/7
    mae_df_7d.loc[p1, p2] = mae_7d/7
    mae_df_7d.loc[p2, p1] = mae_7d/7

print("RSME 7-Day\n", rmse_df_7d)
print("#######################")
print("MAE 7-Day\n", mae_df_7d)
print("#######################")

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

era5_train_row, era5_train_col = get_pixels_values(pt_lat, pt_lon, era5_train_transform)
era5_train_precip = era5_train_data[:, era5_train_row, era5_train_col]

era5_test_row, era5_test_col = get_pixels_values(pt_lat, pt_lon, era5_test_transform)
era5_test_precip = era5_test_data[:, era5_test_row, era5_test_col]

train_dates = pd.date_range(start="2019-01-01", end="2024-12-31", freq="D")
test_dates = pd.date_range(start="2017-01-01", end="2018-12-31", freq="D")

precip_train_df = pd.DataFrame({
    "Date" : train_dates,
    "ERA5" : era5_train_precip
})

precip_test_df = pd.DataFrame({
    "Date" : test_dates,
    "ERA5" : era5_test_precip
})

precip_train_df = precip_train_df.copy()
precip_train_df = precip_train_df.dropna(subset=["ERA5"])
data = precip_train_df["ERA5"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


def create_sequences(data, input_len=30, output_len=7):
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len].flatten())
    return np.array(X), np.array(y)


X, y = create_sequences(scaled_data, input_len=30, output_len=7)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(30, 1)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(7)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)


last_30 = scaled_data[-30:].reshape(1, 30, 1)
forecast_scaled = model.predict(last_30)
forecast = scaler.inverse_transform(forecast_scaled)[0]

print(forecast)

precip_test_df["Date"] = pd.to_datetime(precip_test_df["Date"])
precip_test_df = precip_test_df.dropna(subset=["ERA5"])
test_series = precip_test_df["ERA5"].values.reshape(-1, 1)
test_scaled = scaler.transform(test_series)

X_test, y_test = create_sequences(test_scaled, input_len=30, output_len=7)

y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test)

for i in range(7):
    r2 = r2_score(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    print(f"Day {i+1} - R²: {r2:.4f}, RMSE: {rmse:.4f}")

fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex="all")

for i in range(3):
    axs[i].plot(y_true[:, i], label="Actual", color="black")
    axs[i].plot(y_pred[:, i], label="Predicted", color="green")
    axs[i].set_title(f"ERA5 Precipitation Prediction - Day {i+1}")
    axs[i].set_ylabel("Precipitation (mm)")
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Time step")
plt.tight_layout()
plt.show()
