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

era5 = "../../Data/ERA5/ERA5_2021_2022_Precip_Daily.tif"
imerg = "../../Data/IMERG/IMERG_2021_2022_Precip_Daily.tif"
chirps = "../../Data/CHIRPS/CHIRPS_2021_2022_Precip_Daily.tif"
gsmap = "../../Data/GSMaP/GSMaP_2021_2022_Precip_Daily.tif"

#
# def read_tif(filename):
#     with rasterio.open(filename) as f:
#         return f.read(), f.transform, f.crs, f.bounds

def read_tif(filename):
    with rasterio.open(filename) as f:
        data = f.read()
        dates = [str(band) for band in f.descriptions]
        return data, dates, f.transform, f.crs, f.bounds


era5_data, era5_dates, era5_transform, era5_crs, era5_bounds = read_tif(era5)
imerg_data, imerg_dates, imerg_transform, imerg_crs, imerg_bounds = read_tif(imerg)
chirps_data, chirps_dates, chirps_transform, chirps_crs, chirps_bounds = read_tif(chirps)
gsmap_data, gsmap_dates, gsmap_transform, gsmap_crs, gsmap_bounds = read_tif(gsmap)


print(f"ERA5 data shape: {era5_data.shape}")
print(f"IMERG data shape: {imerg_data.shape}")
print(f"CHIRPS data shape: {chirps_data.shape}")
print(f"GSMAP data shape: {gsmap_data.shape}")


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

print(precip_df.head())

print(precip_df.shape)

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


plt.figure(figsize=(14, 8))
plt.plot(precip_df["Date"], precip_df["ERA5"], label="ERA5", linewidth=1.5)
plt.plot(precip_df["Date"], precip_df["IMERG"], label="IMERG", linewidth=1.5)
plt.plot(precip_df["Date"], precip_df["CHIRPS"], label="CHIRPS", linewidth=1.5)
plt.plot(precip_df["Date"], precip_df["GSMaP"], label="GSMaP", linewidth=1.5)
plt.plot(precip_df["Date"], precip_df["OpenMeteo"], label="OpenMeteo", linewidth=1.5)
plt.plot(precip_df["Date"], precip_df["VisualCrossing"], label="VisualCrossing", linewidth=1.5)

plt.title("Daily Comparision - Precipitation")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.tight_layout()
plt.show()

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


plt.figure(figsize=(14, 8))
plt.plot(precip_df_7d["Date"], precip_df_7d["ERA5"], label="ERA5", linewidth=1.5)
plt.plot(precip_df_7d["Date"], precip_df_7d["IMERG"], label="IMERG", linewidth=1.5)
plt.plot(precip_df_7d["Date"], precip_df_7d["CHIRPS"], label="CHIRPS", linewidth=1.5)
plt.plot(precip_df_7d["Date"], precip_df_7d["GSMaP"], label="GSMaP", linewidth=1.5)
plt.plot(precip_df_7d["Date"], precip_df_7d["OpenMeteo"], label="OpenMeteo", linewidth=1.5)
plt.plot(precip_df_7d["Date"], precip_df_7d["VisualCrossing"], label="VisualCrossing", linewidth=1.5)

plt.title("Comparision 7-Day aggregates - Precipitation")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.tight_layout()
plt.show()


precip_df_14d = precip_df.resample("14D").sum().reset_index()
# print(precip_df_14d.head())

corr_mat_14d = precip_df_14d.drop(columns=["Date"]).corr()
# print(corr_mat_14d)

r2_mat_14d = corr_mat_14d**2
print("R2 14-Day\n", r2_mat_14d)
print("#######################")

rmse_df_14d = pd.DataFrame(index=products, columns=products)
mae_df_14d = pd.DataFrame(index=products, columns=products)

for p1, p2 in itertools.combinations(products, 2):
    rmse_14d = np.sqrt(mean_squared_error(precip_df_14d[p1], precip_df_14d[p2]))
    mae_14d = mean_absolute_error(precip_df_14d[p1], precip_df_14d[p2])
    rmse_df_14d.loc[p1, p2] = rmse_14d/14
    rmse_df_14d.loc[p2, p1] = rmse_14d/14
    mae_df_14d.loc[p1, p2] = mae_14d/14
    mae_df_14d.loc[p2, p1] = mae_14d/14

print("RSME 14-Day\n", rmse_df_14d)
print("#######################")
print("MAE 14-Day\n", mae_df_14d)
print("#######################")

plt.figure(figsize=(14, 8))
plt.plot(precip_df_14d["Date"], precip_df_14d["ERA5"], label="ERA5", linewidth=1.5)
plt.plot(precip_df_14d["Date"], precip_df_14d["IMERG"], label="IMERG", linewidth=1.5)
plt.plot(precip_df_14d["Date"], precip_df_14d["CHIRPS"], label="CHIRPS", linewidth=1.5)
plt.plot(precip_df_14d["Date"], precip_df_14d["GSMaP"], label="GSMaP", linewidth=1.5)
plt.plot(precip_df_14d["Date"], precip_df_14d["OpenMeteo"], label="OpenMeteo", linewidth=1.5)
plt.plot(precip_df_14d["Date"], precip_df_14d["VisualCrossing"], label="VisualCrossing", linewidth=1.5)

plt.title("Comparision 14-Day aggregates - Precipitation")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.tight_layout()
plt.show()

print(precip_df.head())

# smoothed = precip_df.rolling(7).mean().reset_index()
#
# plt.figure(figsize=(14, 8))
# for col in ["ERA5", "IMERG", "CHIRPS", "GSMaP", "OpenMeteo", "VisualCrossing"]:
#     plt.plot(smoothed["Date"], smoothed[col], label=col, linewidth=2)
#
# plt.title("7-Day Smoothed Precip Comparison")
# plt.xlabel("Date")
# plt.ylabel("Precipitation (mm)")
# plt.legend()
# plt.tight_layout()
# plt.show()

