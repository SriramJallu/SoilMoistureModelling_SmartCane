import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools


era5 = "../../Data/ERA5/ERA5_2021_2022_RH_Daily.tif"
cfs = "../../Data/CFSV2/CFSV2_2021_2022_RH_Daily.tif"


def read_tif(filename):
    with rasterio.open(filename) as f:
        data = f.read()
        dates = [str(band) for band in f.descriptions]
        return data, dates, f.transform, f.crs, f.bounds


era5_data, era5_dates, era5_transform, era5_crs, era5_bounds = read_tif(era5)
cfs_data, cfs_dates, cfs_transform, cfs_crs, cfs_bounds = read_tif(cfs)


print(f"ERA5 data shape: {era5_data.shape}")
print(f"CFS data shape: {cfs_data.shape}")


def get_pixels_values(lat, lon, transform):
    row, col = ~transform*(lon, lat)
    return int(row), int(col)

# Chemba
pt_lat, pt_lon = -17.331524, 34.954147

# Transmara, Kenya
# pt_lat, pt_lon = -1.023509, 34.740671

era5_row, era5_col = get_pixels_values(pt_lat, pt_lon, era5_transform)
era5_rh = era5_data[:, era5_row, era5_col]

cfs_row, cfs_col = get_pixels_values(pt_lat, pt_lon, cfs_transform)
cfs_rh = cfs_data[:, cfs_row, cfs_col]


dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")

rh_df = pd.DataFrame({
    "Date" : dates,
    "ERA5" : era5_rh,
    "CFS" : cfs_rh
})


df = pd.read_csv("../../Data/Chemba_loc1_OpenMeteo_API_01012019_30122024_Daily.csv", skiprows=3)
df["time"] = pd.to_datetime(df["time"])

openmeteo = df[df["time"].dt.year.isin([2021, 2022])].copy()

openmeteo.rename(columns={"relative_humidity_2m_mean (%)": "OpenMeteo"}, inplace=True)
rh_df["OpenMeteo"] = openmeteo["OpenMeteo"].values

vc_daily = pd.read_csv("../../Data/Chemba_loc1_VisualCrossing_01012021_31122022_Daily.csv")
vc_daily = vc_daily.drop(columns="name")
vc_daily["datetime"] = pd.to_datetime(vc_daily["datetime"])

vc_daily.rename(columns={"humidity": "VisualCrossing"}, inplace=True)

rh_df["VisualCrossing"] = vc_daily["VisualCrossing"].values

print(rh_df.head())

print(rh_df.shape)


rh_df["Date"] = pd.to_datetime(rh_df["Date"])

corr_mat = rh_df.drop(columns=["Date"]).corr()

r2_mat = corr_mat**2

print("#######################")
print("R2 Daily\n", r2_mat)
print("#######################")

products = ['ERA5', 'CFS', 'OpenMeteo', 'VisualCrossing']
rmse_df = pd.DataFrame(index=products, columns=products)
mae_df = pd.DataFrame(index=products, columns=products)

for p1, p2 in itertools.combinations(products, 2):
    rmse = np.sqrt(mean_squared_error(rh_df[p1], rh_df[p2]))
    mae = mean_absolute_error(rh_df[p1], rh_df[p2])
    rmse_df.loc[p1, p2] = rmse
    rmse_df.loc[p2, p1] = rmse
    mae_df.loc[p1, p2] = mae
    mae_df.loc[p2, p1] = mae


print("RSME Daily\n", rmse_df)
print("#######################")
print("MAE Daily\n", mae_df)
print("#######################")


plt.figure(figsize=(14, 8))
plt.plot(rh_df["Date"], rh_df["ERA5"], label="ERA5", linewidth=1.5)
plt.plot(rh_df["Date"], rh_df["CFS"], label="CFS", linewidth=1.5)
plt.plot(rh_df["Date"], rh_df["OpenMeteo"], label="OpenMeteo", linewidth=1.5)
plt.plot(rh_df["Date"], rh_df["VisualCrossing"], label="VisualCrossing", linewidth=1.5)

plt.title("Daily Comparision - Relative Humidity")
plt.xlabel("Date")
plt.ylabel("Relative Humidity (%)")
plt.legend()
plt.tight_layout()
plt.show()

# sns.scatterplot(x=precip_df["IMERG"], y=precip_df["GSMaP"])
# plt.xlabel("IMERG")
# plt.ylabel("GSMaP")
# plt.show()


rh_df = rh_df.set_index("Date")
rh_df_7d = rh_df.resample("7D").mean().reset_index()
# print(precip_df_7d.head())

corr_mat_7d = rh_df_7d.drop(columns=["Date"]).corr()
# print(corr_mat_7d)

r2_mat_7d = corr_mat_7d**2

print("R2 7-Day\n", r2_mat_7d)
print("#######################")

rmse_df_7d = pd.DataFrame(index=products, columns=products)
mae_df_7d = pd.DataFrame(index=products, columns=products)

for p1, p2 in itertools.combinations(products, 2):
    rmse_7d = np.sqrt(mean_squared_error(rh_df_7d[p1], rh_df_7d[p2]))
    mae_7d = mean_absolute_error(rh_df_7d[p1], rh_df_7d[p2])
    rmse_df_7d.loc[p1, p2] = rmse_7d/7
    rmse_df_7d.loc[p2, p1] = rmse_7d/7
    mae_df_7d.loc[p1, p2] = mae_7d/7
    mae_df_7d.loc[p2, p1] = mae_7d/7

print("RSME 7-Day\n", rmse_df_7d)
print("#######################")
print("MAE 7-Day\n", mae_df_7d)
print("#######################")


plt.figure(figsize=(14, 8))
plt.plot(rh_df_7d["Date"], rh_df_7d["ERA5"], label="ERA5", linewidth=1.5)
plt.plot(rh_df_7d["Date"], rh_df_7d["CFS"], label="CFS", linewidth=1.5)
plt.plot(rh_df_7d["Date"], rh_df_7d["OpenMeteo"], label="OpenMeteo", linewidth=1.5)
plt.plot(rh_df_7d["Date"], rh_df_7d["VisualCrossing"], label="VisualCrossing", linewidth=1.5)

plt.title("Comparision 7-Day aggregates - Relative Humidity")
plt.xlabel("Date")
plt.ylabel("Relative Humidity (%)")
plt.legend()
plt.tight_layout()
plt.show()


rh_df_14d = rh_df.resample("14D").mean().reset_index()
# print(precip_df_14d.head())

corr_mat_14d = rh_df_14d.drop(columns=["Date"]).corr()
# print(corr_mat_14d)

r2_mat_14d = corr_mat_14d**2
print("R2 14-Day\n", r2_mat_14d)
print("#######################")

rmse_df_14d = pd.DataFrame(index=products, columns=products)
mae_df_14d = pd.DataFrame(index=products, columns=products)

for p1, p2 in itertools.combinations(products, 2):
    rmse_14d = np.sqrt(mean_squared_error(rh_df_14d[p1], rh_df_14d[p2]))
    mae_14d = mean_absolute_error(rh_df_14d[p1], rh_df_14d[p2])
    rmse_df_14d.loc[p1, p2] = rmse_14d/14
    rmse_df_14d.loc[p2, p1] = rmse_14d/14
    mae_df_14d.loc[p1, p2] = mae_14d/14
    mae_df_14d.loc[p2, p1] = mae_14d/14

print("RSME 14-Day\n", rmse_df_14d)
print("#######################")
print("MAE 14-Day\n", mae_df_14d)
print("#######################")

plt.figure(figsize=(14, 8))
plt.plot(rh_df_14d["Date"], rh_df_14d["ERA5"], label="ERA5", linewidth=1.5)
plt.plot(rh_df_14d["Date"], rh_df_14d["CFS"], label="CFS", linewidth=1.5)
plt.plot(rh_df_14d["Date"], rh_df_14d["OpenMeteo"], label="OpenMeteo", linewidth=1.5)
plt.plot(rh_df_14d["Date"], rh_df_14d["VisualCrossing"], label="VisualCrossing", linewidth=1.5)

plt.title("Comparision 14-Day aggregates - Relative Humidity")
plt.xlabel("Date")
plt.ylabel("Relative Humidity (%)")
plt.legend()
plt.tight_layout()
plt.show()
