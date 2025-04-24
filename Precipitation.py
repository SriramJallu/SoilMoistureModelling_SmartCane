import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

era5 = "../../Data/ERA5/ERA5_2024_Daily.tif"
imerg = "../../Data/IMERG/IMERG_2024_Daily.tif"
chirps = "../../Data/CHIRPS/CHIRPS_2024_Daily.tif"
gsmap = "../../Data/GSMaP/GSMaP_2024_Daily.tif"

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


pt_lat, pt_lon = -17.331524, 34.954147

era5_row, era5_col = get_pixels_values(pt_lat, pt_lon, era5_transform)
era5_precip = era5_data[:, era5_row, era5_col]


imerg_row, imerg_col = get_pixels_values(pt_lat, pt_lon, imerg_transform)
imerg_precip = imerg_data[:, imerg_row, imerg_col]


chirps_row, chirps_col = get_pixels_values(pt_lat, pt_lon, chirps_transform)
chirps_precip = chirps_data[:, chirps_row, chirps_col]


gsmap_row, gsmap_col = get_pixels_values(pt_lat, pt_lon, gsmap_transform)
gsmap_precip = gsmap_data[:, gsmap_row, gsmap_col]


dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")

precip_df = pd.DataFrame({
    "Date" : dates,
    "ERA5" : era5_precip,
    "IMERG" : imerg_precip,
    "CHIRPS" : chirps_precip,
    "GSMaP" : gsmap_precip,
})


df = pd.read_csv("../../Data/Chemba_loc1_OpenMeteo_API_01012019_30122024_Daily.csv", skiprows=3)
df["time"] = pd.to_datetime(df["time"])

openmeteo = df[df["time"].dt.year == 2024].copy()

openmeteo.rename(columns={"precipitation_sum (mm)": "OpenMeteo"}, inplace=True)

precip_df = precip_df[precip_df["Date"] < "2024-12-31"]

precip_df["OpenMeteo"] = openmeteo["OpenMeteo"].values

print(precip_df.head())

print(precip_df.shape)


precip_df["Date"] = pd.to_datetime(precip_df["Date"])

corr_mat = precip_df.drop(columns=["Date"]).corr()

r2_mat = corr_mat**2

print(r2_mat)

plt.figure(figsize=(14, 8))
plt.plot(precip_df["Date"], precip_df["ERA5"], label="ERA5", linewidth=1.5)
plt.plot(precip_df["Date"], precip_df["IMERG"], label="IMERG", linewidth=1.5)
plt.plot(precip_df["Date"], precip_df["CHIRPS"], label="CHIRPS", linewidth=1.5)
plt.plot(precip_df["Date"], precip_df["GSMaP"], label="GSMaP", linewidth=1.5)
plt.plot(precip_df["Date"], precip_df["OpenMeteo"], label="OpenMeteo", linewidth=1.5)

plt.title("Comparision")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.tight_layout()
plt.show()


rmse_imerg = np.sqrt(mean_squared_error(precip_df["OpenMeteo"], precip_df["ERA5"]))
mae_imerg = mean_absolute_error(precip_df["OpenMeteo"], precip_df["ERA5"])

print(rmse_imerg, mae_imerg)

# sns.scatterplot(x=precip_df["IMERG"], y=precip_df["GSMaP"])
# plt.xlabel("IMERG")
# plt.ylabel("GSMaP")
# plt.show()

