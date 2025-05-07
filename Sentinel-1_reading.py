import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pyproj import Transformer


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

era5_precip_train = "../../Data/ERA5/ERA5_2015_2022_Precip_Daily.tif"


era5_precip_train_data, era5_precip_train_transform = read_tif(era5_precip_train)
s1_vv_train_data, s1_vv_train_transform = read_tif(s1_vv_train)

print(s1_vv_train_data.shape)

era5_row, era5_col = get_pixels_values(pt_lat, pt_lon, era5_precip_train_transform)
s1_row, s1_col = get_pixels_values_s1(pt_lat, pt_lon, s1_vv_train_transform)

print(s1_row, s1_col)
print(era5_row, era5_col)

era5_train_precip = era5_precip_train_data[:, era5_row, era5_col]
s1_train_vv = s1_vv_train_data[:, s1_row, s1_col]

# print(s1_train_vv)

train_dates = pd.date_range(start="2015-01-01", end="2022-12-31", freq="D")

train_df = pd.DataFrame({
    "Date" : train_dates,
    "Precip" : era5_train_precip
})


train_df = train_df[train_df["Date"] >= '2016-01-29']


with rasterio.open(s1_vv_train) as src:
    data = src.read()
    original_band_names = src.descriptions
    transform = src.transform
    profile = src.profile
    s1_crs = src.crs

s1_dates = pd.to_datetime(original_band_names)

print(s1_dates)

precip_agg = []

for i in range(len(s1_dates)):
    if i == 0:
        start_date = train_df["Date"].min()
    else:
        start_date = s1_dates[i - 1] + pd.Timedelta(days=1)

    end_date = s1_dates[i]

    mask = (train_df["Date"] >= start_date) & (train_df["Date"] <= end_date)
    total_precip = train_df.loc[mask, "Precip"].sum()
    precip_agg.append(total_precip)

agg_df = pd.DataFrame({
    "Date": s1_dates,
    "VV": s1_train_vv,
    "Precip": precip_agg
})

print(agg_df.head())


# print(s1_crs)
# print(original_band_names)
# dates = [name.split('_')[-1] for name in original_band_names]
# dates = pd.to_datetime(dates, errors='coerce')
#
# print(dates)
#
# valid_indices = [i for i, d in enumerate(dates) if pd.notnull(d)]
# dates = [dates[i] for i in valid_indices]
# data = data[valid_indices, :, :]
# print(data)
#
#
# sorted_indices = np.argsort(dates)
# dates_sorted = [dates[i] for i in sorted_indices]
# data_sorted = data[sorted_indices, :, :]
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
# output_path = "../../Data/Sentinel-1/S1_VV_2023_2024_sorted.tif"
#
#
# with rasterio.open(output_path, "w", **profile) as dst:
#     for i in range(data_sorted.shape[0]):
#         dst.write(data_sorted[i, :, :], i + 1)
#         dst.set_band_description(i + 1, dates_sorted[i].strftime("%Y-%m-%d"))
#
# print("Saved sorted GeoTIFF to:", output_path)


