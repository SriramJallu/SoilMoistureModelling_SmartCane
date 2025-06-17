import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import matplotlib.pyplot as plt


def read_tif(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        meta = src.meta
        bands = src.descriptions

    return data, meta, bands


def get_valid_dates(bands, suffix):
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


sm_downscaled_path = "../../Data/SMAP/SMAP_downscaled_appraoch1_smap_test.tif"
lat1, lon1 = 52.26582, 6.78917
lat3, lon3 = 52.330112, 6.418003
lat5, lon5 = 52.171540, 6.488395

sm_data, sm_meta, sm_bands = read_tif(sm_downscaled_path)
print(sm_data.shape)

transform = sm_meta["transform"]
crs = sm_meta["crs"]
transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
x1, y1 = transformer.transform(lon1, lat1)
row1, col1 = rowcol(transform, x1, y1)
print("Pixel location:", row1, col1)

x3, y3 = transformer.transform(lon3, lat3)
row3, col3 = rowcol(transform, x3, y3)
print("Pixel location:", row3, col3)

x5, y5 = transformer.transform(lon5, lat5)
row5, col5 = rowcol(transform, x5, y5)
print("Pixel location:", row5, col5)

sm_data1 = sm_data[:, row1, col1]
sm_data3 = sm_data[:, row3, col3]
sm_data5 = sm_data[:, row5, col5]

sm_dates = sorted(get_valid_dates(sm_bands, "SM"))

sm_dates = pd.to_datetime(sm_dates, format="%Y_%m_%d")
sm_series1 = pd.Series(sm_data1, index=sm_dates)
sm_series3 = pd.Series(sm_data3, index=sm_dates)
sm_series5 = pd.Series(sm_data5, index=sm_dates)
sm_dates = sm_dates[sm_dates >= "2018-01-01"]
sm_series1 = sm_series1[sm_series1.index >= "2018-01-01"]
sm_series3 = sm_series3[sm_series3.index >= "2018-01-01"]
sm_series5 = sm_series5[sm_series5.index >= "2018-01-01"]


precip_era5_path = "../../Data/ERA5_NL/ERA5_2015_2020_Precip_NL_Daily.tif"
precip_data, precip_meta, precip_bands = read_tif(precip_era5_path)
precip_data = precip_data * 1000

sm_new_dates = sorted(get_valid_dates(sm_bands, "SM"))
precip_dates = get_valid_dates(precip_bands, "Precip")
precip_data, precip_bands_filt = filter_by_dates(precip_data, precip_bands, "Precip", sm_new_dates)
print(precip_data.shape)

sm_new_dates = pd.to_datetime(sm_new_dates, format="%Y_%m_%d")
transform2 = precip_meta["transform"]
crs = precip_meta["crs"]

transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
x1, y1 = transformer.transform(lon1, lat1)
row1, col1 = rowcol(transform2, x1, y1)
print("Pixel location:", row1, col1)

x3, y3 = transformer.transform(lon3, lat3)
row3, col3 = rowcol(transform2, x3, y3)
print("Pixel location:", row3, col3)

x5, y5 = transformer.transform(lon5, lat5)
row5, col5 = rowcol(transform2, x5, y5)
print("Pixel location:", row5, col5)


precip_data1 = precip_data[:, row1, col1]
precip_data3 = precip_data[:, row3, col3]
precip_data5 = precip_data[:, row5, col5]

precip_series1 = pd.Series(precip_data1, index=sm_new_dates)
precip_series3 = pd.Series(precip_data3, index=sm_new_dates)
precip_series5 = pd.Series(precip_data5, index=sm_new_dates)

precip_series1 = precip_series1[precip_series1.index >= "2018-01-01"]
precip_series3 = precip_series3[precip_series3.index >= "2018-01-01"]
precip_series5 = precip_series5[precip_series5.index >= "2018-01-01"]


fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

# axs[0].plot(sm_dates, sm_series1, label="Location 1 Built-up", linewidth=2)
axs[0].plot(sm_dates, sm_series3, label="Location 3 Forest", linewidth=2)
axs[0].plot(sm_dates, sm_series5, label="Location 5 Field", linewidth=2)
axs[0].set_ylabel("Soil Moisture", fontsize=12)
axs[0].set_title("Soil Moisture Predictions", fontsize=14)
plt.legend()

axs[1].plot(sm_dates, precip_series3, label="Location 3 Forest", linewidth=2)
axs[1].plot(sm_dates, precip_series5, label="Location 5 Field", linewidth=2)
axs[1].set_xlabel("Date", fontsize=12)
axs[1].set_ylabel("Precipitation (mm)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()
