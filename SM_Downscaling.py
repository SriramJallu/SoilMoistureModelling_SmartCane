import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from pyproj import Transformer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from skimage.transform import resize
from sklearn.ensemble import RandomForestRegressor
from rasterio.enums import Resampling

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)


smap_sm_path = "../../Data/SMAP/SMAP_2016_2022_SM_NL_Daily.tif"
smap_sm_am_path = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_AM_NL_Daily.tif"
smap_sm_pm_path = "../../Data/SMAP/SMAP_2016_2022_SoilMoisture_PM_NL_Daily.tif"
ndvi_path = "../../Data/VIIRS/VIIRS_NDVI_2016_2021_500m.tif"
lst_day_path = "../../Data/VIIRS/VIIRS_LST_Day_2017_2021_1km.tif"
lst_night_path = "../../Data/VIIRS/VIIRS_LST_Night_2017_2021_1km.tif"
dem_path = "../../Data/StaticVars/DEM_Map_90m.tif"
slope_path = "../../Data/StaticVars/Slope_Map_90m.tif"
soil_texture_path = "../../Data/StaticVars/SoilTexture_Map_250m.tif"


def read_tif(tif):
    with rasterio.open(tif) as src:
        data = src.read().astype(np.float32)
        meta = src.meta
        bands = src.descriptions
    return data, meta, bands


def fill_missing_data(data, max_window=28):
    T, W, H = data.shape
    data_reshaped = data.reshape(T, -1)
    filled = np.empty_like(data_reshaped)

    for idx in range(data_reshaped.shape[1]):
        series = pd.Series(data_reshaped[:, idx])
        series_filled = series.fillna(method="ffill", limit=max_window)
        filled[:, idx] = series_filled.to_numpy()

    return filled.reshape(T, W, H)


def resampling_data(src_data, src_meta, target_shape, resample=Resampling.bilinear):
    T, H, W = src_data.shape
    resampled = np.zeros((T, target_shape[0], target_shape[1]), dtype=np.float32)

    for t in range(T):
        with rasterio.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=H,
                width=W,
                count=1,
                dtype=src_data.dtype,
                crs=src_meta["crs"],
                transform=src_meta["transform"]
            ) as dataset:
                dataset.write(src_data[t, :, :], 1)

            with memfile.open() as dataset:
                resampled[t] = dataset.read(
                    out_shape=(1, target_shape[0], target_shape[1]),
                    resampling=resample
                )[0]
    return resampled


ndvi_data, ndvi_meta, ndvi_bands = read_tif(ndvi_path)
ndvi_data = ndvi_data[366:]
ndvi_bands = ndvi_bands[366:]
print(len(ndvi_bands))
print(ndvi_bands[:5])
print(ndvi_data.shape)

lst_day_data, lst_day_meta, lst_day_bands = read_tif(lst_day_path)
print(len(lst_day_bands))
print(lst_day_bands[:5])
print(lst_day_data.shape)

lst_night_data, lst_night_meta, lst_night_bands = read_tif(lst_night_path)
print(len(lst_night_bands))
print(lst_night_bands[:5])
print(lst_night_data.shape)

dem_data, dem_meta, dem_band = read_tif(dem_path)
slope_data, slope_meta, slope_band = read_tif(slope_path)
soil_texture_data, soil_texture_meta, soil_texture_band = read_tif(soil_texture_path)


ndvi_filled = fill_missing_data(ndvi_data, max_window=28)
lst_day_filled = fill_missing_data(lst_day_data, max_window=28)
lst_night_filled = fill_missing_data(lst_night_data, max_window=28)


smap_data, smap_meta, smap_bands = read_tif(smap_sm_path)
smap_shape = (smap_meta["height"], smap_meta["width"])
print(smap_shape)
smap_transform = smap_meta["transform"]
smap_crs = smap_meta["crs"]

ndvi_resampled = resampling_data(ndvi_filled, ndvi_meta, smap_shape)
lst_day_resampled = resampling_data(lst_day_filled, lst_day_meta, smap_shape)
lst_night_resampled = resampling_data(lst_day_filled, lst_night_meta, smap_shape)
print(ndvi_resampled.shape)
print(lst_day_resampled.shape)
print(lst_night_resampled.shape)


output_path = "../../Data/VIIRS/VIIRS_LST_Day_Resampled_2017_2021.tif"
T, H, W = lst_day_resampled.shape

new_meta = smap_meta.copy()
new_meta.update({
    "count": T,
    "dtype": "float32"
})

with rasterio.open(output_path, "w", **new_meta) as dst:
    for i in range(T):
        dst.write(lst_day_resampled[i, :, :], i + 1)

print("Done")

# original_series = lst_day_data[:, 3, 5]
# filled_series = lst_day_filled[:, 3, 5]
#
# days = np.arange(len(original_series))
#
# plt.figure(figsize=(12, 5))
# plt.plot(days, original_series, label='Original', color='red', linestyle='--', marker='*', markersize=5, alpha=0.6)
# plt.plot(days, filled_series, label='Filled (last 28 days)', color='blue', marker='o', markersize=3, alpha=0.6)
#
# plt.xlabel('Days since 2017-01-01')
# plt.ylabel('LST Day Temperature (K)')
# plt.title(f'LST Day at Pixel ({2}, {4})')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# def resample_to_match(source_array, source_meta, target_meta):
#     T = source_array.shape[0]
#     H_target, W_target = target_meta['height'], target_meta['width']
#     resampled = np.empty((T, H_target, W_target), dtype=np.float32)
#     for t in range(T):
#         reproject(
#             source=source_array[t],
#             destination=resampled[t],
#             src_transform=source_meta['transform'],
#             src_crs=source_meta['crs'],
#             dst_transform=target_meta['transform'],
#             dst_crs=target_meta['crs'],
#             resampling=Resampling.bilinear
#         )
#     return resampled
