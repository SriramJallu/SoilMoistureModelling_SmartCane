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

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)

gssm_path = "../../Data/GSSM/GSSM_2016_2020_SM_Daily_1km.tif"
smap_sm_path = "../../Data/SMAP/SMAP_2016_2022_SM_Daily.tif"
era5_paths = [
    "../../Data/ERA5/ERA5_2015_2022_Precip_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_Temp_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_RH_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_WindSpeed_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_Radiation_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_ET_Daily.tif",
    "../../Data/ERA5/ERA5_2015_2022_SoilTemp_Daily.tif"
]


def read_tif(tif):
    with rasterio.open(tif) as src:
        data = src.read().astype(np.float32)
    return data


def stack_weather(tif):
    feature_array = [read_tif(i) for i in tif]
    return np.stack(feature_array, axis=-1)


def generate_dates(start_year, end_year):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    return [(start + timedelta(days=i)).strftime('%Y%m%d') for i in range((end - start).days + 1)]


def upsample(data_10k, target_shape):
    return np.stack([
        resize(band, target_shape, order=1, preserve_range=True)
        for band in data_10k
    ])


era5_data = stack_weather(era5_paths)
gssm_data = read_tif(gssm_path)
smap_data = read_tif(smap_sm_path)

era5_dates = generate_dates(2015, 2022)
gssm_dates = generate_dates(2016, 2020)
smap_dates = generate_dates(2016, 2022)

target_shape = gssm_data.shape[1:]
print(target_shape)

era5_1km = np.stack(
    [upsample(era5_data[:, :, :, i], target_shape) for i in range(era5_data.shape[-1])],
    axis=-1
)

smap_1km = upsample(smap_data, target_shape)

print(era5_1km.shape)
print(smap_1km.shape)


feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

T, H, W, F = era5_1km.shape
era5_flat = era5_1km.reshape(T*H*W, F)
smap_flat = smap_1km.reshape(T*H*W, 1)

X_stack = np.hstack([era5_flat, smap_flat])
X_scaled = feature_scaler.fit_transform(X_stack).reshape(T, H, W, F+1)

y_flat = gssm_data.reshape(-1, 1)
y_scaled = target_scaler.fit_transform(y_flat).reshape(gssm_data.shape)



