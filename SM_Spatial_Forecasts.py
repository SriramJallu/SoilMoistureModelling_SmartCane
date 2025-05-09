import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pyproj import Transformer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)

era5_precip_train = "../../Data/ERA5/ERA5_2015_2022_Precip_Daily.tif"
era5_precip_test = "../../Data/ERA5/ERA5_2023_2024_Precip_Daily.tif"
era5_temp_train = "../../Data/ERA5/ERA5_2015_2022_Temp_Daily.tif"
era5_temp_test = "../../Data/ERA5/ERA5_2023_2024_Temp_Daily.tif"
era5_rh_train = "../../Data/ERA5/ERA5_2015_2022_RH_Daily.tif"
era5_rh_test = "../../Data/ERA5/ERA5_2023_2024_RH_Daily.tif"
era5_windspeed_train = "../../Data/ERA5/ERA5_2015_2022_WindSpeed_Daily.tif"
era5_windspeed_test = "../../Data/ERA5/ERA5_2023_2024_WindSpeed_Daily.tif"
era5_radiation_train = "../../Data/ERA5/ERA5_2015_2022_Radiation_Daily.tif"
era5_radiation_test = "../../Data/ERA5/ERA5_2023_2024_Radiation_Daily.tif"
era5_et_train = "../../Data/ERA5/ERA5_2015_2022_ET_Daily.tif"
era5_et_test = "../../Data/ERA5/ERA5_2023_2024_ET_Daily.tif"
era5_soiltemp_train = "../../Data/ERA5/ERA5_2015_2022_SoilTemp_Daily.tif"
era5_soiltemp_test = "../../Data/ERA5/ERA5_2023_2024_SoilTemp_Daily.tif"

smap_sm_train = "../../Data/SMAP/SMAP_2016_2022_SM_Daily.tif"
smap_sm_test = "../../Data/SMAP/SMAP_2023_2024_SM_Daily.tif"


def read_stack(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def stack_features(*arrays):
    return np.stack(arrays, axis=-1)


def create_sequences(X, Y, seq_len=20):
    Xs, Ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        Ys.append(Y[i+seq_len])
    return np.array(Xs), np.array(Ys)


precip = read_stack(era5_precip_train)[365:]
temp = read_stack(era5_temp_train)[365:]
rh = read_stack(era5_rh_train)[365:]
windspeed = read_stack(era5_windspeed_train)[365:]
radiation = read_stack(era5_radiation_train)[365:]
et = read_stack(era5_et_train)[365:]
soiltemp = read_stack(era5_soiltemp_train)[365:]

features_train = stack_features(precip, temp, rh, windspeed, radiation, et, soiltemp)
print(features_train.shape)

sm = read_stack(smap_sm_train)
target_train = np.expand_dims(sm, axis=-1)

print(target_train.shape)


precip_t = read_stack(era5_precip_test)
temp_t = read_stack(era5_temp_test)
rh_t = read_stack(era5_rh_test)
windspeed_t = read_stack(era5_windspeed_test)
radiation_t = read_stack(era5_radiation_test)
et_t = read_stack(era5_et_test)
soiltemp_t = read_stack(era5_soiltemp_test)

features_test = stack_features(precip_t, temp_t, rh_t, windspeed_t, radiation_t, et_t, soiltemp_t)
print(features_test.shape)


sm_t = read_stack(smap_sm_test)
target_test = np.expand_dims(sm_t, axis=-1)
print(target_test.shape)


X_train, y_train = create_sequences(features_train, target_train, 30)
X_test, y_test = create_sequences(features_test, target_test, 30)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

model = tf.keras.Sequential([
    tf.keras.layers.ConvLSTM2D()
])

