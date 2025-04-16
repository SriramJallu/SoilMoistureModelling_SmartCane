import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score, mean_squared_error
import xarray as xr
import h5py
import rasterio
from rasterio.transform import from_origin

filename = "../../Data/OMI_Test/OMI.nc"


with h5py.File(filename, 'r') as f:
    # Access the datasets
    lat = f['BAND2_ANCILLARY/STANDARD_MODE/GEODATA/latitude'][:]
    lon = f['BAND2_ANCILLARY/STANDARD_MODE/GEODATA/longitude'][:]
    ts = f['BAND2_ANCILLARY/STANDARD_MODE/ANCDATA/TS'][:]

print("Latitude shape:", lat.shape)
print("Longitude shape:", lon.shape)
print("Temperature shape:", ts.shape)

lat_max = np.max(lat)
lat_min = np.min(lat)
lon_max = np.max(lon)
lon_min = np.min(lon)

pixel_size = np.abs(lon[1] - lon[0])


transform = from_origin(lon_min, lat_max, pixel_size, pixel_size)

# Define the metadata for the GeoTIFF
metadata = {
    'driver': 'GTiff',
    'count': 1,
    'dtype': 'float32',
    'crs': 'EPSG:4326',
    'width': len(lon),
    'height': len(lat),
    'transform': transform
}

# Output file path
output_filename = "../../Data/OMI_temperature.tif"

# Create the GeoTIFF file
with rasterio.open(output_filename, 'w', **metadata) as dst:
    dst.write(ts, 1)

print(f"GeoTIFF file saved as {output_filename}")

# df = pd.read_csv("../../Data/Chemba_loc1_OpenMeteo_API_01012019_30122024_Daily.csv", skiprows=3)
# df["time"] = pd.to_datetime(df["time"])
# df.set_index("time", inplace=True)
#
# ts = df["soil_moisture_7_to_28cm_mean (m³/m³)"].dropna()
#
# result = adfuller(ts)
# print(f"ADF Statistic: {result[0]:.4f}")
# print(f"p-value: {result[1]:.4f}")
#
# model = SARIMAX(ts,
#                 order=(2,1,2),
#                 seasonal_order=(1,1,1,365),
#                 enforce_stationarity=False,
#                 enforce_invertibility=False)
#
# results = model.fit()
# print(results.summary())
