import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score, mean_squared_error
import h5py
from osgeo import gdal
import rasterio
from rasterio.transform import from_origin

filename = "../../Data/OMI_Test/OMI.nc"

with h5py.File(filename, 'r') as f:
    lat = f['BAND2_ANCILLARY/STANDARD_MODE/GEODATA/latitude'][:]
    lon = f['BAND2_ANCILLARY/STANDARD_MODE/GEODATA/longitude'][:]
    ts = f['BAND2_ANCILLARY/STANDARD_MODE/ANCDATA/TS'][:]

print("Latitude shape:", lat.shape)
print("Longitude shape:", lon.shape)
print("Temperature shape:", ts.shape)

lat = lat.squeeze()
lon = lon.squeeze()
ts = ts.squeeze()


print("Latitude shape (squeezed):", lat.shape)
print("Longitude shape (squeezed):", lon.shape)
print("Temperature shape (squeezed):", ts.shape)

data = []

for i in range(lat.shape[0]):
    for j in range(lon.shape[1]):
        latitude = lat[i, j]
        longitude = lon[i, j]
        temperature = ts[i, j]
        data.append([latitude, longitude, temperature])

df = pd.DataFrame(data, columns=['Latitude', 'Longitude', 'Temperature'])

# print(df.head())
df_cleaned = df[(df['Latitude'] >= -90) & (df['Latitude'] <= 90) &
                (df['Longitude'] >= -180) & (df['Longitude'] <= 180)]

min_lat = -17.489864
max_lat = -17.309857
min_lon = 33.713467
max_lon = 36.517824

df_subset = df_cleaned[
    (df_cleaned['Latitude'] >= min_lat) & (df_cleaned['Latitude'] <= max_lat) &
    (df_cleaned['Longitude'] >= min_lon) & (df_cleaned['Longitude'] <= max_lon)
]

print(f"Subset data size: {df_subset.shape}")

# print(df_subset)

lon = df_subset['Longitude'].values
lat = df_subset['Latitude'].values
temp = df_subset['Temperature'].values

lat_res_deg = 13 / 111.0       # ~0.117 degree
lon_res_deg = 25 / 111.0       # ~0.225 degree

# Now compute the pixel edges from the centers
lat_edges = np.unique(lat) - lat_res_deg / 2
lat_edges = np.append(lat_edges, lat_edges[-1] + lat_res_deg)

lon_edges = np.unique(lon) - lon_res_deg / 2
lon_edges = np.append(lon_edges, lon_edges[-1] + lon_res_deg)

# Create a 2D grid
lat_centers = np.unique(lat)
lon_centers = np.unique(lon)
lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
temp_grid = np.full(lon_grid.shape, np.nan)

# Fill grid
for i in range(len(temp)):
    lon_idx = np.where(lon_centers == lon[i])[0][0]
    lat_idx = np.where(lat_centers == lat[i])[0][0]
    temp_grid[lat_idx, lon_idx] = temp[i]


plt.figure(figsize=(10, 6))
mesh = plt.pcolormesh(lon_edges, lat_edges, temp_grid, cmap='inferno', shading='auto')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Temperature Raster")
plt.colorbar(mesh, label='Temperature')
plt.gca().set_aspect('equal')
plt.show()


# temperature_grid = df_subset.pivot(index='Latitude', columns='Longitude', values='Temperature')

# plt.figure(figsize=(10, 6))
# plt.imshow(temperature_grid, cmap='viridis', origin='upper', extent=[df['Longitude'].min(), df['Longitude'].max(), df['Latitude'].min(), df['Latitude'].max()])
# plt.colorbar(label='Temperature')
# plt.title('Temperature Raster over Latitude and Longitude')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()

# pixel_size_x_deg = 25 / 111.0
# pixel_size_y_deg = 13 / 111.0
#
# # Top-left corner coordinates
# top_left_lon = float(lon[0, 0])
# top_left_lat = float(lat[0, 0])
#
# # Create the affine transform
# transform = from_origin(top_left_lon, top_left_lat, pixel_size_x_deg, pixel_size_y_deg)
#
# print("Transform:", transform)
# print("Type:", type(transform))
#
# # Define the metadata for the GeoTIFF
# metadata = {
#     'driver': 'GTiff',
#     'count': 1,
#     'dtype': 'float32',
#     'crs': 'EPSG:4326',
#     'width': ts.shape[1],
#     'height': ts.shape[0],
#     'transform': transform
# }
#
# output_filename = "../../Data/OMI_temperature.tif"
#
# with rasterio.open(output_filename, 'w', **metadata) as dst:
#     dst.write(ts.astype('float32'), 1)
#
# print(f"GeoTIFF file saved as {output_filename}")

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
