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


sm_downscaled_path = "../../Data/SMAP/SMAP_downscaled_appraoch1_smap_test.tif"
lat, lon = 52.24840, 6.78723

sm_data, sm_meta, sm_bands = read_tif(sm_downscaled_path)
print(sm_data.shape)

transform = sm_meta["transform"]
crs = sm_meta["crs"]
transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
x, y = transformer.transform(lon, lat)
row, col = rowcol(transform, x, y)
print("Pixel location:", row, col)

sm_data = sm_data[:, row, col]

sm_dates = sorted(get_valid_dates(sm_bands, "SM"))

sm_dates = pd.to_datetime(sm_dates, format="%Y_%m_%d")
sm_series = pd.Series(sm_data, index=sm_dates)

plt.plot(sm_dates, sm_series, label="SM Time seires", linewidth=2)
plt.legend()
plt.show()
