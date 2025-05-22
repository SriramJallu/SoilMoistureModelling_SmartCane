import rasterio
from rasterio.transform import rowcol
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from pyproj import Transformer
import tensorflow as tf

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)


era5_train_paths = [
    "../../Data/ERA5_NL/ERA5_2016_2022_Precip_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2016_2022_Temp_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2016_2022_RH_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2016_2022_WindSpeed_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2016_2022_Radiation_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2016_2022_ET_NL_Daily.tif",
    "../../Data/ERA5_NL/ERA5_2016_2022_SoilTemp_NL_Daily.tif"
]


def read_tif(tif, lon, lat, startdate, enddate, label=None, plot=True):
    try:
        with rasterio.open(tif) as src:
            data = src.read().astype(np.float32)
            transform = src.transform
            crs = src.crs
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            row, col = rowcol(transform, x, y)
            era5_data = data[:, row, col]
            dates = pd.date_range(start=startdate, end=enddate, freq="D")
            df = pd.DataFrame({
                "Dates": dates,
                "ERA5": era5_data
            })
            df_20 = df[df["Dates"].dt.year.isin([2020])].copy()

            if plot and label:
                plt.figure(figsize=(14, 8))
                plt.plot(df_20["Dates"], df_20["ERA5"], label=label)
                plt.title(f"Daily {label}")
                plt.xlabel("Date")
                plt.ylabel(f"{label}")
                plt.legend()
                plt.tight_layout()
                plt.show()
            return era5_data
    except Exception as e:
        print(f"Error reading {tif}: {e}")
        return None


lon, lat = 6.69944, 52.27333
startdate, enddate = "2016-01-01", "2022-12-31"

precip_plot = read_tif(era5_train_paths[0], lon, lat, startdate, enddate, "Precipitation (mm)")
temp_plot = read_tif(era5_train_paths[1], lon, lat, startdate, enddate, "Temperature (°C)")
