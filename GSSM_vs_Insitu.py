import rasterio
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from pyproj import Transformer
from rasterio.transform import rowcol, xy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


def read_tif(tif):
    with rasterio.open(tif) as src:
        data = src.read().astype(np.float32)
        meta = src.meta
        bands = src.descriptions
    return data, meta, bands


et_path = "../../Data/MODIS/MODIS_ET_Daily_2015_2020.tif"

et_data, et_meta, et_bands = read_tif(et_path)
print(et_data.shape)

print(et_bands[:5])
print(et_meta)

sm_api_path = "../../Data/dataverse_files/Loc_10_API.csv"
headers_api = pd.read_csv(sm_api_path, skiprows=3, nrows=0).columns.tolist()
sm_test = pd.read_csv(sm_api_path, skiprows=4, parse_dates=["time"], names=headers_api)
sm_test["time"] = pd.to_datetime(sm_test["time"], format='%Y-%m-%d', errors='coerce')
sm_test = sm_test.set_index("time")
print(sm_test.head())

# daily_et = {}
#
# for i, bands in enumerate(et_bands):
#     band_date = datetime.strptime(bands.split('_ET')[0], "%Y_%m_%d")
#     year = band_date.year
#
#     if i == len(et_bands) - 1 or datetime.strptime(et_bands[i + 1].split('_ET')[0], "%Y_%m_%d").year != year:
#         end_of_year = datetime(year, 12, 31)
#         days = (end_of_year - band_date).days + 1
#     else:
#         days = 8
#
#     et_per_day = et_data[i] / days
#
#     for d in range(days):
#         day = band_date + timedelta(days=d)
#         daily_et[day] = et_per_day.copy()
#
# sorted_dates = sorted(daily_et.keys())
#
# stacked_data = np.stack([daily_et[date] for date in sorted_dates])
#
# multiband_meta = et_meta.copy()
# multiband_meta.update({
#     "count": len(sorted_dates),
#     "dtype": "float32"
# })
#
# output_path = "../../Data/MODIS/MODIS_ET_Daily_2015_2020.tif"
#
# with rasterio.open(output_path, "w", **multiband_meta) as dst:
#     for i in range(len(sorted_dates)):
#         dst.write(stacked_data[i], i + 1)
#         dst.set_band_description(i + 1, sorted_dates[i].strftime("%Y_%m_%d") + "_ET")



# gssm_sm_path = "../../Data/GSSM/GSSM_2016_2020_SM_NL_Daily_1km.tif"
# sm_test_path = "../../Data/dataverse_files/1_station_measurements/2_calibrated/ITCSM_11_cd.csv"
#
# gssm_data, gssm_meta, gssm_bands = read_tif(gssm_sm_path)
# gssm_data = gssm_data[366-9:]
#
#
# lat, lon = 52.23111, 6.55889
#
# transform = gssm_meta["transform"]
# crs = gssm_meta["crs"]
# transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
#
#
# x, y = transformer.transform(lon, lat)
# row, col = rowcol(transform, x, y)
# print("Pixel location:", row, col)
#
# gssm_series = gssm_data[:, row, col]
#
#
# headers = pd.read_csv(sm_test_path, skiprows=21, nrows=0).columns.tolist()
# sm_test = pd.read_csv(sm_test_path, skiprows=23, parse_dates=["Date time"], names=headers)
#
# sm_test["Date time"] = pd.to_datetime(sm_test["Date time"], format='%d-%m-%Y %H:%M', errors='coerce')
# sm_test = sm_test[sm_test["Date time"] >= '2017-01-01']
# sm_test = sm_test[sm_test[" 5 cm SM"] >= 0]
# sm_test = sm_test.set_index("Date time")
# sm_test = sm_test.resample("D").mean()
#
# date_range_2020 = pd.date_range(start="2017-01-01", periods=1461, freq="D")
# gssm_2020_series = pd.Series(gssm_series, index=date_range_2020)
# gssm_filtered = gssm_2020_series.reindex(sm_test.index)
#
# combined_df = pd.DataFrame({
#     "GSSM": gssm_filtered,
#     "insitu": sm_test[" 5 cm SM"]
# })
#
# combined_df = combined_df.dropna()
# rmse_insitu = mean_squared_error(combined_df["insitu"], combined_df["GSSM"], squared=False)
# r2_insitu = r2_score(combined_df["insitu"], combined_df["GSSM"])
#
# print(f"RMSE vs in-situ (2020): {rmse_insitu:.4f}")
# print(f"R² vs in-situ (2020): {r2_insitu:.4f}")
#
# plt.figure(figsize=(12, 5))
# plt.plot(combined_df.index, combined_df["insitu"], label="In-situ SM", linewidth=2)
# plt.plot(combined_df.index, combined_df["GSSM"], label="GSSM", linewidth=2)
# plt.xlabel("Date", fontsize=12)
# plt.ylabel("Soil Moisture", fontsize=12)
# plt.title("GSSM vs In-situ Measurements (2020)", fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
