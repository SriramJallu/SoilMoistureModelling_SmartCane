import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

om_daily = pd.read_csv("../../Data/Chemba_loc1_OpenMeteo_API_01012021_30122024_Daily.csv", skiprows=3)
om_daily["time"] = pd.to_datetime(om_daily["time"])

om_daily_loc2 = pd.read_csv("../../Data/Chemba_loc2_OpenMeteo_API_01012021_30122024_Daily.csv", skiprows=3)
om_daily_loc2["time"] = pd.to_datetime(om_daily_loc2["time"])

print("###############################")

vc_daily = pd.read_csv("../../Data/Chemba_VisualCrossing_01012024_31122024.csv")
vc_daily = vc_daily.drop(columns = "name")

vc_daily_23 = pd.read_csv("../../Data/Chemba_VisualCrossing_01012023_31122023.csv")
vc_daily_23 = vc_daily_23.drop(columns = "name")

vc_daily_22 = pd.read_csv("../../Data/Chemba_VisualCrossing_01012022_31122022.csv")
vc_daily_22 = vc_daily_22.drop(columns = "name")

vc_daily_21 = pd.read_csv("../../Data/Chemba_VisualCrossing_01012021_31122021.csv")
vc_daily_21 = vc_daily_21.drop(columns = "name")

vc_daily = pd.concat([vc_daily_21, vc_daily_22, vc_daily_23, vc_daily], axis=0, ignore_index=True)

print(vc_daily.shape)
print("#################################################################################################################")

om_daily["time"] = pd.to_datetime(om_daily["time"], errors='coerce')
vc_daily["time"] = pd.to_datetime(vc_daily["datetime"], errors='coerce')

row_dec31 = vc_daily[vc_daily["datetime"] == "2024-12-31"]
print("vc_31_dec:")
print(row_dec31)

missing_dates = ["2024-12-31"]

source_dates = ["2024-12-30"]

for miss_date, src_date in zip(missing_dates, source_dates):
    row_to_copy = om_daily[om_daily["time"] == src_date]
    if not row_to_copy.empty:
        new_row = row_to_copy.copy()
        new_row["time"] = pd.to_datetime(miss_date)
        om_daily = pd.concat([om_daily, new_row], ignore_index=True)

om_daily = om_daily.sort_values("time").reset_index(drop=True)
print(om_daily.shape)

print(om_daily.columns)
print(vc_daily.columns)


def compare_dfs(df1, df2, col1, col2, lab1, lab2):
    diff = df1[col1] - df2[col2]

    print("Mean absolute difference:", diff.abs().mean())
    print("Max difference:", diff.max())
    print("Min difference:", diff.min())

    rmse = np.sqrt(mean_squared_error(df1[col1], df2[col2]))
    print("RMSE:", rmse)

    r2 = r2_score(df2[col2], df1[col1])
    print("R² score:", r2)

    correlation = df1[col1].corr(df2[col2])
    print(("Correlation:", correlation))

    # plt.plot(df1["time"], df1[col1], label=lab1)
    # plt.plot(df2["time"], df2[col2], label=lab2)
    # plt.legend()
    # plt.title(f"Comparison of {col1}")
    # plt.xlabel("Date")
    # plt.ylabel("Value")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()


compare_dfs(om_daily, vc_daily, "wind_speed_10m_mean (km/h)", "windspeed", "OpenMeteo", "VisualCrossing")



def adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")


# adf_test(om_daily['precipitation_sum (mm)'])
# adf_test(vc_daily['precip'])

def time_series_decompose(*dfs, cols, labs, period= 365):

    plt.figure(figsize=(12, 12))

    plt.subplot(4, 1, 1)
    for df, col, lab in zip(dfs, cols, labs):
        plt.plot(df["time"], df[col], label=lab)
    plt.title(f"Time Series Comparison: {cols[0]}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.legend()

    # Loop through each dataframe and decompose
    for df, col, lab in zip(dfs, cols, labs):
        result = seasonal_decompose(df[col], period=period, model="additive")

        time = df["time"]

        # Match time with non-NaN trend values
        valid_trend = result.trend.dropna()
        valid_trend_time = time[valid_trend.index]

        valid_resid = result.resid.dropna()
        valid_resid_time = time[valid_resid.index]


        plt.subplot(4, 1, 2)
        plt.plot(valid_trend_time, valid_trend, label=f"{lab}", linestyle='-', linewidth=2)

        plt.subplot(4, 1, 3)
        plt.plot(time, result.seasonal, label=f"{lab}", linestyle='-', linewidth=2)

        plt.subplot(4, 1, 4)
        plt.plot(valid_resid_time, valid_resid, label=f"{lab}", linestyle='-', linewidth=2)

    for i, title in enumerate(["Trend Component", "Seasonal Component", "Residual Component"], start=2):
        plt.subplot(4, 1, i)
        plt.title(title)
        plt.legend()
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


time_series_decompose(om_daily, om_daily_loc2, vc_daily, cols=["temperature_2m_mean (°C)", "temperature_2m_mean (°C)", "temp"],
                      labs=["OpenMeteo - loc1", "OpenMeteo - loc2", "VisualCrossing"], period=365)

time_series_decompose(om_daily, om_daily_loc2, cols=["et0_fao_evapotranspiration (mm)", "et0_fao_evapotranspiration (mm)"], labs=["OpenMeteo - loc1", "OpenMeteo - loc2"], period=365)

# fig, ax = plt.subplots(figsize=(8, 6))
# gdf.plot(ax=ax, color='blue', markersize=10, edgecolor='black')
#
# ax.set_title("MultiPoints", fontsize=12)
# ax.set_xlabel("Longitude")
# ax.set_ylabel("Latitude")
# plt.grid(True)
#
# plt.show()


# locs = "../Data/SmartCane_Farms.geojson"
#
# gdf = gpd.read_file(locs)
#
# print(gdf)
#
# def load_openmeteo_daily(base_path: str, cols_to_add=None):
#     """
#     Load and merge OpenMeteo daily and hourly CSVs into a single daily dataframe.
#
#     Parameters:
#         base_path (str): Base file path without suffix (_Daily.csv or _Hourly.csv)
#         cols_to_add (list of str): List of columns from hourly data to include in daily aggregation.
#                                    If None, a default list is used.
#
#     Returns:
#         pd.DataFrame: Merged daily DataFrame with aggregated hourly features added.
#     """
#     if cols_to_add is None:
#         cols_to_add = [
#             "relative_humidity_2m (%)",
#             "wind_speed_10m (km/h)",
#             "soil_temperature_0_to_7cm (°C)",
#             "soil_moisture_0_to_7cm (m³/m³)",
#             "wind_speed_100m (km/h)"
#         ]
#
#     # Load daily data
#     om_daily = pd.read_csv(base_path + "_Daily.csv")
#     om_daily["time"] = pd.to_datetime(om_daily["time"])
#
#     # Load hourly data
#     om_hrs = pd.read_csv(base_path + "_Hourly.csv", header=None)
#     om_hrs.columns = om_hrs.iloc[3].tolist()
#     om_hrs = om_hrs.iloc[4:].reset_index(drop=True)
#     om_hrs["time"] = pd.to_datetime(om_hrs["time"], format="%Y-%m-%dT%H:%M", errors="coerce")
#     om_hrs.set_index("time", inplace=True)
#     om_hrs = om_hrs.apply(pd.to_numeric)
#
#     # Resample to daily
#     columns_to_sum = ["rain (mm)", "precipitation (mm)"]
#     columns_to_mean = [col for col in om_hrs.columns if col not in columns_to_sum]
#     om_hrs_to_daily = om_hrs.resample("D").agg({**{col: "sum" for col in columns_to_sum},
#                                                 **{col: "mean" for col in columns_to_mean}})
#     om_hrs_to_daily = om_hrs_to_daily.reset_index()
#
#     # Merge hourly into daily
#     om_hrs_to_daily["time"] = pd.to_datetime(om_hrs_to_daily["time"])
#     om_daily = pd.merge(om_daily, om_hrs_to_daily[["time"] + cols_to_add], on="time", how="inner")
#
#     return om_daily


# om_daily = load_openmeteo_daily("../../Data/Chemba_OpenMeteo_API_01012022_30122024")
# om_daily_loc2 = load_openmeteo_daily("../../Data/Chemba_loc2_OpenMeteo_API_01012022_30122024")
