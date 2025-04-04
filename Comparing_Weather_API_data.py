import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


# locs = "../Data/SmartCane_Farms.geojson"
#
# gdf = gpd.read_file(locs)
#
# print(gdf)

om_daily = pd.read_csv("../../Data/Chemba_OpenMeteo_API_01012023_30122024_Daily.csv")
print(om_daily.tail(5))

om_hrs = pd.read_csv("../../Data/Chemba_OpenMeteo_API_01012023_30122024_Hourly.csv", header = None)

om_hrs.columns = om_hrs.iloc[3].tolist()

om_hrs = om_hrs.iloc[4:].reset_index(drop=True)

om_hrs["time"] = pd.to_datetime(om_hrs["time"], format="%Y-%m-%dT%H:%M", errors="coerce")
print(om_hrs["time"].dtype)

om_hrs.set_index("time", inplace=True)

om_hrs = om_hrs.apply(pd.to_numeric)
columns_to_sum = ["rain (mm)", "precipitation (mm)"]
columns_to_mean = [col for col in om_hrs.columns if col not in columns_to_sum]

om_hrs_to_daily = om_hrs.resample("D").agg({**{col: "sum" for col in columns_to_sum},
                                          **{col: "mean" for col in columns_to_mean}})

om_hrs_to_daily = om_hrs_to_daily.reset_index()

print(om_hrs_to_daily.tail(5))

print("###############################")

# print(om_hrs_to_daily.shape)

cols_to_add = ["relative_humidity_2m (%)", "et0_fao_evapotranspiration (mm)", "wind_speed_10m (km/h)", "soil_temperature_0_to_7cm (°C)", "soil_moisture_0_to_7cm (m³/m³)"]

om_daily["time"] = pd.to_datetime(om_daily["time"])
om_hrs_to_daily["time"] = pd.to_datetime(om_hrs_to_daily["time"])


om_daily = pd.merge(om_daily, om_hrs_to_daily[["time"] + cols_to_add], on="time", how="inner")
print(om_daily.shape)

# om_daily[cols_to_add] = om_hrs_to_daily[cols_to_add].values
print("###############################")

vc_daily = pd.read_csv("../../Data/Chemba_VisualCrossing_01012024_31122024.csv")
vc_daily = vc_daily.drop(columns = "name")

vc_daily_23 = pd.read_csv("../../Data/Chemba_VisualCrossing_01012023_31122023.csv")
vc_daily_23 = vc_daily_23.drop(columns = "name")


vc_daily = pd.concat([vc_daily_23, vc_daily], axis=0, ignore_index=True)

print(vc_daily.shape)
print("#################################################################################################################")

om_daily["time"] = pd.to_datetime(om_daily["time"], errors='coerce')
vc_daily["time"] = pd.to_datetime(vc_daily["datetime"], errors='coerce')

row_dec31 = vc_daily[vc_daily["datetime"] == "2024-12-31"]
print("vc_31_dec:")
print(row_dec31)

# a = om_daily["precipitation_sum (mm)"]
# b = vc_daily["precip"]
#
# mask = a.notna() & b.notna()
#
# om_daily = om_daily[mask]
# vc_daily = vc_daily[mask]
#
# om_daily = om_daily[om_daily['time'].isin(vc_daily['time'])]
# vc_daily = vc_daily[vc_daily['time'].isin(om_daily['time'])]
#

missing_dates = ["2023-12-31", "2024-12-31"]

source_dates = ["2023-12-30", "2024-12-30"]

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

    plt.plot(df1["time"], df1[col1], label=lab1)
    plt.plot(df2["time"], df2[col2], label=lab2)
    plt.legend()
    plt.title(f"Comparison of {col1} between {lab1} and {lab2}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


compare_dfs(om_daily, vc_daily, "precipitation_sum (mm)", "precip", "om", "vc")



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

    plt.figure(figsize=(12, 10))

    # Loop through each dataframe and decompose
    for df, col, lab in zip(dfs, cols, labs):
        result = seasonal_decompose(df[col], period=period, model="additive")

        # Plot the Trend component
        plt.subplot(3, 1, 1)
        plt.plot(result.trend, label=f"{lab}", linestyle='-', linewidth=2)

        # Plot the Seasonal component
        plt.subplot(3, 1, 2)
        plt.plot(result.seasonal, label=f"{lab}", linestyle='-', linewidth=2)

        # Plot the Residual component
        plt.subplot(3, 1, 3)
        plt.plot(result.resid, label=f"{lab}", linestyle='-', linewidth=2)

    # Add titles and legends
    plt.subplot(3, 1, 1)
    plt.title("Trend Component")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.title("Seasonal Component")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.title("Residual Component")
    plt.legend()

    plt.tight_layout()
    plt.show()

time_series_decompose(om_daily, vc_daily, cols=["precipitation_sum (mm)", "precip"], labs = ["OM", "VC"], period=30)




# fig, ax = plt.subplots(figsize=(8, 6))
# gdf.plot(ax=ax, color='blue', markersize=10, edgecolor='black')
#
# ax.set_title("MultiPoints", fontsize=12)
# ax.set_xlabel("Longitude")
# ax.set_ylabel("Latitude")
# plt.grid(True)
#
# plt.show()