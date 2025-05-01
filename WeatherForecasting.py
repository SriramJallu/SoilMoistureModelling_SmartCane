import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import random

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


def read_tif(filename):
    with rasterio.open(filename) as f:
        data = f.read()
        return data, f.transform


def get_pixels_values(lat, lon, transform):
    row, col = ~transform*(lon, lat)
    return int(row), int(col)

# Chemba
pt_lat, pt_lon = -17.331524, 34.954147

########################################################################################################################
era5_precip_train_data, era5_precip_train_transform = read_tif(era5_precip_train)
era5_precip_test_data, era5_precip_test_transform = read_tif(era5_precip_test)

era5_temp_train_data, era5_temp_train_transform = read_tif(era5_temp_train)
era5_temp_test_data, era5_temp_test_transform = read_tif(era5_temp_test)

era5_rh_train_data, era5_rh_train_transform = read_tif(era5_rh_train)
era5_rh_test_data, era5_rh_test_transform = read_tif(era5_rh_test)

era5_windspeed_train_data, era5_windspeed_train_transform = read_tif(era5_windspeed_train)
era5_windspeed_test_data, era5_windspeed_test_transform = read_tif(era5_windspeed_test)

era5_radiation_train_data, era5_radiation_train_transform = read_tif(era5_radiation_train)
era5_radiation_test_data, era5_radiation_test_transform = read_tif(era5_radiation_test)

########################################################################################################################
## Filtering train data
era5_row, era5_col = get_pixels_values(pt_lat, pt_lon, era5_temp_train_transform)

era5_train_precip = era5_precip_train_data[:, era5_row, era5_col]
era5_train_temp = era5_temp_train_data[:, era5_row, era5_col]
era5_train_rh = era5_rh_train_data[:, era5_row, era5_col]
era5_train_windspeed = era5_windspeed_train_data[:, era5_row, era5_col]
era5_train_radiation = era5_radiation_train_data[:, era5_row, era5_col]

########################################################################################################################
## Filtering test data
era5_test_precip = era5_precip_test_data[:, era5_row, era5_col]
era5_test_temp = era5_temp_test_data[:, era5_row, era5_col]
era5_test_rh = era5_rh_test_data[:, era5_row, era5_col]
era5_test_windspeed = era5_windspeed_test_data[:, era5_row, era5_col]
era5_test_radiation = era5_radiation_test_data[:, era5_row, era5_col]

train_dates = pd.date_range(start="2015-01-01", end="2022-12-31", freq="D")
test_dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")

temp_train_df = pd.DataFrame({
    "Date" : train_dates,
    "Precip" : era5_train_precip,
    "Temp" : era5_train_temp,
    "RH" : era5_train_rh,
    "WindSpeed" : era5_train_windspeed,
    "Radiation" : era5_train_radiation
})

temp_test_df = pd.DataFrame({
    "Date" : test_dates,
    "Precip" : era5_test_precip,
    "Temp" : era5_test_temp,
    "RH" : era5_test_rh,
    "WindSpeed" : era5_test_windspeed,
    "Radiation" : era5_test_radiation
})

temp_train_df = temp_train_df.dropna(subset=["Precip", "Temp", "RH", "WindSpeed", "Radiation"])

features = temp_train_df[["Precip", "Temp", "RH", "WindSpeed", "Radiation"]].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)


def create_sequences(data, input_len=30, output_len=7):
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len].flatten())
    return np.array(X), np.array(y)


X, y = create_sequences(scaled_data, input_len=30, output_len=7)

model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(30, 5))),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(35, activation='relu')
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)


def monte_carlo_predict(model, X, num_samples=100):
    predictions = np.zeros((num_samples, X.shape[0], 35))
    for i in range(num_samples):
        predictions[i] = model(X, training=True)
    return predictions


last_30 = scaled_data[-30:].reshape(1, 30, 5)
predictions_mc = monte_carlo_predict(model, last_30, num_samples=200)


mean_preds = np.mean(predictions_mc, axis=0)
lower_bound = np.percentile(predictions_mc, 2.5, axis=0)
upper_bound = np.percentile(predictions_mc, 97.5, axis=0)

mean_preds = scaler.inverse_transform(mean_preds.reshape(7, 5))
lower_bound = scaler.inverse_transform(lower_bound.reshape(7, 5))
upper_bound = scaler.inverse_transform(upper_bound.reshape(7, 5))

print(f"Predicted Precipitation (Mean): {mean_preds}")
print(f"Lower Bound (2.5%): {lower_bound}")
print(f"Upper Bound (97.5%): {upper_bound}")


temp_test_df["Date"] = pd.to_datetime(temp_test_df["Date"])
test_features = temp_test_df[["Precip", "Temp", "RH", "WindSpeed", "Radiation"]].values
test_scaled = scaler.transform(test_features)

X_test, y_test = create_sequences(test_scaled, input_len=30, output_len=7)

predictions_mc_test = monte_carlo_predict(model, X_test, num_samples=200)

mean_preds_test = np.mean(predictions_mc_test, axis=0)
lower_bound_test = np.percentile(predictions_mc_test, 2.5, axis=0)
upper_bound_test = np.percentile(predictions_mc_test, 97.5, axis=0)

mean_preds_test_inv = scaler.inverse_transform(mean_preds_test.reshape(-1, 5))
lower_bound_test_inv = scaler.inverse_transform(lower_bound_test.reshape(-1, 5))
upper_bound_test_inv = scaler.inverse_transform(upper_bound_test.reshape(-1, 5))
y_true_inv = scaler.inverse_transform(y_test.reshape(-1, 5))

mean_preds_test = mean_preds_test_inv.reshape(-1, 35)
lower_bound_test = lower_bound_test_inv.reshape(-1, 35)
upper_bound_test = upper_bound_test_inv.reshape(-1, 35)
y_true = y_true_inv.reshape(-1, 35)


feature_names = ["Precip", "Temp", "RH", "WindSpeed", "Radiation"]

for day in range(7):
    print(f"--- Day {day+1} ---")
    for feat_idx, feat_name in enumerate(feature_names):
        idx = day * 5 + feat_idx
        r2 = r2_score(y_true[:, idx], mean_preds_test[:, idx])
        rmse = np.sqrt(mean_squared_error(y_true[:, idx], mean_preds_test[:, idx]))
        print(f"{feat_name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")


fig, axs = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
for i in range(5):
    axs[i].plot(y_true[:, i], label="Actual", color="black")
    axs[i].plot(mean_preds_test[:, i], label="Predicted (Mean)", color="green")
    axs[i].fill_between(
        np.arange(len(mean_preds_test)),
        lower_bound_test[:, i],
        upper_bound_test[:, i],
        color="gray",
        alpha=0.5,
        label="95% CI"
    )
    axs[i].set_title(f"Day 1 - {feature_names[i]} Forecast")
    axs[i].set_ylabel(feature_names[i])
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Time step")
plt.tight_layout()
plt.show()
