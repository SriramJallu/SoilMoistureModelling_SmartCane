# Soil Moisture Forecasting with Satellite-based weather data

- Satellite-based weather was compared against Weather API data to check its validity.
- "Weather_Variable".py files are scripts for comparing different sources of satellite-based weather data with weather API data.
- Spatial downscaling of SMAP Soil Moisture L3 product was conducted, from a coarse 10 km to 1 km resolution. This was achieved by using 1 km MODIS NDVI, LST, ET products.
- SM_Downscaling_Approach1_''.py and SM_Downscaling_Approach2_''.py are the corresponding scripts.
- Downscaled soil mositure was used, along with weather data for forecasting Soil Moisture. Two experiments were conducted here. Exp 1: Only historical weather data and soil moisture data was used. Exp 2: Along with these two, weather forecasts were also used in training the models.
- SM_Spatial_Forecasts_''.py: Scripts for experiment 1, SM_Spatial_Forecasts_Plus_WF_''.py: Scripts for experiment 2.