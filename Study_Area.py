import geopandas as gpd
import matplotlib.pyplot as plt

file = "../../Data/Study_Area/Study_Area.geojson"

study_area = gpd.read_file(file)

print(study_area)


fig, ax = plt.subplots(figsize=(8, 6))
study_area.plot(ax=ax, color='blue', markersize=10, edgecolor='black')

ax.set_title("SA", fontsize=12)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.grid(True)

plt.show()
