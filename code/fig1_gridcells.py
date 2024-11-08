import geopandas as gpd
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

color_palette = ['#0072B2', '#D95F02', '#999999']
os.makedirs("../results", exist_ok=True)

def create_geodataframe(df):
    """Converts DataFrame to GeoDataFrame with geometry based on Lon and Lat columns."""
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Lon, df.Lat))

# Load gridlists
try:
    train_df = pd.read_csv(os.path.join("../gridlists", "training_gridlist.csv"))
    val_df = pd.read_csv(os.path.join("../gridlists", "val_gridlist.csv"))
    test_df = pd.read_csv(os.path.join("../gridlists", "test_gridlist.csv"))
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure the CSV files are in the 'gridlists' directory.")

# Convert Dataframes to GeoDataFrames
train_gdf = create_geodataframe(train_df)
val_gdf = create_geodataframe(val_df)
test_gdf = create_geodataframe(test_df)

# Plot
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
fig, ax = plt.subplots(figsize=(22, 18))
world.plot(ax=ax, color='white', edgecolor='black')
train_gdf.plot(ax=ax, marker='o', color=color_palette[2], markersize=30, label='Training Set')
val_gdf.plot(ax=ax, marker='o', color=color_palette[0], markersize=30, label='Validation Set')
test_gdf.plot(ax=ax, marker='o', color=color_palette[1], markersize=30, label='Test Set')
plt.title("Training, Validation, and Test Grid Cells", fontsize=40)
ax.set_xlabel('Longitude', fontsize=30)
ax.set_ylabel('Latitude', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(handles=[
    mpatches.Patch(color=color_palette[2], label='Training Set'),
    mpatches.Patch(color=color_palette[0], label='Validation Set'),
    mpatches.Patch(color=color_palette[1], label='Test Set')
], prop={'size': 18})
plt.tight_layout()
plt.savefig(os.path.join("../results/figures", "fig1.jpg"), dpi=300)

