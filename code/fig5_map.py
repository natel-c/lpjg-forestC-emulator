# use conda env: geopandas-env
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
import numpy as np
import os
import pandas as pd


os.makedirs("../results", exist_ok=True)

# Define global variables
DPI = 300
RES = 0.5  # LPJ-GUESS standard resolution

# These functions are needed to locate the predicted data on the lpj-guess data mesh.
# Find the longitude ID in a range from -180° to 180°
def find_lon_id(value, RES):
    return int((value + 180 - 0.25) / RES)

# Find the latitude ID in a range from -90° to 90°
def find_lat_id(value, res):
    return int((value + 90 - 0.25) / RES)

# This function writes the data onto the given mesh.
def populate_lpjg_grid(df, column_name):
    mesh = np.full((360, 720), np.nan)
    #df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        lon_id = find_lon_id(row['Lon'], RES)
        lat_id = find_lat_id(row['Lat'], RES)
        value = row[column_name]
        mesh[lat_id, lon_id] = value
    return mesh

# Function to calculate differences
def calculate_diff(df, target, percent_error=False):
    if percent_error:
        diff = (df[f'{target}_pred'] - df[f'{target}_true']) / df[f'{target}_true'] * 100
        vmax, vmin = 100, -100
    else:
        diff = df[f'{target}_pred'] - df[f'{target}_true']
        vmax, vmin = np.max(diff), np.min(diff)
    
    diff_df = pd.DataFrame(diff, columns=[target])
    diff_df['Lon'] = df['lon']
    diff_df['Lat'] = df['lat']
    
    return diff_df, vmin, vmax

def create_frame_diff(ax, data_mesh, cmap, vmin, vmax):
    # Plot the map
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    img = ax.imshow(data_mesh, origin='lower', extent=(-180, 180, -90, 90), 
                    transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)

    ax.add_feature(cfeature.LAND, facecolor='grey')
    ax.coastlines(linewidth=0.5)
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    return img

def plot_all_maps(targets, models, path_dir, percent_error, cmap):
    fig, axes = plt.subplots(len(targets), len(models), 
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             figsize=(14, 10))

    labels = [chr(97 + i) for i in range(len(models) * len(targets))]  # 'a', 'b', 'c', ...
    
    label_idx = 0
    for col, model in enumerate(models):
        for row, target in enumerate(targets):
            ax = axes[row, col]
           
            run_dir = os.path.join(path_dir, 'cfluxes', model, 'predictions')
            df = pd.read_csv(os.path.join(run_dir, 'predictions_Test_MPI-ESM1-2-HR_RCPglobal85.csv'))
            df = df.drop(['veg_class', 'model', 'scenario'], axis =1)
            df = df[(df.time_since_disturbance > 221)].groupby(['lon', 'lat'], as_index=False).mean()
                
            # Calculate difference
            diff_df, vmin, vmax = calculate_diff(df, target, percent_error=percent_error)
            data_mesh = populate_lpjg_grid(diff_df, target)
                
            create_frame_diff(ax, data_mesh, cmap, vmin, vmax)
            
            label_idx = row * len(models) + col
            ax.text(-0.05, 1.05, f'({labels[label_idx]}) {target} - {model.upper()}',
                    transform=ax.transAxes, fontsize=10, fontweight='bold',
                    va='bottom', ha='left')
                
            ax.tick_params(labelsize=8)  
            ax.set_xlabel('Longitude', fontsize=10) 
            ax.set_ylabel('Latitude', fontsize=10)  

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax), cmap=cmap), 
                        ax=axes, orientation='vertical', fraction=0.05, pad=0.1)
    cbar.set_label('Percent error (%)', fontsize=10) 
    cbar.ax.tick_params(labelsize=8)  
    
    plt.savefig(os.path.join(path_dir, 'figures/fig5_cfluxes_map.jpg'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
# Usage
targets = ['GPP', 'NPP', 'Rh']
models = ['rf', 'nn']
plot_all_maps(targets, models, path_dir='../results', percent_error=True, cmap='bwr')