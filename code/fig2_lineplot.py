import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os
import pandas as pd

os.makedirs("../results", exist_ok=True)

# Define global variables
COLORS = {'VegC': (0/255, 79/255, 0/255), 'SoilC': (196/255, 121/255, 0/255)  , 'LitterC': (178/255, 178/255, 178/255)}
SCENARIOS = [26, 45, 70, 85]
CLIMATE_MODELS = ['GFDL-ESM4', 'MPI-ESM1-2-HR', 'MRI-ESM2-0']
BOREAL = ['boreal_evergreen_forest_woodland', 'boreal_deciduous_forest_woodland']
TEMPERATE = ['temperate_broadleaved_evergreen_forest', 'temperate_deciduous_forest']
TROPICAL = ['tropical_rainforest', 'tropical_deciduous_forest', 'tropical_seasonal_forest']
MIXED = ['temperate_boreal_mixed_forest']
LABELS = [
    '(a) Boreal - RCP2.6', '(b) Temperate - RCP2.6', '(c) Mixed - RCP2.6', '(d) Tropical - RCP2.6', 
    '(e) Boreal - RCP4.5', '(f) Temperate - RCP4.5', '(g) Mixed - RCP4.5', '(h) Tropical - RCP4.5', 
    '(i) Boreal - RCP7.0', '(j) Temperate - RCP7.0', '(k) Mixed - RCP7.0', '(l) Tropical - RCP7.0', 
    '(m) Boreal - RCP8.5', '(n) Temperate - RCP8.5', '(o) Mixed - RCP8.5', '(p) Tropical - RCP8.5'
]
# Load and filter data
def load_and_filter_data(path_dir, climate_models, emulator, scenario):
    df_list = []
    for model in climate_models:
        file_path = os.path.join(path_dir, 'cstocks', emulator, 'predictions', f'predictions_Test_{model}_RCP{scenario}.csv')
        df = pd.read_csv(file_path)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# Process data
def process_data(df, veg_classes):
    df_filtered = df[df.veg_class.isin(veg_classes)]
    df_filtered = df_filtered.drop(['veg_class', 'model', 'scenario'], axis=1)
    return df_filtered.groupby('time_since_disturbance').mean()

# Plotting function
def plot_predictions(df_nn, df_rf, ax, label='', y_max=None, soilC_y_max=None, soilC_ticks=None, fontsize=14):
    """
    Plot the predictions and true values for a given region on a specific subplot axis, with SoilC on a different y-axis.
    """
    # VegC and LitterC will share the same y-axis
    ax.plot(df_nn['VegC_pred'], linestyle='--', linewidth=3, label='NN VegC', color=COLORS['VegC'])
    ax.plot(df_nn['VegC_true'], linewidth=3, label='LPJ-GUESS VegC', color=COLORS['VegC'])
    ax.plot(df_rf['VegC_pred'], linestyle='dotted', linewidth=3, label='RF VegC', color=COLORS['VegC'])

    ax.plot(df_nn['LitterC_pred'], linestyle='--', linewidth=3, label='NN LitterC', color=COLORS['LitterC'])
    ax.plot(df_nn['LitterC_true'], linewidth=3, label='LPJ-GUESS LitterC', color=COLORS['LitterC'])
    ax.plot(df_rf['LitterC_pred'], linestyle='dotted', linewidth=3, label='RF LitterC', color=COLORS['LitterC'])

    ax.set_xlim(1850, 2100)
    ax.set_ylabel('VegC and LitterC (kgCm$^{-2}$)', fontsize=fontsize)
    ax.set_xlabel('Year', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    
    # Add a secondary y-axis for SoilC
    ax2 = ax.twinx()
    ax2.plot(df_nn['SoilC_pred'], linestyle='--', linewidth=3, label='NN SoilC', color=COLORS['SoilC'])
    ax2.plot(df_nn['SoilC_true'], linewidth=3, label='LPJ-GUESS SoilC', color=COLORS['SoilC'])
    ax2.plot(df_rf['SoilC_pred'], linestyle='dotted', linewidth=3, label='RF SoilC', color=COLORS['SoilC'])
    
    ax2.set_ylabel('SoilC (kgCm$^{-2}$)', fontsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    
    if y_max is not None:
        ax.set_ylim(top=y_max) 
    
    if soilC_y_max is not None:
        ax2.set_ylim(top=soilC_y_max)  
    
    if soilC_ticks is not None:
        ax2.set_yticks(soilC_ticks) 
    
    # Add subplot label
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=fontsize, fontweight='bold', va='top', ha='left')
    
# Plot
#Create a grid of subplots: 4 rows (scenarios) x 4 columns (biome)
fig, axs = plt.subplots(len(SCENARIOS), 4, figsize=(22, 18)) 
# Loop over each scenario and forest biome to plot data
label_idx = 0
for row, scenario in enumerate(SCENARIOS):
    # Load and process data for each scenario
    df_nn = load_and_filter_data('../results', CLIMATE_MODELS, 'nn', scenario)
    df_rf = load_and_filter_data('../results', CLIMATE_MODELS , 'rf', scenario)
    
    # Process data for Boreal, Temperate, Tropical, and Mixed regions
    df_nn_boreal = process_data(df_nn, BOREAL)
    df_rf_boreal = process_data(df_rf, BOREAL)
    df_nn_boreal.index = list(range(1849, 2100, 1))
    df_rf_boreal.index = list(range(1849, 2100, 1))

    df_nn_temperate = process_data(df_nn, TEMPERATE)
    df_rf_temperate = process_data(df_rf, TEMPERATE)
    df_rf_temperate.index = list(range(1849, 2100, 1))
    df_nn_temperate.index = list(range(1849, 2100, 1))

    df_nn_tropical = process_data(df_nn, TROPICAL)
    df_rf_tropical = process_data(df_rf, TROPICAL)
    df_rf_tropical.index = list(range(1849, 2100, 1))
    df_nn_tropical.index = list(range(1849, 2100, 1))
    
    df_nn_mixed = process_data(df_nn, MIXED)
    df_rf_mixed = process_data(df_rf, MIXED)
    df_rf_mixed.index = list(range(1849, 2100, 1))
    df_nn_mixed.index = list(range(1849, 2100, 1))
# Plot each forest biome (column) in its respective row (scenario)
    plot_predictions(df_nn_boreal, df_rf_boreal, axs[row, 0], label=LABELS[label_idx], y_max=8, soilC_y_max=26)
    label_idx += 1
    plot_predictions(df_nn_temperate, df_rf_temperate, axs[row, 1], label=LABELS[label_idx], y_max=9, soilC_y_max=10)
    label_idx += 1
    plot_predictions(df_nn_mixed, df_rf_mixed, axs[row, 2], label=LABELS[label_idx], y_max=12, soilC_y_max=18)
    label_idx += 1
    plot_predictions(df_nn_tropical, df_rf_tropical, axs[row, 3], label=LABELS[label_idx], y_max=16, soilC_y_max=8)
    label_idx += 1

# Adjust layout to create space for the legend at the bottom
fig.subplots_adjust(bottom=0.2)

# Define legend elements
legend_elements = [
    mpatches.Patch(color=COLORS['VegC'], label='VegC'),
    mpatches.Patch(color=COLORS['SoilC'], label='SoilC'),
    mpatches.Patch(color=COLORS['LitterC'], label='LitterC'),
    mlines.Line2D([], [], color='black', linestyle='-', label='LPJ-GUESS'),
    mlines.Line2D([], [], color='black', linestyle='--', label='NN'),
    mlines.Line2D([], [], color='black', linestyle='dotted', label='RF')
]

# Add the legend to the figure at the bottom center
fig.legend(handles=legend_elements, loc='lower right', ncol=6, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join('..', 'results/figures', f'fig2_cstocks_lineplot.jpg'), dpi=200)
plt.close()

    