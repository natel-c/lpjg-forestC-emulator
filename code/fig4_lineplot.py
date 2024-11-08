import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os
import pandas as pd

os.makedirs("../results", exist_ok=True)

# Define global variables
COLORS = {'NPP': (0/255, 79/255, 0/255), 'Rh': (196/255, 121/255, 0/255)  , 'GPP': (178/255, 178/255, 178/255)}
SCENARIOS = [26, 45, 70, 85]
CLIMATE_MODELS = ['GFDL-ESM4', 'MPI-ESM1-2-HR', 'MRI-ESM2-0']
TARGETS = ['GPP', 'NPP', 'Rh']
LABELS = [
    '(a) Boreal - RCP2.6', '(b) Temperate - RCP2.6', '(c) Mixed - RCP2.6', '(d) Tropical - RCP2.6', 
    '(e) Boreal - RCP4.5', '(f) Temperate - RCP4.5', '(g) Mixed - RCP4.5', '(h) Tropical - RCP4.5', 
    '(i) Boreal - RCP7.0', '(j) Temperate - RCP7.0', '(k) Mixed - RCP7.0', '(l) Tropical - RCP7.0', 
    '(m) Boreal - RCP8.5', '(n) Temperate - RCP8.5', '(o) Mixed - RCP8.5', '(p) Tropical - RCP8.5'
]
veg_classes_dict = {
    'boreal': ['boreal_evergreen_forest_woodland', 'boreal_deciduous_forest_woodland'],
    'temperate': ['temperate_broadleaved_evergreen_forest', 'temperate_deciduous_forest'],
    'tropical': ['tropical_rainforest', 'tropical_deciduous_forest', 'tropical_seasonal_forest'],
    'mixed': ['temperate_boreal_mixed_forest']
}

# Load and filter data
def load_and_filter_data(path_dir, climate_models, emulator, scenario):
    df_list = []
    for model in climate_models:
        file_path = os.path.join(path_dir, 'cfluxes', emulator, 'predictions', f'predictions_Test_{model}_RCP{scenario}.csv')
        df = pd.read_csv(file_path)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# Process data
def process_data(df, veg_classes):
    df_filtered = df[df.veg_class.isin(veg_classes)]
    df_filtered = df_filtered.drop(['veg_class', 'model', 'scenario'], axis=1)
    return df_filtered.groupby('time_since_disturbance').mean()

# Plotting 
def plot_predictions(df_nn, df_rf, ax, label='', y_max=None, gpp_y_max=None, gpp_ticks=None, fontsize=14):
    for target in TARGETS:
        color = COLORS[target]
        ax.plot(df_nn[f'{target}_pred'], linestyle='--', linewidth=3, label=f'NN {target}', color=color)
        ax.plot(df_nn[f'{target}_true'], linewidth=3, label=f'LPJ-GUESS {target}', color=color)
        ax.plot(df_rf[f'{target}_pred'], linestyle='dotted', linewidth=3, label=f'RF {target}', color=color)
    
    ax.set_xlim(1850, 2100)
    ax.set_ylabel('kgCm$^{{-2}}$', fontsize=fontsize)
    ax.set_xlabel('Year', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=fontsize, fontweight='bold', va='top', ha='left')

# Plot
#Create a grid of subplots: 4 rows (scenarios) x 4 columns (biome)
fig, axs = plt.subplots(4, 4, figsize=(22, 18))
label_idx = 0
for row, scenario in enumerate(SCENARIOS):
    df_nn = load_and_filter_data('../results', CLIMATE_MODELS, 'nn', scenario)
    df_rf = load_and_filter_data('../results', CLIMATE_MODELS, 'rf', scenario)
    for col, (veg_key, veg_classes) in enumerate(veg_classes_dict.items()):
        df_nn_veg = process_data(df_nn, veg_classes)
        df_rf_veg = process_data(df_rf, veg_classes)
        df_nn_veg.index = list(range(1849, 2100))
        df_rf_veg.index = list(range(1849, 2100))
        plot_predictions(df_nn_veg, df_rf_veg, axs[row, col], label=LABELS[label_idx], y_max=1.5, gpp_y_max=3)
        label_idx += 1

# Adjust layout to create space for the legend at the bottom
fig.subplots_adjust(bottom=0.2)

# Define legend elements
legend_elements = [
    mpatches.Patch(color=COLORS['GPP'], label='GPP'),
    mpatches.Patch(color=COLORS['NPP'], label='NPP'),
    mpatches.Patch(color=COLORS['Rh'], label='Rh'),
    mlines.Line2D([], [], color='black', linestyle='-', label='LPJ-GUESS'),
    mlines.Line2D([], [], color='black', linestyle='--', label='NN'),
    mlines.Line2D([], [], color='black', linestyle='dotted', label='RF')
]

# Add the legend to the figure at the bottom center
fig.legend(handles=legend_elements, loc='lower right', ncol=6, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join('..', 'results/figures', f'fig4_cfluxes_lineplot.jpg'), dpi=200)
plt.close()
