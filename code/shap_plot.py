# use conda env: tf_gpu_shap
import argparse  # For parsing command-line arguments
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set backend to avoid display issues
from tensorflow.keras.models import load_model
import shap
import dill

# Define the valid task options
VALID_TASKS = ['cfluxes', 'cstocks']

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run SHAP analysis for the specified task.")
parser.add_argument(
    '--task',
    type=str,
    required=True,
    help="Task to perform SHAP analysis for. Must be 'cfluxes' or 'cstocks'."
)
args = parser.parse_args()

# Check if the task is valid
if args.task not in VALID_TASKS:
    raise ValueError(f"Invalid task: {args.task}. Must be one of {VALID_TASKS}.")

# Set the task
task = args.task
                                                                                                                                                                    
FEATURES=['time_since_disturbance','temp','prec','insol','temp_min','temp_max','mtemp_max','gdd0','co2','soilc_init','litterc_init', 'vegc_init', 'clay','silt','sand']  
SEED = 42

# Define targets based on task
if task == 'cfluxes':
    TARGETS = ['GPP', 'NPP', 'Rh']
    FIG_NUMBER = '7'
elif task == 'cstocks':
    TARGETS = ['VegC', 'SoilC', 'LitterC']
    FIG_NUMBER = '6'

# Set global font size
plt.rcParams.update({'font.size': 8}) 

# Load the models
rf = joblib.load(f'../models/rf_{task}.joblib')
nn = load_model(f"../models/nn_{task}.keras")

# Load data
data_dir = "../data"
df_test_26 = pd.read_csv(os.path.join(data_dir, "test_MPI-ESM1-2-HR_26_1850_2100.csv"))
df_test_45 = pd.read_csv(os.path.join(data_dir, "test_MPI-ESM1-2-HR_45_1850_2100.csv"))
df_test_70 = pd.read_csv(os.path.join(data_dir, "test_MPI-ESM1-2-HR_70_1850_2100.csv"))
df_test_85 = pd.read_csv(os.path.join(data_dir, "test_MPI-ESM1-2-HR_85_1850_2100.csv"))

df_test = pd.concat([df_test_26, df_test_45, df_test_70, df_test_85], axis = 0)
X_test = df_test[FEATURES]
y_test = df_test[TARGETS]

# Sample X_test
X_test_shap = shap.sample(X_test, nsamples=500, random_state=42)

# RF
# Create object that can calculate SHAP values
# rf_explainer = shap.TreeExplainer(rf, X_test_shap)
# rf_shap_values = rf_explainer(X_test_shap)

# # Save the explainer
# joblib.dump(rf_explainer, f'../shap/{task}/rf_explainer.joblib')
# # Save the SHAP values
# joblib.dump(rf_shap_values, f'../shap/{task}/rf_shap_values.joblib')

rf_shap_values = joblib.load(f'../shap/{task}/rf_shap_values.joblib')

# NN
feature_scaler = joblib.load(f'../models/nn_{task}_feature_scaler.joblib')
X_test_scaled_shap = feature_scaler.transform(X_test_shap)
X_test_scaled_shap = pd.DataFrame(X_test_scaled_shap)
X_test_scaled_shap.columns = FEATURES

# Concatenate predictions along axis 1 (assuming multiple outputs are of shape (N, 1))
# nn_explainer = shap.KernelExplainer(model=lambda x: np.concatenate(nn.predict(x), axis=1), data=X_test_scaled_shap)
# nn_shap_values = nn_explainer.shap_values(X_test_scaled_shap)

# # Save the explainer
# with open(f'../shap/{task}/nn_explainer.dill', 'wb') as f:
#     dill.dump(nn_explainer, f)

# # Save the SHAP values
# with open(f'../shap/{task}/nn_shap_values.dill', 'wb') as f:
#     dill.dump(nn_shap_values, f

with open(f'../shap/{task}/nn_shap_values.dill', 'rb') as f:
    nn_shap_values = dill.load(f)
    
# Set up subplots (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2)  # Adjust figsize as needed

# Loop through each target and plot RF and NN SHAP summary plots
for i, target in enumerate(TARGETS):
    # RF plot (first column)
    rf_output_shap_values = rf_shap_values[:, :, i]
    ax_rf = axes[i, 0]
    plt.sca(ax_rf)
    shap.summary_plot(rf_output_shap_values, X_test_shap, show=False, sort=False, color_bar=False)
    ax_rf.text(-0.05, 1.0, f'({chr(97+i)}) {target} - RF', transform=ax_rf.transAxes, fontsize=8, fontweight='bold', verticalalignment='top', horizontalalignment='left')

    # NN plot (second column)
    nn_output_shap_values = nn_shap_values[i][:]
    ax_nn = axes[i, 1]
    plt.sca(ax_nn)
    shap.summary_plot(nn_output_shap_values, X_test_scaled_shap, show=False, sort=False)
    ax_nn.text(-0.05, 1.0, f'({chr(97+i+3)}) {target} - NN', transform=ax_nn.transAxes, fontsize=8,fontweight='bold', verticalalignment='top', horizontalalignment='left')

    # Remove x-axis labels for all rows except the last one
    if i < 2:
        ax_rf.get_xaxis().set_visible(False)
        ax_nn.get_xaxis().set_visible(False)
    
    # Remove y-axis labels for the second column
    ax_nn.get_yaxis().set_visible(False)
    plt.subplots_adjust(hspace=0.1, wspace=0.1) 
    for ax in axes.flat:  # Iterate over all subplot axes
        ax.set_xlabel('SHAP value')  # Set custom x-axis label


plt.tight_layout()
plt.savefig(f'../results/figures/fig_{FIG_NUMBER}_{task}_shap.jpg', dpi=300)
plt.close()

