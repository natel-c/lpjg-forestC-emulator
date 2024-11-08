import dill
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np
import os
import pandas as pd

TARGETS=['GPP', 'NPP', 'Rh']                                                                                                                                                                              
FEATURES=['time_since_disturbance','temp','prec','insol','temp_min','temp_max','mtemp_max','gdd0','co2','soilc_init','litterc_init', 'vegc_init', 'clay','silt','sand']  
SEED = 42

# Load data
data_dir = "/home/natel-c/git_projects/lpjg-forest-emulator-2/data"
df_test_126 = pd.read_csv(os.path.join(data_dir, "test", "MPI-ESM1-2-HR_126_1850_2100.csv"))
df_test_245 = pd.read_csv(os.path.join(data_dir, "test", "MPI-ESM1-2-HR_245_1850_2100.csv"))
df_test_370 = pd.read_csv(os.path.join(data_dir, "test", "MPI-ESM1-2-HR_370_1850_2100.csv"))
df_test_585 = pd.read_csv(os.path.join(data_dir, "test", "MPI-ESM1-2-HR_585_1850_2100.csv"))

# Preprocess data
df_test = pd.concat([df_test_126, df_test_245, df_test_370, df_test_585], axis = 0)
X_test = df_test[FEATURES]
y_test = df_test[TARGETS]

# Sample X_test
# RF
X_test_shap = shap.sample(X_test, nsamples=500, random_state=42)

# NN
feature_scaler = joblib.load('../models/nn_cfluxes_feature_scaler.joblib')
X_test_scaled_shap = feature_scaler.transform(X_test_shap)
X_test_scaled_shap = pd.DataFrame(X_test_scaled_shap)
X_test_scaled_shap.columns = FEATURES

# Load the SHAP values
rf_shap_values = joblib.load('../shap/cfluxes/rf_shap_values.joblib')
with open('../shap/cfluxes/nn_shap_values.dill', 'rb') as f:
    nn_shap_values = dill.load(f)

# # Plot RF shap values
# for i in range(3):
#     print(i)
#     if i == 0: 
#         target ='GPP'
#     elif i == 1: 
#         target = 'NPP'
#     else:
#         target = 'Rh'
#     print(target)
#     output_shap_values =rf_shap_values[:, :, i]
    
#     shap.summary_plot(output_shap_values, X_test_shap)
#     plt.savefig(f'../shap/cfluxes/summary_plot_RF_{target}.jpg', dpi=300, bbox_inches='tight')
#     plt.close()
    
# # Plot NN shap values
# shap.summary_plot(nn_shap_values[0][:], X_test_scaled_shap)
# plt.savefig('../shap/cfluxes/summary_plot_NN_GPP.jpg', dpi=300, bbox_inches='tight')
# plt.close()
# shap.summary_plot(nn_shap_values[1][:], X_test_scaled_shap)
# plt.savefig('../shap/cfluxes/summary_plot_NN_NPP.jpg', dpi=300, bbox_inches='tight')
# plt.close()

# shap.summary_plot(nn_shap_values[2][:], X_test_scaled_shap)
# plt.savefig('../shap/cfluxes/summary_plot_NN_Rh.jpg', dpi=300, bbox_inches='tight')
# plt.close()



# Create subplots for RF and NN, each with 3 rows
fig_rf, axes_rf = plt.subplots(nrows=3, ncols=1, figsize=(10, 50))  # Increase height for better fit
fig_nn, axes_nn = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))  # Increase height for better fit

# Plot RF SHAP values
for i, target in enumerate(TARGETS):
    shap.summary_plot(rf_shap_values[:, :, i], X_test_shap, show=False)
    plt.sca(axes_rf[i])  # Set the current axis
   
# Plot NN SHAP values
for i, target in enumerate(TARGETS):
    shap.summary_plot(nn_shap_values[i][:], X_test_scaled_shap, show=False)
    plt.sca(axes_nn[i])  # Set the current axis
   
# Adjust layout and save both figures
#plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
fig_rf.savefig('../shap/cfluxes/summary_plot_RF_targets.jpg', dpi=300, bbox_inches='tight')
fig_nn.savefig('../shap/cfluxes/summary_plot_NN_targets.jpg', dpi=300, bbox_inches='tight')
plt.show()















