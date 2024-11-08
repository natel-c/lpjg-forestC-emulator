# use conda env: shap
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set backend to avoid display issues
from tensorflow.keras.models import load_model
import shap
import dill

TARGETS=['GPP', 'NPP', 'Rh']                                                                                                                                                                              
FEATURES=['time_since_disturbance','temp','prec','insol','temp_min','temp_max','mtemp_max','gdd0','co2','soilc_init','litterc_init', 'vegc_init', 'clay','silt','sand']  
SEED = 42
task = "cfluxes"

# Load the models
rf_cfluxes = joblib.load(f'../models/rf_{task}.joblib')
nn_cfluxes = load_model(f"../models/nn_{task}.keras")

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
# rf_explainer = shap.TreeExplainer(rf_cfluxes, X_test_shap)
# rf_shap_values = rf_explainer(X_test_shap)

# # Save the explainer
# joblib.dump(rf_explainer, '../shap/cfluxes/rf_explainer.joblib')
# # Save the SHAP values
# joblib.dump(rf_shap_values, '../shap/cfluxes/rf_shap_values.joblib')

rf_shap_values = joblib.load('../shap/cfluxes/rf_shap_values.joblib')

# Assuming you have N outputs
for i in range(3):
    print(i)
    if i == 0: 
        target ='GPP'
    elif i == 1: 
        target = 'NPP'
    else:
        target = 'Rh'
    output_shap_values = rf_shap_values[:, :, i]
    
    shap.summary_plot(output_shap_values, X_test_shap)
    plt.savefig(f'../results/figures/shap/summary_plot_{target}_RF.jpg', dpi=300, bbox_inches='tight')
    plt.close()
   
# NN
feature_scaler = joblib.load(f'../models/nn_{task}_feature_scaler.joblib')
X_test_scaled_shap = feature_scaler.transform(X_test_shap)
X_test_scaled_shap = pd.DataFrame(X_test_scaled_shap)
X_test_scaled_shap.columns = FEATURES

# Concatenate predictions along axis 1 (assuming multiple outputs are of shape (N, 1))
# nn_explainer = shap.KernelExplainer(model=lambda x: np.concatenate(nn_cfluxes.predict(x), axis=1), data=X_test_scaled_shap)
# nn_shap_values = nn_explainer.shap_values(X_test_scaled_shap)

# # Save the explainer
# with open(f'../shap/{task}/nn_explainer.dill', 'wb') as f:
#     dill.dump(nn_explainer, f)

# # Save the SHAP values
# with open(f'../shap/{task}/nn_shap_values.dill', 'wb') as f:
#     dill.dump(nn_shap_values, f

with open(f'../shap/{task}/nn_shap_values.dill', 'rb') as f:
    nn_shap_values = dill.load(f)
    
shap.summary_plot(nn_shap_values[0][:], X_test_scaled_shap)
plt.savefig('../results/figures/shap/summary_plot_GPP_NN.jpg', dpi=300, bbox_inches='tight')
plt.close()
shap.summary_plot(nn_shap_values[1][:], X_test_scaled_shap)
plt.savefig('../results/figures/shap/summary_plot_NPP_NN.jpg', dpi=300, bbox_inches='tight')
plt.close()

shap.summary_plot(nn_shap_values[2][:], X_test_scaled_shap)
plt.savefig('../results/figures/shap/summary_plot_Rh_NN.jpg', dpi=300, bbox_inches='tight')
plt.close()


