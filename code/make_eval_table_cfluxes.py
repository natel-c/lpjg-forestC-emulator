import os
import pandas as pd
import numpy as np

task = 'cfluxes'
#%%
# Average metrics per scenario and emulator
nn_file_path_hist = os.path.join("../results", task, "nn", "evaluation_metrics_historical.csv")  # NN metrics
rf_file_path_hist = os.path.join("../results", task, "rf", "evaluation_metrics_historical.csv")  # RF metrics
df_nn_hist = pd.read_csv(nn_file_path_hist)
df_rf_hist = pd.read_csv(rf_file_path_hist)

df_nn_hist['scenario'] = df_nn_hist['dataset'].apply(lambda x: x.split('_')[-1])
df_rf_hist['scenario'] = df_rf_hist['dataset'].apply(lambda x: x.split('_')[-1])

nn_file_path_rcp = os.path.join("../results", task, "nn", "evaluation_metrics_rcp.csv")  # NN metrics
rf_file_path_rcp = os.path.join("../results", task, "rf", "evaluation_metrics_rcp.csv")  # RF metrics
df_nn_rcp = pd.read_csv(nn_file_path_rcp)
df_rf_rcp = pd.read_csv(rf_file_path_rcp)

df_nn_rcp['scenario'] = df_nn_rcp['dataset'].apply(lambda x: x.split('_')[-1])
df_rf_rcp['scenario'] = df_rf_rcp['dataset'].apply(lambda x: x.split('_')[-1])

df_nn = pd.concat([df_nn_hist, df_nn_rcp], axis =0)
df_rf = pd.concat([df_rf_hist, df_rf_rcp], axis =0)


print("Making a table with the evaluation results for the paper")
result = []
for metric in ['nrmse', 'rel_bias', 'r2']:
    for scenario in df_nn['scenario'].unique():
        row = {
            '': metric.upper(),  
            'Scenario': scenario, 
           'GPP_ANN': np.mean(df_nn.loc[(df_nn['target'] == 'GPP') & (df_nn['scenario'] == scenario), metric].values),
           'GPP_RF': np.mean(df_rf.loc[(df_rf['target'] == 'GPP') & (df_rf['scenario'] == scenario), metric].values),
           'NPP_ANN': np.mean(df_nn.loc[(df_nn['target'] == 'NPP') & (df_nn['scenario'] == scenario), metric].values),
           'NPP_RF': np.mean(df_rf.loc[(df_rf['target'] == 'NPP') & (df_rf['scenario'] == scenario), metric].values),
           'Rh_ANN': np.mean(df_nn.loc[(df_nn['target'] == 'Rh') & (df_nn['scenario'] == scenario), metric].values),
           'Rh_RF': np.mean(df_rf.loc[(df_rf['target'] == 'Rh') & (df_rf['scenario'] == scenario), metric].values),
           }
        result.append(row)

# Convert the list of dictionaries to a DataFrame
result_df = pd.DataFrame(result)
result_df = np.round(result_df, 2)
result_df.to_csv(f"../results/{task}/{task}_eval_metrics.csv", index = False)

print(f"Evaluating results saved to ./results/{task}/{task}_eval_metrics.csv")