import joblib
import os
from pathlib import Path
import pandas as pd
import math
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
#%% Defining global variables
TARGETS=['GPP', 'NPP', 'Rh']                                                                                                                                                                             
FEATURES=['time_since_disturbance','temp','prec','insol','temp_min','temp_max','mtemp_max','gdd0','co2','soilc_init','litterc_init', 'vegc_init', 'clay','silt','sand']  
#%% Directory and file handling
def create_directories(base_dir, task):
    os.makedirs(f'{base_dir}/{task}/nn/predictions', exist_ok=True)
    os.makedirs(f'{base_dir}/{task}/rf/predictions', exist_ok=True)

task = 'cfluxes'
path_dir = "../results"
create_directories(path_dir, task)
#%% Data loading and preparation
data_dir = "../data"
# Load the saved scaler
feature_scaler = joblib.load(f'../models/nn_{task}_feature_scaler.joblib')
#%%
#  Evaluation metrics
def normalized_root_mean_squared_error(y_true, y_pred):
    """Calculate normalized root mean squared error."""
    # Calculate Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    # Calculate Root Mean Square Error
    rmse = math.sqrt(mse)
    #Calculate Normalized Root Mean Square Error
    nrmse = rmse/(np.max(y_true) - np.min(y_true))
    return nrmse

def relative_bias(y_true, y_pred):
    """Calculate relative bias."""
    rel_bias = (np.sum(y_pred - y_true) / np.sum(y_true)) * 100
    return rel_bias

def evaluate_model(model, datasets, targets, path_dir, model_type):
    """
    Evaluate model and save predictions and metrics.

    Args:
    model (object): Trained model.
    datasets (dict): Datasets to evaluate.
    targets (list): List of target column names.
    path_dir (str): Directory path to save results.
    model_type (str): Type of model ('nn' or 'rf').

    Returns:
    pd.DataFrame: DataFrame containing evaluation metrics.
    """
    metrics_df = pd.DataFrame(columns=['dataset', 'veg_class', 'target', 'nrmse', 'rel_bias', 'r2'])
    predictions_df = pd.DataFrame()

    for name, (X, y, df) in datasets.items():
        y_pred = model.predict(X)
        y_pred = post_process_predictions(y_pred, model_type)
        
        predictions = create_predictions_df(y_pred, y.index)
        predictions_df = pd.concat([predictions_df, merge_predictions_with_true(y, predictions, df)])
        
        save_predictions(predictions_df, path_dir, model_type, name)
        
        metrics_df = calculate_metrics(metrics_df, name, y, predictions, df, targets)

    save_metrics(metrics_df, path_dir, model_type)
    return metrics_df

def post_process_predictions(y_pred, model_type):
    if model_type == 'nn':
        return [np.maximum(pred, 0) for pred in y_pred]
    return y_pred

def create_predictions_df(y_pred, index):
    return pd.DataFrame({
        'GPP_pred': y_pred[0].flatten() if isinstance(y_pred, list) else y_pred[:, 0].flatten(),
        'NPP_pred': y_pred[1].flatten() if isinstance(y_pred, list) else y_pred[:, 1].flatten(),
        'Rh_pred': y_pred[2].flatten() if isinstance(y_pred, list) else y_pred[:, 2].flatten()
    }, index=index)

def merge_predictions_with_true(y, predictions, df):
    return pd.DataFrame({
        'GPP_true': y['GPP'], 'GPP_pred': predictions['GPP_pred'],
        'NPP_true': y['NPP'], 'NPP_pred': predictions['NPP_pred'],
        'Rh_true': y['Rh'], 'Rh_pred': predictions['Rh_pred'],
        'veg_class': df['veg_class'], 'time_since_disturbance': df['time_since_disturbance'],
        'model': df['model'], 'scenario': df['RCP'],
        'lon': df['Lon'], 'lat': df['Lat']
    })

def save_predictions(predictions_df, path_dir, model_type, name):
    model_path = 'nn' if model_type == 'nn' else 'rf'
    predictions_df.to_csv(f'{path_dir}/{task}/{model_path}/predictions/predictions_{name}.csv', index=False)

def calculate_metrics(metrics_df, name, y, predictions, df, targets):
    y_pred_df = predictions.copy()
    y_pred_df['veg_class'] = df['veg_class']
    
    grouped_true = y.groupby(df['veg_class'])
    grouped_pred = y_pred_df.groupby('veg_class')
    
    for veg_class in grouped_true.groups.keys():
        veg_class_true = grouped_true.get_group(veg_class)
        veg_class_pred = grouped_pred.get_group(veg_class).drop('veg_class', axis=1)
        
        data_list = [] 
        for target in targets:
            nrmse = np.round(normalized_root_mean_squared_error(veg_class_true[target], veg_class_pred[f'{target}_pred']), 2)
            rel_bias = np.round(relative_bias(veg_class_true[target], veg_class_pred[f'{target}_pred']), 2)
            r2 = np.round(r2_score(veg_class_true[target], veg_class_pred[f'{target}_pred']), 2)
            
            data_list.append({
                'dataset': name, 
                'veg_class': veg_class,  
                'target': target,
                'nrmse': nrmse,
                'rel_bias': rel_bias,
                'r2': r2
            })
                    
        new_metrics_df = pd.DataFrame(data_list)
        metrics_df = pd.concat([metrics_df, new_metrics_df], ignore_index=True)
    
    return metrics_df

def save_metrics(metrics_df, path_dir, model_type):
    metrics_df.to_csv(f'{path_dir}/{task}/{model_type}/evaluation_metrics_historical.csv', index=False)
def process_model_file(file_path, feature_scaler=None, features=None, targets=None):
    """
    Process a single model file and return the dataset components (X, y, df)
    
    Args:
    file_path (Path): Path to the CSV file.
    feature_scaler (object, optional): Scaler for features.
    features (list, optional): List of feature column names.
    targets (list, optional): List of target column names.
    
    Returns:
    tuple: (X_test, y_test, df_test)
    """
    df_test = pd.read_csv(file_path)
    
    model_name = file_path.stem.split('_')[1]
        
    df_test['model'] = model_name
    df_test['RCP'] = "historical"
    
    df_test = df_test[df_test.time_since_disturbance <165]
    
    X_test = df_test[features] if features else df_test
    y_test = df_test[targets] if targets else None
    
    if feature_scaler:
        X_test = feature_scaler.transform(X_test)
    
    return X_test, y_test, df_test

def create_datasets(model_files, feature_scaler=None, features=None, targets=None, model_type='nn'):
    """
    Create datasets dictionary from model files.
    
    Args:
    model_files (list): List of file paths.
    feature_scaler (object, optional): Scaler for features.
    features (list, optional): List of feature column names.
    targets (list, optional): List of target column names.
    model_type (str): Type of model ('nn' or 'rf').
    
    Returns:
    dict: Dictionary of datasets.
    """
    datasets = {}
    
    for file_path in model_files:
        file_path = Path(file_path)
        X_test, y_test, df_test = process_model_file(file_path, feature_scaler, features, targets)
        
        key = f"Test_{file_path.stem.split('_')[1]}_historical"
        datasets[key] = (X_test, y_test, df_test)
    
    return datasets
#%%
# Loading trained models
nn_model = load_model(f"../models/nn_{task}.keras")
rf_model = model = joblib.load(f"../models/rf_{task}.joblib")
print("Model loaded")

# List of model filenames
model_files = [
    "../data/test_GFDL-ESM4_26_1850_2100.csv",
    "../data/test_MPI-ESM1-2-HR_26_1850_2100.csv",
    "../data/test_MRI-ESM2-0_26_1850_2100.csv",
    
]
print("Reading data ...")
# For NN models
datasets_nn = create_datasets(model_files, feature_scaler, FEATURES, TARGETS, model_type='nn')
# For RF models
datasets_rf = create_datasets(model_files, features=FEATURES, targets=TARGETS, model_type='rf')

print("Predicting and evaluating emulators")
metrics_df_nn = evaluate_model(nn_model, datasets_nn, TARGETS, path_dir, model_type='nn')
metrics_df_rf = evaluate_model(rf_model, datasets_rf, TARGETS , path_dir, model_type='rf')

