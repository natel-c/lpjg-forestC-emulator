import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#%% Defining global variables
TARGETS=['VegC', 'SoilC', 'LitterC']                                                                                                                                                                              
FEATURES=['time_since_disturbance','temp','prec','insol','temp_min','temp_max','mtemp_max','gdd0','co2','soilc_init','litterc_init', 'vegc_init', 'clay','silt','sand']  
# Set a seed for reproducibility
SEED = 42
#%% Data loading and preparation
data_dir = "../data"

df_train = pd.read_csv(os.path.join(data_dir, "training_dataset.csv"))
X_train = df_train[FEATURES]
y_train = df_train[TARGETS]
#%% Training the Random Forest Regressor
model = RandomForestRegressor(bootstrap=True, 
                              n_estimators=1, # change this to 1000
                              max_samples=0.2, 
                              max_features=0.8, 
                              max_depth=200, 
                              min_samples_split=250, 
                              n_jobs=-1,
                              random_state=SEED)
print("Training ...")
model.fit(X_train, y_train)
#joblib.dump(model, "../models/rf_cstocks.joblib")
print("Finished training")
