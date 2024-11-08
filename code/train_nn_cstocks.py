import joblib
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
#%% Defining global variables
TARGETS=['VegC', 'SoilC', 'LitterC']                                                                                                                                                                              
FEATURES=['time_since_disturbance','temp','prec','insol','temp_min','temp_max','mtemp_max','gdd0','co2','soilc_init','litterc_init', 'vegc_init', 'clay','silt','sand']  

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
#%% Data loading and preparation
data_dir = "../data"

df_train = pd.read_csv(os.path.join(data_dir, "training_dataset.csv"))
X_train = df_train[FEATURES]
y_train = df_train[TARGETS]

df_val = pd.read_csv(os.path.join(data_dir, "validation_dataset.csv"))
X_val = df_val[FEATURES]
y_val = df_val[TARGETS]

# Initialize scalers for features
feature_scaler = MinMaxScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_val_scaled = feature_scaler.transform(X_val)
# Save the scaler
joblib.dump(feature_scaler, '../models/nn_cstocks_feature_scaler.joblib')
#%% Model building
def build_nn_model(input_shape, 
                    learning_rate=0.001, 
                    layers=2, 
                    neurons=64, 
                    activation='relu', 
                    dropout_rate=0.0):
    
    input_layer = Input(shape=(input_shape,))
    shared = input_layer
    
    # Add layers based on the 'layers' parameter
    for _ in range(layers):
        shared = Dense(neurons, activation=activation)(shared)
        if dropout_rate > 0:
            shared = Dropout(dropout_rate)(shared)
    
    output1 = Dense(1, name='VegC_output')(shared)
    output2 = Dense(1, name='SoilC_output')(shared)
    output3 = Dense(1, name='LitterC_output')(shared)
    model = Model(inputs=input_layer, outputs=[output1, output2, output3])
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss={'VegC_output': 'mse', 'SoilC_output': 'mse', 'LitterC_output': 'mse'})
    return model

model =  build_nn_model(
    input_shape=len(FEATURES),
    learning_rate=0.001,
    layers=2, 
    neurons=64, 
    activation='tanh', 
    dropout_rate=0.2)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
print("Training ...")
model.fit(X_train_scaled, 
          [y_train['VegC'], y_train['SoilC'], y_train['LitterC']], 
          epochs=1, # change this to 1000
          batch_size=32,
          validation_data=(X_val_scaled, [y_val['VegC'], y_val['SoilC'], y_val['LitterC']]),
          callbacks=[early_stopping],
          verbose=1)

#model.save('../models/nn_cstocks.keras')
print("Finished training")

