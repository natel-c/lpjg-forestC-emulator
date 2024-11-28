# Machine Learning Emulation of Forest Carbon Stocks and Fluxes

This repository contains code accompanying our JGR submission, "Emulating grid-based Forest Carbon Dynamics using Machine Learning: an LPJ-GUESS Application."

## Authors 
- Carolina Natel de Moura ([carolina.moura@kit.edu](mailto:carolina.moura@kit.edu))
- David Martin Belda  
- Peter Anthoni  
- Neele Haß  
- Almut Arneth  
- Sam Rabin

## Objective
The project aims to develop an emulator for the Lund-Potsdam-Jena General Ecosystem Simulator (LPJ-GUESS) to enable faster simulations of forest carbon pools and fluxes in LandSyMM. This emulation approach addresses the need for reduced computational expenses in large-scale, grid-based carbon modeling.

## Getting started

## 1. Download the Data, Models and SHAP values

Retrieve the data sets, pre-trained models and pre-calculated shap values from the [Zenodo repository](https://zenodo.org/records/14230951).

## 2. Install Dependencies

To set up the required conda environments for this project, use the provided `.yml` files to create environments with the necessary dependencies.

#### Creating Conda Environments

1. **TensorFlow GPU + SHAP Environment**  
   This environment includes TensorFlow (with GPU support) and SHAP for training and interpreting machine learning models.

   ```bash
   conda env create -f tf-gpu-shap-env.yml
   ```

2. **Geopandas Environment**  
   This environment includes GeoPandas and other geospatial libraries for map plotting and figure generation.

    ```bash
    conda env create -f geopandas-env.yml
    ```
## 3. Training the Emulators

Activate the conda environment and navigate to the code directory:

```bash
conda activate tf-gpu-shap
cd code
```

### Carbon Stocks

- **Neural Network Training**

To reproduce paper results, edit the following line in train_nn_cstocks.py:

```bash
Change the number of epochs to 1000
epochs = 1  # change this to 1000
```

Note: Increasing the number of epochs to 1000 may result in longer training times. Also, be aware that this will NOT overwrite the original trained model in the models/ directory, unless you manually adjust the script.

Then run the training script:

```bash
python train_nn_cstocks.py
```
- **Random Forest Training**

To reproduce paper results, edit the following line in train_rf_cstocks.py:

```bash
Change the number of estimators to 1000
n_estimators=1, # change this to 1000
```

Then run the training script:
```bash
python train_rf_cstocks.py
```

### Carbon Fluxes
- **Neural Network Training**

Similarly to Carbon Stocks, edit train_nn_cfluxes.py to set epochs = 1000, then:

```bash
python train_nn_cfluxes.py
```
- **Random Forest Training**

Similarly to Carbon Stocks, edit train_rf_cfluxes.py to set n_estimators = 1000, then:

```bash
python train_rf_cfluxes.py
```
## 4. Shapley Explainations
After training the models, you can generate Shapley value explanations for carbon stocks and fluxes to understand the importance of each feature in the model’s predictions.

To calculate the Shapley values, edit the shap_plot.py script by removing the commented tags. Otherwise, the script will regenerate the Shapley plots using the saved values from the shap/ directory.

To run the Shapley Explanations, execute:

```bash
python shap_plot.py --task cstocks
python shap_plot.py --task cfluxes
```

## 5. Prediction and Evaluation
After training, evaluate emulator performance. The results will be saved into results/.
```bash
python evaluate_cstocks_historical.py
python evaluate_cstocks_rcp.py
python make_eval_table_cstocks.py
python evaluate_cfluxes_historical.py
python evaluate_cfluxes_rcp.py
python make_eval_table_cfluxes.py
```
## Reproducing the Figures

If you're only interested in reproducing figures, follow these steps:

1. Download data and models from the Zenodo repository.
2. Install dependencies.
3. Use the scripts below for each figure:

- **Figure 1**:
```bash
conda activate geopandas-env
python fig1_gridcells.py 
```
- **Figures 2 & 4**:
```bash
conda activate tf-gpu-shap
python fig2_lineplot.py
python fig4_lineplot.py
```
- **Figure 3 & 5**
```bash
conda activate geopandas-env
python fig3_map.py
python fig5_map.py
```
- **Figures 6 & 7**
```bash
conda activate tf-gpu-shap
python shap_plot.py --task cstocks
python shap_plot.py --task cfluxes
```

# Contact

For questions or feedback, please reach out to Carolina Natel at carolina.moura@kit.edu.

# Citation

If you use any of this code in your experiments, please make sure to cite the code: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14230951.svg)](https://doi.org/10.5281/zenodo.14230951)

# License 

This project is licensed under the BSD-3-Clause License. See [BSD-3-Clause](https://opensource.org/license/BSD-3-Clause) for details.
