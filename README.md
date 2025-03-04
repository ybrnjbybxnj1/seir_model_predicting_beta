# SEIR model prediction repository

## Overview
This repository contains Python scripts and modules for simulating and predicting the spread of infectious diseases using the SEIR (Susceptible, Exposed, Infected, Recovered) model. The project features two primary approaches for predicting the effective transmission rate (Beta):

- **Traditional approach**: uses polynomial regression with Ridge regularization.
- **Deep learning approach**: leverages an LSTM network to predict Beta values from historical data.

Both methods integrate their beta predictions into the SEIR model to simulate infection dynamics over time.

## Project structure

### Polynomial Regression Based Prediction

- **seir_model_prediction_time_calculator.py**: loads data, trains a polynomial regression model, and predicts Beta values.
- **seir_model_simulation_avg_beta.py**: simulates the SEIR model using average Beta values.
- **seir_model_simulation_real_beta.py**: simulates the SEIR model using actual Beta values from the dataset.
- **seir_model_training_and_prediction.py**: main script for training the model and making predictions on specified seeds.
- **rmse_visualization_boxplot.py**: calculates and visualizes RMSE (Root Mean Square Error) for predictions across multiple seeds.
- **rmse_analysis_10_seeds_seir.py**: provides RMSE analysis on a subset of seeds.
- **seir_model_beta_prediction.py**: predicts Beta values using a trained model.
- **seir_model_multiple_beta_trajectories.py**: simulates the SEIR model using multiple Beta trajectories.
- **seir_model_infected_population_analysis.py**: analyzes the infected population and computes percentiles over time.
- **seir_model_percentile_analysis_10_seeds.py**: similar analysis for a smaller subset of seeds.
- **seir_model_prediction_functions.py**: contains functions for loading data, training the model, and making predictions.
- **seir_model_prediction_10_seeds.py**: similar to the above but focuses on a smaller subset of seeds.
- **seir_model_shifted_beta_prediction.py**: implements a shifted prediction model for Beta values based on previous infection rates.
- **beta_values_visualization.py**: visualizes Beta values for multiple seeds.
- **beta_values_analysis.py**: analyzes and plots Beta values from the dataset.

### LSTM-based prediction and simulation (New)

- **LSTM_model_code.py**: Integrates deep learning for enhanced Beta prediction and SEIR simulation. This file includes:

  #### Data preparation
  - **create_sequences()**: converts multivariate time series data into fixed-length sequences.
  - **load_and_prepare_data()**: loads CSV files from multiple seeds, applies preprocessing (including shifting for `prev_I` and computing `log(Beta)`), and fits a global scaler.

  #### Model architecture
  - **build_lstm_model()**: constructs an LSTM network with two layers, dropout regularization, and uses the RMSprop optimizer.

  #### Training configuration
  - **lr_scheduler()**: implements learning rate decay after a set number of epochs.
  - **train_lstm_model()**: trains the LSTM model using early stopping and a learning rate scheduler.
  - **train_and_save_model()**: an end-to-end pipeline that loads data, trains the model, saves the trained model and scaler, and plots the training history.

  #### Prediction and simulation
  - **LSTMPredictor**: a class that wraps the trained LSTM model for rolling-window predictions, handling normalization and denormalization.
  - **simulate_seir_lstm()**: uses the `LSTMPredictor` to simulate the SEIR model by updating compartments (S, E, I, R) using predicted Beta values.
  - **predict_multiple_seeds()**: evaluates the model on multiple seeds, plotting actual versus predicted infection counts and Beta values, and computing RMSE.
  - **plot_simulation_grid()**: creates a grid layout to visualize simulation results across multiple seeds with detailed legends and error metrics.

  #### Main execution block
  - Orchestrates the training, evaluation, and visualization workflows with configurable parameters (e.g., data directories, window size, number of seeds, simulation start day).

### Converters

- **abm_to_seir_data_converter.py**: converts agent-based model (ABM) data into SEIR format, calculating compartments (S, E, I, R) and the transmission rate (Beta) for multiple seeds.
- **data_daily_to_weekly_data_converter.py**: processes daily SEIR data files into weekly summaries, selecting every 7th row for analysis.
- **real_data_converter.py**: fits the SEIR model to real-world infection data, optimizing parameters (Beta, Sigma, Gamma) and visualizing the results.

### New files added

- **plot_rmse_distribution.py**: visualizes the distribution of RMSE values for Infections (I) and Beta predictions across multiple seeds.
- **polynomial_features_on_all_6_features_with_rmse.py**: implements a polynomial regression model to predict Beta values using six features: day, S, E, I, R, and prev_I; visualizes the RMSE for both Infections and Beta predictions.
