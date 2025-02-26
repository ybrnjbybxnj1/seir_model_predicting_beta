# SEIR Model Prediction Repository

## Overview
This repository contains Python scripts and modules for simulating and predicting the spread of infectious diseases using the SEIR (Susceptible, Exposed, Infected, Recovered) model. The project features two primary approaches for predicting the effective transmission rate (Beta):

Traditional approach: uses polynomial regression with Ridge regularization.
Deep learning approach: leverages an LSTM network to predict Beta values from historical data.
Both methods integrate their beta predictions into the SEIR model to simulate infection dynamics over time.

## Project Structure

### Polynomial Regression Based Prediction

- **seir_model_prediction_time_calculator.py**: Loads data, trains a polynomial regression model, and predicts Beta values.
- **seir_model_simulation_avg_beta.py**: Simulates the SEIR model using average Beta values.
- **seir_model_simulation_real_beta.py**: Simulates the SEIR model using actual Beta values from the dataset.
- **seir_model_training_and_prediction.py**: Main script for training the model and making predictions on specified seeds.
- **rmse_visualization_boxplot.py**: Calculates and visualizes RMSE (Root Mean Square Error) for predictions across multiple seeds.
- **rmse_analysis_10_seeds_seir.py**: Provides RMSE analysis on a subset of seeds.
- **seir_model_beta_prediction.py**: Predicts Beta values using a trained model.
- **seir_model_multiple_beta_trajectories.py**: Simulates the SEIR model using multiple Beta trajectories.
- **seir_model_infected_population_analysis.py**: Analyzes the infected population and computes percentiles over time.
- **seir_model_percentile_analysis_10_seeds.py**: Similar analysis for a smaller subset of seeds.
- **seir_model_prediction_functions.py: Contains functions for loading data, training the model, and making
predictions.
- **seir_model_prediction_10_seeds.py**: Similar to the above but focuses on a smaller subset of seeds.
- **seir_model_shifted_beta_prediction.py**: Implements a shifted prediction model for Beta values based on
previous infection rates.
- **beta_values_visualization.py**: Visualizes Beta values for multiple seeds.
   **beta_values_analysis.py**: Analyzes and plots Beta values from the dataset.

### LSTM-Based Prediction and Simulation (New)

- **LSTM_model_code.py**: Integrates deep learning for enhanced Beta prediction and SEIR simulation. This file includes:

  #### Data Preparation
  - **create_sequences()**: Converts multivariate time series data into fixed-length sequences.
  - **load_and_prepare_data()**: Loads CSV files from multiple seeds, applies preprocessing (including shifting for `prev_I` and computing `log(Beta)`), and fits a global scaler.

  #### Model Architecture
  - **build_lstm_model()**: Constructs an LSTM network with two layers, dropout regularization, and uses the RMSprop optimizer.

  #### Training Configuration
  - **lr_scheduler()**: Implements learning rate decay after a set number of epochs.
  - **train_lstm_model()**: Trains the LSTM model using early stopping and a learning rate scheduler.
  - **train_and_save_model()**: An end-to-end pipeline that loads data, trains the model, saves the trained model and scaler, and plots the training history.

  #### Prediction and Simulation
  - **LSTMPredictor**: A class that wraps the trained LSTM model for rolling-window predictions, handling normalization and denormalization.
  - **simulate_seir_lstm()**: Uses the `LSTMPredictor` to simulate the SEIR model by updating compartments (S, E, I, R) using predicted Beta values.
  - **predict_multiple_seeds()**: Evaluates the model on multiple seeds, plotting actual versus predicted infection counts and Beta values, and computing RMSE.
  - **plot_simulation_grid()**: Creates a grid layout to visualize simulation results across multiple seeds with detailed legends and error metrics.

  #### Main Execution Block
  - Orchestrates the training, evaluation, and visualization workflows with configurable parameters (e.g., data directories, window size, number of seeds, simulation start day).
