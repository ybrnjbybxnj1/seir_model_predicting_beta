# SEIR Model Prediction Repository

## Overview
This repository contains Python scripts and modules for simulating and predicting the spread of infectious diseases using the SEIR (Susceptible, Exposed, Infected, Recovered) model. The models utilize historical data to predict future infection rates and the effective transmission rate (Beta) over time.

## Contents
- **seir_model_prediction_time_calculator.py**: Loads data, trains a polynomial regression model with Ridge regularization, and predicts Beta values for the SEIR model.
- **seir_model_simulation_avg_beta.py**: Simulates the SEIR model using average Beta values and plots the results.
- **seir_model_simulation_real_beta.py**: Simulates the SEIR model using actual Beta values from the dataset and plots the results.
- **seir_model_training_and_prediction.py**: Main script to train the model and make predictions on specified seeds.
- **rmse_visualization_boxplot.py**: Calculates and visualizes the RMSE (Root Mean Square Error) for predictions across multiple seeds.
- **rmse_analysis_10_seeds_seir.py**: Similar to the above but focuses on a smaller subset of seeds.
- **seir_model_beta_prediction.py**: Predicts Beta values using a trained model and simulates the SEIR model.
- **seir_model_multiple_beta_trajectories.py**: Simulates the SEIR model using multiple Beta trajectories and plots the results.
- **seir_model_infected_population_analysis.py**: Simulates the SEIR model and calculates percentiles for the infected population over time.
- **seir_model_percentile_analysis_10_seeds.py**: Simulates the SEIR model for a specified number of seeds and plots the results.
- **seir_model_prediction_functions.py**: Contains functions for loading data, training the model, and making predictions.
- **seir_model_prediction_10_seeds.py**: Similar to the above but focuses on a smaller subset of seeds.
- **seir_model_shifted_beta_prediction.py**: Implements a shifted prediction model for Beta values based on previous infection rates.
- **beta_values_visualization.py**: Visualizes Beta values for multiple seeds.
- **beta_values_analysis.py**: Analyzes and plots Beta values from the dataset.
