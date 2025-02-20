import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os
import joblib
import time
from sklearn.metrics import mean_squared_error
import math
from matplotlib.lines import Line2D  

def load_and_prepare_data(data_dir, num_seeds, test_seed=None):
    all_data = []
    for i in range(num_seeds):
        if test_seed is not None and i == test_seed:
            continue
        file_path = os.path.join(data_dir, f'seir_seed_{i}.csv')
        df = pd.read_csv(file_path)
        df[['S', 'E', 'I', 'R', 'Beta']] = df[['S', 'E', 'I', 'R', 'Beta']].fillna(0)
        df['prev_I'] = df['I'].shift(-2).fillna(0)
        df['seed'] = i
        df['day'] = np.arange(len(df))
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df[combined_df['Beta'] > 0].copy()
    combined_df['log_beta'] = np.log(combined_df['Beta'])
    return combined_df

def train_model(train_df, degree=3, save_path=None):
    X = train_df[['day', 'prev_I']].values
    y = train_df['log_beta'].values
    
    weights = np.linspace(0.1, 1, len(X))
    
    model = make_pipeline(
        PolynomialFeatures(degree),
        Ridge(alpha=10))
    model.fit(X, y, ridge__sample_weight=weights)
    
    if save_path:
        joblib.dump(model, save_path)
    return model

def load_saved_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File not found: {model_path}")
    return joblib.load(model_path)

def predict_beta(model, day, prev_I, min_beta=1e-7):
    log_beta = model.predict([[day, prev_I]])
    beta = np.exp(log_beta)[0]
    return max(beta, min_beta)

def simulate_seir(seed_data, start_day, model, max_days=200, stop_early=True):
    initial = seed_data.iloc[start_day]
    S, E, I, R, Beta = initial[['S', 'E', 'I', 'R', 'Beta']]

    gamma = 0.08
    sigma = 0.1

    prev_I = seed_data.iloc[start_day-1]['I'] if start_day > 0 else 0.0

    beta = predict_beta(model, start_day, prev_I)
    
    history = {'days': [start_day], 'I': [I], 'Beta': [beta]}
    peak_I = I
    peak_day = start_day
    
    for day in range(1, max_days+1):
        current_day = start_day + day
        prev_I_sim = history['I'][-1]
        beta = predict_beta(model, current_day, prev_I_sim)

        scaling = max(1.0, (100 - I)/100) if I < 100 else 1.0
        new_exposed = beta * S * I * scaling
        new_infectious = sigma * E
        new_recoveries = gamma * I
        
        S = max(S - new_exposed, 0)
        E = max(E + new_exposed - new_infectious, 0)
        I = max(I + new_infectious - new_recoveries, 0)
        R = max(R + new_recoveries, 0)
        
        if I > peak_I:
            peak_I = I
            peak_day = current_day
        
        history['days'].append(current_day)
        history['I'].append(I)
        history['Beta'].append(beta)
        
        if stop_early and I < 1e-6 and E < 1e-6:
            break
    
    return peak_day, peak_I, history

def plot_results(seed_data, history, start_day, seed, ax):
    # RMSE for infections
    predicted_I = np.array(history['I'])
    actual_I = seed_data.iloc[start_day:]['I'].values
    n = min(len(predicted_I), len(actual_I))
    mask_I = ~np.isnan(predicted_I[:n]) & ~np.isnan(actual_I[:n])
    if np.sum(mask_I) > 0:
        rmse_I = np.sqrt(mean_squared_error(actual_I[:n][mask_I], predicted_I[:n][mask_I]))
    else:
        rmse_I = np.nan
    
    # RMSE for Beta
    predicted_Beta = np.array(history['Beta'])
    actual_Beta = seed_data.iloc[start_day:]['Beta'].values
    n_beta = min(len(predicted_Beta), len(actual_Beta))
    mask_Beta = ~np.isnan(predicted_Beta[:n_beta]) & ~np.isnan(actual_Beta[:n_beta])
    if np.sum(mask_Beta) > 0:
        rmse_Beta = np.sqrt(mean_squared_error(actual_Beta[:n_beta][mask_Beta], predicted_Beta[:n_beta][mask_Beta]))
    else:
        rmse_Beta = np.nan
    
    ax.plot(seed_data['day'], seed_data['I'], '--', color='orange', label='Actual Infections')
    ax.plot(history['days'], history['I'], '-', color='blue', 
            label=f'Predicted Infections (RMSE: {rmse_I:.2f})')
    ax.axvline(start_day, color='purple', linestyle='--', alpha=0.8, label='Start Day')
    ax.set_xlabel('Days')
    ax.set_ylabel('Infected', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(seed_data['day'], seed_data['Beta'], color='lightgray', 
             linestyle='-', alpha=0.7, label='Actual Beta')
    ax2.plot(history['days'], history['Beta'], '--', color='green', 
             alpha=0.9, label=f'Predicted Beta (RMSE: {rmse_Beta:.2e})')
    ax2.set_yscale('log')
    ax2.set_ylabel('Beta (log)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax.text(0.05, 0.95, f'RMSE I: {rmse_I:.2f}\nRMSE Beta: {rmse_Beta:.2e}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))
    
    ax.set_title(f'Seed {seed} (Start: {start_day})', fontsize=10)
    
    return rmse_I, rmse_Beta


def train_and_save_model(data_dir, num_seeds, model_save_path, exclude_seed=None, degree=3):
    print(f"Training model on {num_seeds} seeds (excluding seed {exclude_seed})...")
    start_time = time.time()
    train_df = load_and_prepare_data(data_dir, num_seeds, test_seed=exclude_seed)
    model = train_model(train_df, degree=degree, save_path=model_save_path)
    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.2f} seconds.")
    
    training_results = pd.DataFrame([{'training_time_seconds': training_time}])
    training_csv = "training_results.csv"
    training_results.to_csv(training_csv, index=False)
    print(f"Training results saved to {training_csv}")
    
    return model, training_time

def predict_multiple_seeds(seeds, start_day, data_dir, model_path, results_csv='prediction_results.csv'):
    model = load_saved_model(model_path)
    
    overall_start_time = time.time()
    results_list = []
    
    all_rmse_I = []
    all_rmse_Beta = []
    
    seeds = list(seeds)
    num_windows = math.ceil(len(seeds) / 10)
    
    for window in range(num_windows):
        window_seeds = seeds[window*10 : (window+1)*10]
        fig, axes = plt.subplots(5, 2, figsize=(20, 8))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        axes = axes.flatten()
        
        for idx, seed in enumerate(window_seeds):
            ax = axes[idx]
            seed_result = {'seed': seed, 'start_day': start_day}
            seed_start_time = time.time()
            try:
                file_path = os.path.join(data_dir, f'seir_seed_{seed}.csv')
                seed_data = pd.read_csv(file_path)
                seed_data[['S', 'E', 'I', 'R', 'Beta']] = seed_data[['S', 'E', 'I', 'R', 'Beta']].fillna(0)
                seed_data['day'] = np.arange(len(seed_data))
                
                if start_day >= len(seed_data):
                    print(f"Skipping seed {seed} - insufficient data")
                    seed_result['error'] = 'insufficient data'
                    results_list.append(seed_result)
                    ax.axis('off')
                    continue
                
                peak_day, peak_I, history = simulate_seir(seed_data, start_day, model, stop_early=False)
                simulation_time = time.time() - seed_start_time
                seed_result['peak_day'] = peak_day
                seed_result['peak_I'] = peak_I
                seed_result['simulation_time_seconds'] = simulation_time
                
                rmse_I, rmse_Beta = plot_results(seed_data, history, start_day, seed, ax)
                seed_result['rmse_I'] = rmse_I
                seed_result['rmse_Beta'] = rmse_Beta
                results_list.append(seed_result)
                
                all_rmse_I.append(rmse_I)
                all_rmse_Beta.append(rmse_Beta)
                
            except Exception as e:
                print(f"Error processing seed {seed}: {str(e)}")
                seed_result['error'] = str(e)
                results_list.append(seed_result)
                ax.axis('off')
        
        for j in range(len(window_seeds), len(axes)):
            axes[j].axis('off')
        
        legend_elements = [
            Line2D([0], [0], color='orange', linestyle='--', label='Actual Infections'),
            Line2D([0], [0], color='blue', linestyle='-', label='Predicted Infections'),
            Line2D([0], [0], color='purple', linestyle='--', label='Start Day'),
            Line2D([0], [0], color='lightgray', linestyle='-', label='Actual Beta'),
            Line2D([0], [0], color='green', linestyle='--', label='Predicted Beta'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))
        fig.suptitle(f"Seeds Window {window+1} (Seeds: {window_seeds})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
    
    overall_prediction_time = time.time() - overall_start_time
    print(f"Predictions completed in {overall_prediction_time:.2f} seconds.")
    
    results_df = pd.DataFrame(results_list)
    results_df['overall_prediction_time_seconds'] = overall_prediction_time
    results_df.to_csv(results_csv, index=False)
    print(f"Prediction results saved to {results_csv}")
    
    fig_box, ax1 = plt.subplots(figsize=(10, 6))
    bp1 = ax1.boxplot(all_rmse_I, positions=[1], widths=0.6)
    ax1.set_ylabel("Infections RMSE", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    bp2 = ax2.boxplot(all_rmse_Beta, positions=[2], widths=0.6)
    ax2.set_ylabel("Beta RMSE", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    ax1.set_xlim(0.5, 2.5)
    ax1.set_xticks([1,2])
    ax1.set_xticklabels(["Infections RMSE", "Beta RMSE"])
    ax1.set_title("RMSE Distributions Across Seeds")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_DIR = '../data/'
    NUM_SEEDS = 1500
    MODEL_PATH = 'name.joblib'
    PREDICT_DATA_DIR = '../data/'
    
    model, training_time = train_and_save_model(DATA_DIR, NUM_SEEDS, MODEL_PATH, degree=3)
    
    predict_multiple_seeds(
        seeds=range(30), 
        start_day=50,
        data_dir=PREDICT_DATA_DIR,
        model_path=MODEL_PATH
    )

    
