import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os
import joblib

def load_and_prepare_data(data_dir, num_seeds, test_seed=None):
    '''Load and prepare training data'''
    all_data = []
    for i in range(num_seeds):
        if test_seed is not None and i == test_seed:
            continue
        file_path = os.path.join(data_dir, f'seir_seed_{i}.csv')
        df = pd.read_csv(file_path)
        df['seed'] = i
        df['day'] = np.arange(len(df))
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df[combined_df['Beta'] > 0].copy()
    combined_df['log_beta'] = np.log(combined_df['Beta'])
    return combined_df

def train_model(train_df, degree=3, save_path=None):
    '''Train polynomial regression model with Ridge regularization'''
    X = train_df[['day']].values
    y = train_df['log_beta'].values
    
    weights = np.linspace(0.1, 1, len(X))
    
    model = make_pipeline(
        PolynomialFeatures(degree),
        Ridge(alpha=10)
    )
    model.fit(X, y, ridge__sample_weight=weights)
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")
    
    return model

def load_saved_model(model_path):
    '''Load a trained model from disk'''
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def predict_beta(model, day, min_beta=1e-7):
    '''Predict beta value for given day'''
    log_beta = model.predict([[day]])
    beta = np.exp(log_beta)[0]
    return max(beta, min_beta)

def simulate_seir(seed_data, start_day, model, max_days=200):
    '''Simulate SEIR model with time-varying beta'''
    initial = seed_data.iloc[start_day]
    S, E, I, R, Beta = initial[['S', 'E', 'I', 'R', 'Beta']]
    
    gamma = 0.08
    sigma = 0.1
    
    history = {'days': [start_day], 'I': [I], 'Beta': [predict_beta(model, start_day)]}
    peak_I = I
    peak_day = start_day
    
    for day in range(1, max_days+1):
        current_day = start_day + day
        beta = predict_beta(model, current_day)
        
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
        
        if I < 1e-6 and E < 1e-6:
            break
    
    return peak_day, peak_I, history

def plot_results(seed_data, history, start_day, seed, ax):
    '''Plot results on a specified axis'''
    ax1 = ax
    ax1.plot(seed_data['day'], seed_data['I'], '--', color='orange', label='Actual Infections')
    ax1.plot(history['days'], history['I'], '-', color='blue', label='Predicted Infections')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Infected', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axvline(start_day, color='purple', linestyle='--', alpha=0.8, label='Start Day')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(seed_data['day'], seed_data['Beta'], color='lightgray', 
             linestyle='-', alpha=0.7, label='Actual Beta')
    ax2.plot(history['days'], history['Beta'], '--', color='green', 
             alpha=0.9, label='Predicted Beta')
    ax2.set_yscale('log')
    ax2.set_ylabel('Beta (log)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.set_title(f'Seed {seed} (Start: {start_day})', fontsize=10)

def train_and_save_model(data_dir, num_seeds, model_save_path, exclude_seed=None, degree=2):
    print(f"Training model on {num_seeds} seeds (excluding seed {exclude_seed})...")
    train_df = load_and_prepare_data(data_dir, num_seeds, test_seed=exclude_seed)
    model = train_model(train_df, degree=degree, save_path=model_save_path)
    return model

def predict_multiple_seeds(seeds, start_day, data_dir, model_path):
    '''Predict and plot for multiple seeds in subplots'''
    model = load_saved_model(model_path)
    
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    axes = axes.flatten()
    
    for idx, seed in enumerate(seeds):
        ax = axes[idx]
        try:
            file_path = os.path.join(data_dir, f'seir_seed_{seed}.csv')
            seed_data = pd.read_csv(file_path)
            seed_data['day'] = np.arange(len(seed_data))
            
            if start_day >= len(seed_data):
                print(f"Skipping seed {seed} - insufficient data")
                continue
                
            _, _, history = simulate_seir(seed_data, start_day, model)
            plot_results(seed_data, history, start_day, seed, ax)
            
        except Exception as e:
            print(f"Error processing seed {seed}: {str(e)}")
            ax.axis('off')
    
    plt.show()

if __name__ == "__main__":
    DATA_DIR = '../data/'
    NUM_SEEDS = 1500
    MODEL_PATH = 'name.joblib'
    PREDICT_DATA_DIR = '../data/'
    
    # Uncomment to train model
    # train_and_save_model(DATA_DIR, NUM_SEEDS, MODEL_PATH, degree=3)
    
    # Predict on first 10 seeds
    predict_multiple_seeds(
        seeds=range(10), 
        start_day=50,
        data_dir=PREDICT_DATA_DIR,
        model_path=MODEL_PATH
    )