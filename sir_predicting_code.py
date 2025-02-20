import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os
import joblib


def load_data(data_dir, num_seeds, exclude_seed=None):
    all_data = []
    for i in range(num_seeds):
        if i == exclude_seed:
            continue
        path = os.path.join(data_dir, f'sir_seed_{i}.csv')
        df = pd.read_csv(path)
        df['seed'] = i
        df['day'] = np.arange(len(df))
        all_data.append(df)
    return pd.concat(all_data).reset_index(drop=True)

def prepare_train_data(raw_df):
    df = raw_df[raw_df['Beta'] > 0].copy()
    df['log_beta'] = np.log(df['Beta'])
    return df


def train_and_save_model(train_df, model_path, degree=2):
    X = train_df[['day']].values
    y = train_df['log_beta'].values
    
    weights = np.linspace(0.1, 1, len(X))
    
    model = make_pipeline(
        PolynomialFeatures(degree),
        Ridge(alpha=0.1)
    )
    model.fit(X, y, ridge__sample_weight=weights)
    
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    return joblib.load(model_path)

def predict_beta(model, day, min_beta=1e-7):
    log_beta = model.predict([[day]])[0]
    return max(np.exp(log_beta), min_beta)

def calculate_gamma(seed_data, start_day, window=3):
    subset = seed_data.iloc[max(0, start_day-window):start_day]
    delta_R = subset['R'].iloc[-1] - subset['R'].iloc[0]
    avg_I = subset['I'].mean()
    gamma = 0.08
    return gamma
    #return np.clip(delta_R/avg_I if avg_I > 0 else 0.08, 0.01, 0.1)

def run_simulation(seed_data, start_day, model, max_days=150):
    initial = seed_data.iloc[start_day]
    S, I, R = initial[['S', 'I', 'R']]
    N = S + I + R
    gamma = calculate_gamma(seed_data, start_day)
    
    # Tracking
    history = {
        'days': [start_day],
        'S': [S],
        'I': [I],
        'R': [R],
        'Beta': [predict_beta(model, start_day)]
    }
    for day in range(1, max_days+1):
        current_day = start_day + day
        beta = predict_beta(model, current_day)
        
        scaling = 1.0 if I >= 100 else max(1.0, (100 - I)/100)
        new_infections = beta * S * I * scaling
        new_recoveries = gamma * I
        
        S = max(S - new_infections, 0)
        I = max(I + new_infections - new_recoveries, 0)
        R = max(R + new_recoveries, 0)
        
        history['days'].append(current_day)
        history['S'].append(S)
        history['I'].append(I)
        history['R'].append(R)
        history['Beta'].append(beta)
        
        # Early stopping
        if I < 1e-6 and abs(new_infections - new_recoveries) < 1e-6:
            break
            
    return history


def plot_results(actual_data, simulation_history, start_day, seed):
    plt.figure(figsize=(14, 7))

    ax1 = plt.gca()
    
    ax1.plot(actual_data['day'], actual_data['I'], '--', color='orange', label='Actual')
    ax1.plot(simulation_history['days'], simulation_history['I'], '-', color='blue', label='Predicted')
    ax1.axvline(start_day, color='purple', linestyle='--', label='Prediction Start')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Infected Population')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(actual_data['day'], actual_data['Beta'], color='gray', label='Actual Beta')
    ax2.plot(simulation_history['days'], simulation_history['Beta'], '--', color='green', label='Predicted Beta')
    ax2.axvline(start_day, color='purple', linestyle='--')
    ax2.set_ylabel('Beta Value (log scale)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f'SIR Model Predictions (Seed {seed}, Start Day {start_day})')
    plt.tight_layout()
    plt.show()


def training_workflow(data_dir, num_seeds,model_save_path, exclude_seed=None):
    raw_data = load_data(data_dir, num_seeds, exclude_seed=exclude_seed)
    train_df = prepare_train_data(raw_data)
    model = train_and_save_model(train_df, model_save_path)
    return model

def prediction_workflow(seed, start_day, data_dir, model_path):
    seed_path = os.path.join(data_dir, f'sir_seed_{seed}.csv')
    seed_data = pd.read_csv(seed_path)
    seed_data['day'] = np.arange(len(seed_data))
    
    model = load_model(model_path)
    
    if start_day >= len(seed_data):
        raise ValueError(f"Start day {start_day} exceeds data length {len(seed_data)}")
    
    history = run_simulation(seed_data, start_day, model)
    
    plot_results(seed_data, history, start_day, seed)
    return history


if __name__ == "__main__":
    #DATA_DIR = '../data/'
    MODEL_PATH = 'name.joblib'
    DATA_DIR_TO_PREDICT = '../data/'
    NUM_SEEDS = 30

    #training_workflow(
    #    data_dir=DATA_DIR,
    #    num_seeds = NUM_SEEDS,
    #    model_save_path=MODEL_PATH#,
    #    #exclude_seed=1
    #)
    
    prediction_workflow(
        seed=11,
        start_day=5,
        data_dir=DATA_DIR_TO_PREDICT,
        model_path=MODEL_PATH
    )