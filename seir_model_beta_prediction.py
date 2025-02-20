import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os

def load_and_prepare_data(data_dir, num_seeds, test_seed):
    all_data = []
    for i in range(num_seeds):
        file_path = os.path.join(data_dir, f'seir_seed_{i}.csv')
        df = pd.read_csv(file_path)
        df['seed'] = i
        df['day'] = np.arange(len(df))
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df[combined_df['Beta'] > 0].copy()
    
    # split into train/test by seed
    train_df = combined_df[combined_df['seed'] != test_seed]
    test_df = combined_df[combined_df['seed'] == test_seed]
    
    train_df['log_beta'] = np.log(train_df['Beta'])
    return train_df, test_df

def train_model(train_df, degree=3):
    X = train_df[['day']].values
    y = np.log(train_df['Beta'].values)
    
    # temporal weighting
    weights = np.linspace(0.1, 1, len(X))  # linear weighting
    
    model = make_pipeline(
        PolynomialFeatures(degree),
        Ridge(alpha=10)  # regularization
    )
    model.fit(X, y, ridge__sample_weight=weights)
    return model

def predict_beta(model, day, min_beta=1e-7):
    log_beta = model.predict([[day]])
    beta = np.exp(log_beta)[0]
    return max(beta, min_beta)

def calculate_gamma(seed_data, start_day, window=2):
    subset = seed_data.iloc[max(0, start_day - window):start_day]

    total_delta_R = subset['R'].iloc[-1] - subset['R'].iloc[0]
    total_I = subset['I'].iloc[0]
    
    if total_I == 0:
        return 0.01
    gamma = total_delta_R / total_I

    gamma = 0.08
    return np.clip(gamma, 0.01, 0.1)

def calculate_sigma(seed_data, start_day, window=2):
    subset = seed_data.iloc[max(0, start_day - window):start_day]

    if len(subset) < 2:
        return 0.1  # default value of sigma
    
    sigma_values = []
    for i in range(len(subset) - 1):
        current = subset.iloc[i]
        next_day = subset.iloc[i+1]
        
        S = current['S']
        E = current['E']
        I = current['I']
        Beta = current['Beta']
        N = S + E + I + current['R']
        
        new_exposed = Beta * S * I / N
        delta_E = next_day['E'] - E
        
        if E <= 1e-6:  # avoid division by 0
            continue
        
        sigma = (new_exposed - delta_E) / E
        sigma_values.append(sigma)
    
    if not sigma_values:
        return 0.1  # Default
    
    sigma_avg = np.mean(sigma_values)
    return np.clip(sigma_avg, 0.1, 0.5)

def simulate_seir(seed_data, start_day, model, max_days=300):
    initial = seed_data.iloc[start_day]
    S = initial['S']
    E = initial['E']
    I = initial['I']
    R = initial['R']
    N = S + E + I + R
    
    #gamma = calculate_gamma(seed_data, start_day)
    #sigma = calculate_sigma(seed_data, start_day)
    
    gamma = 0.08
    sigma = 0.1

    history = {
        'days': [start_day],
        'I': [I],
        'Beta': [predict_beta(model, start_day)]
    }
    
    peak_I = I
    peak_day = start_day
    
    for day in range(1, max_days+1):
        current_day = start_day + day
        beta = predict_beta(model, current_day)
        
        # dynamic scaling factor to prevent early collapse
        scaling = 1.0
        if I < 100:
            scaling = max(1.0, (100 - I)/100)
        
        new_exposed = beta * S * I / N * scaling
        new_infectious = sigma * E
        new_recoveries = gamma * I
        
        # update s_e_i_r values
        S -= new_exposed
        E += new_exposed - new_infectious
        I += new_infectious - new_recoveries
        R += new_recoveries
        
        # correct negative values
        S = max(S, 0)
        E = max(E, 0)
        I = max(I, 0)
        R = max(R, 0)
        
        # update peak tracking
        if I > peak_I:
            peak_I = I
            peak_day = current_day
        
        history['days'].append(current_day)
        history['I'].append(I)
        history['Beta'].append(beta)
        
        # stopping criteria
        if I < 1e-6 and E < 1e-6:
            break
    
    return peak_day, peak_I, history

def plot_results(seed_data, history, start_day, seed):
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    
    # plot data
    ax1.plot(seed_data['day'], seed_data['I'], '--', color='orange', label='Actual I')
    ax1.plot(history['days'], history['I'], '-', color='blue', label='Predicted I')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Infected Population', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axvline(start_day, color='purple', linestyle='--', alpha=0.8, label='Prediction Start')
    #ax1.grid()
    
    # plot beta values
    ax2 = ax1.twinx()
    ax2.plot(seed_data['day'], seed_data['Beta'], color='lightgray', linestyle='-', alpha=0.7, label='Actual Beta')
    ax2.plot(history['days'], history['Beta'], '--', color='green', alpha=0.9, label='Predicted Beta')
    ax2.set_yscale('log')
    ax2.set_ylabel('Beta (log scale)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title(f'SEIR Model Predictions (Seed {seed}, Start Day {start_day})')
    plt.grid()
    plt.show()

def main(seed, start_day, data_dir, num_seeds):
    train_df, test_df = load_and_prepare_data(data_dir, num_seeds, test_seed=seed)
    seed_data = pd.read_csv(os.path.join(data_dir, f'seir_seed_{seed}.csv'))
    seed_data['day'] = np.arange(len(seed_data))
    
    if start_day >= len(seed_data):
        raise ValueError("Start day exceeds available data.")
    
    model = train_model(train_df, degree=3)
    
    peak_day, peak_I, history = simulate_seir(seed_data, start_day, model)
    
    plot_results(seed_data, history, start_day, seed)

if __name__ == "__main__":
    num_seeds = 30
    seed = 5
    day_to_start_prediction = 5
    path = '../data/'
    main(seed, day_to_start_prediction, path, num_seeds)