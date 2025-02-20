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
        file_path = os.path.join(data_dir, f'sir_seed_{i}.csv')
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
        Ridge(alpha=0.1)  # regularization to prevent overfitting
    )
    model.fit(X, y, ridge__sample_weight=weights)
    return model

def predict_beta(model, day, min_beta=1e-7):
    log_beta = model.predict([[day]])
    beta = np.exp(log_beta)[0]
    return max(beta, min_beta)

def calculate_gamma(seed_data, start_day, window=21):
    subset = seed_data.iloc[max(0, start_day - window):start_day]
    #print(subset)
    
    # cumulative changes for stability
    total_delta_R = subset['R'].iloc[-1] - subset['R'].iloc[0]
    #print(total_delta_R)
    total_I = subset['I'].sum()
    #print(total_I)
    
    #if total_I == 0:
        #return 0.01  # default minimum
    
    gamma = total_delta_R / total_I
    #gamma = 0.08
    return np.clip(gamma, 0.01, 0.1)  # constrain realistic values
    #return gamma

gamma_list = []
def simulate_sir(seed_data, start_day, model, max_days=50):
    initial = seed_data.iloc[start_day]
    S = initial['S']
    I = initial['I']
    R = initial['R']
    N = S + I + R
    gamma = calculate_gamma(seed_data, start_day)

    gamma_list.append(gamma)
    
    history = {
        'days': [start_day],
        'I': [I],
        'Beta': [predict_beta(model, start_day)]
    }
    
    peak_I = I
    peak_day = start_day
    
    for day in range(1, max_days + 1):
        current_day = start_day + day
        beta = predict_beta(model, current_day)
        
        # dynamic scaling factor to prevent early collapse
        scaling = 1.0
        if I < 100: 
            scaling = max(1.0, (100 - I)/100)
            
        new_infections = beta * S * I / N * scaling
        new_recoveries = gamma * I
        
        S -= new_infections
        I += new_infections - new_recoveries
        R += new_recoveries
        
        # correct negative values
        S = max(S, 0)
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
        if I < 1e-6 or (abs(new_infections - new_recoveries) < 1e-6):
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
    ax1.axhline(0, color = 'red', linestyle='--')
    
    # plot Beta on log scale
    ax2 = ax1.twinx()
    ax2.plot(seed_data['day'], seed_data['Beta'], color='lightgray', linestyle='-', alpha=0.7, label='Actual Beta')
    ax2.plot(history['days'], history['Beta'], '--', color='green', alpha=0.9, label='Predicted Beta')
    ax2.axhline(gamma_list[0], linestyle=':', color = 'magenta', label = f'Gamma: {gamma_list[0]}')
    ax2.set_yscale('log')
    ax2.set_ylabel('Beta (log scale)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title(f'SIR Model Predictions (Seed {seed}, Start Day {start_day})')
    plt.grid(True)
    plt.show()

def main(seed, start_day, data_dir, num_seeds):
    train_df, test_df = load_and_prepare_data(data_dir, num_seeds, test_seed=seed)
    seed_data = pd.read_csv(os.path.join(data_dir, f'sir_seed_{seed}.csv'))
    seed_data['day'] = np.arange(len(seed_data))
    
    if start_day >= len(seed_data):
        raise ValueError("Start day exceeds available data.")
    
    # polynomial degree = 2
    model = train_model(train_df, degree=2)
    
    peak_day, peak_I, history = simulate_sir(seed_data, start_day, model)
    
    #print(f"Predicted peak for seed {seed} starting at day {start_day}:")
    #print(f"Peak of {peak_I:.2f} infected individuals occurs on day {peak_day}")
    
    plot_results(seed_data, history, start_day, seed)

if __name__ == "__main__":
    num_seeds = 10
    seed = 1
    day_to_start_prediction = 20
    path = '../data/'
    main(seed, day_to_start_prediction, path, num_seeds)