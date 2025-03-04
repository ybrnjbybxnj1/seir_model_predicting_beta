import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os
import joblib

def load_and_prepare_data(data_dir, num_seeds, test_seed=None):
    '''Load and prepare training data'''
    all_data = []
    for i in range(num_seeds):
        # Skip the test seed if specified
        if test_seed is not None and i == test_seed:
            continue
        file_path = os.path.join(data_dir, f'seir_seed_{i}.csv')
        df = pd.read_csv(file_path)  # Load data from CSV
        df['seed'] = i  # Add seed identifier
        df['day'] = np.arange(len(df))  # Add day index
        all_data.append(df)
    
    # Combine all data into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df[combined_df['Beta'] > 0].copy()  # Filter out non-positive Beta values
    combined_df['log_beta'] = np.log(combined_df['Beta'])  # Calculate log of Beta
    return combined_df

def train_model(train_df, degree=3, save_path=None):
    '''Train polynomial regression model'''
    train_df['prev_I'] = train_df['I'].shift(2).fillna(0)  # Create previous I feature
    train_df['log_beta'] = np.log(train_df['Beta'].clip(lower=1e-7))  # Log-transform Beta with clipping
    X = train_df[['day', 'S', 'E', 'I', 'R', 'prev_I']].values  # Features
    y = train_df['log_beta'].values  # Target variable

    # Create weights for SGDRegressor
    weights = np.linspace(0.1, 1, len(X))
    
    # Create a pipeline for scaling, polynomial features, and regression
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(include_bias=False, degree=degree),
        SGDRegressor(max_iter=10000, penalty='l2', alpha=0.1, 
                     warm_start=False)
    )
    model.fit(X, y, sgdregressor__sample_weight=weights)  # Fit the model with sample weights
    
    # Save the model if a path is provided
    if save_path:
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")
    
    return model

def load_saved_model(model_path):
    '''Load a trained model from disk'''
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def predict_beta(model, features, min_beta=1e-7):
    '''Predict beta value for given features'''
    log_beta = model.predict([features])  # Predict log beta
    beta = np.exp(log_beta)[0]  # Exponentiate to get beta
    return max(beta, min_beta)  # Ensure Beta is not below the minimum threshold

def simulate_seir(seed_data, start_day, model, max_days=200):
    '''Simulate SEIR model with time-varying beta'''
    initial = seed_data.iloc[start_day]  # Get initial conditions
    S, E, I, R = initial[['S', 'E', 'I', 'R']]  # Susceptible, Exposed, Infected, Recovered
    
    gamma = 0.08  # Recovery rate
    sigma = 0.1   # Rate of progression from exposed to infected
    
    I_history = list(seed_data['I'].iloc[:start_day+1])  # History of infections
    
    history = {
        'days': [start_day],
        'I': [I],
        'Beta': []
    }
    
    # Calculate previous I for the first prediction
    if start_day >= 2:
        prev_I = I_history[start_day-2]
    else:
        prev_I = 0.0
    features = [start_day, S, E, I, R, prev_I]  # Feature set for prediction
    beta = predict_beta(model, features)  # Predict Beta
    history['Beta'].append(beta)  # Store predicted Beta
    
    peak_I = I  # Track peak infections
    peak_day = start_day  # Track day of peak infections
    
    # Simulate the SEIR model over the specified number of days
    for day in range(1, max_days+1):
        current_day = start_day + day
        
        # Scaling factor based on current infections
        scaling = max(1.0, (100 - I)/100) if I < 100 else 1.0
        
        # Calculate new infections, recoveries, and transitions
        new_exposed = beta * S * I * scaling
        new_infectious = sigma * E
        new_recoveries = gamma * I
        
        # Update compartments
        S = max(S - new_exposed, 0)
        E = max(E + new_exposed - new_infectious, 0)
        new_I = max(I + new_infectious - new_recoveries, 0)
        R = max(R + new_recoveries, 0)
        
        I_history.append(new_I)  # Update infection history
        I = new_I  # Update current infections
        
        # Prepare features for the next day prediction
        next_day = current_day + 1
        if next_day >= 2:
            prev_I_next = I_history[next_day-2] if (next_day-2) < len(I_history) else 0.0
        else:
            prev_I_next = 0.0
        
        features_next = [next_day, S, E, I, R, prev_I_next]
        beta = predict_beta(model, features_next)  # Predict next day's Beta
        
        # Store results in history
        history['days'].append(current_day)
        history['I'].append(I)
        history['Beta'].append(beta)
        
        # Update peak information
        if I > peak_I:
            peak_I = I
            peak_day = current_day
        
        # Stop simulation if infections and exposures are negligible
        if I < 1e-7 and E < 1e-7:
            break
    
    return peak_day, peak_I, history  

def train_and_save_model(data_dir, num_seeds, model_save_path, exclude_seed=None, degree=3):
    '''Train model and save it to disk'''
    print(f"Training model on {num_seeds} seeds (excluding seed {exclude_seed})...")
    train_df = load_and_prepare_data(data_dir, num_seeds, test_seed=exclude_seed)  
    model = train_model(train_df, degree=degree, save_path=model_save_path) 
    return model

def plot_results(seed_data, history, start_day, seed, ax):
    '''Plot results on a specified axis and return RMSE for beta and I'''
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

    # Calculate RMSE for I
    rmse_I = np.sqrt(np.mean((np.array(history['I']) - np.array(seed_data['I'].iloc[history['days']]))**2))
    
    # Calculate RMSE for Beta, ensuring no NaN values
    actual_beta = seed_data['Beta'].iloc[history['days']]
    predicted_beta = history['Beta']
    
    if len(predicted_beta) == len(actual_beta) and not np.any(np.isnan(predicted_beta)):
        rmse_Beta = np.sqrt(np.mean((predicted_beta - actual_beta)**2))
    else:
        rmse_Beta = np.nan 

    # Display RMSE on the plot
    ax1.text(0.05, 0.95, f'RMSE (I): {rmse_I:.2f}', transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax1.text(0.05, 0.5, f'RMSE (Beta): {rmse_Beta}' if not np.isnan(rmse_Beta) else 'RMSE (Beta): N/A', 
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.set_title(f'Seed {seed} (Start: {start_day})', fontsize=10)

    return rmse_I, rmse_Beta 

def predict_multiple_seeds(seeds, start_day, data_dir, model_path):
    '''Predict and plot for multiple seeds in subplots'''
    model = load_saved_model(model_path)  
    
    fig, axes = plt.subplots(5, 2, figsize=(15, 20)) 
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    axes = axes.flatten()
    
    rmse_I_list = []  
    rmse_Beta_list = []  
    
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
            rmse_I, rmse_Beta = plot_results(seed_data, history, start_day, seed, ax) 
            
            rmse_I_list.append(rmse_I)  
            rmse_Beta_list.append(rmse_Beta) 
            
        except Exception as e:
            print(f"Error processing seed {seed}: {str(e)}")
            ax.axis('off')
    
    plt.show()

    # Create boxplots for RMSE values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.boxplot(rmse_I_list, vert=False)
    ax1.set_title('RMSE of Infected counts')
    ax1.set_xlabel('RMSE (I)')
    
    ax2.boxplot(rmse_Beta_list, vert=False)
    ax2.set_title('RMSE of beta values')
    ax2.set_xlabel('RMSE (Beta)')
    
    plt.tight_layout()
    plt.show()

    # Save RMSE values to a DataFrame and CSV
    rmse_df = pd.DataFrame({
        'Seed': seeds[:len(rmse_I_list)],
        'RMSE_I': rmse_I_list,
        'RMSE_Beta': rmse_Beta_list
    })
    rmse_save_path = '..rmse_file_path/file.csv'

    # Save RMSE values to a CSV file
    rmse_df.to_csv(rmse_save_path, index=False)
    print(f"RMSE values saved to {rmse_save_path}")

if __name__ == "__main__":
    # Define paths and parameters
    DATA_DIR = '..data_to_train_on/'
    NUM_SEEDS = 1500
    MODEL_PATH = 'model_name.joblib'
    PREDICT_DATA_DIR = '..data_to_predicct_on/'
    
    # Uncomment to train model
    # train_and_save_model(DATA_DIR, NUM_SEEDS, MODEL_PATH, degree=3)
    
    # Predict on first 10 seeds
    predict_multiple_seeds(
        seeds=range(20,30), 
        start_day=50,
        data_dir=PREDICT_DATA_DIR,
        model_path=MODEL_PATH
    )