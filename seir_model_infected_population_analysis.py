import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

GAMMA = 0.08
SIGMA = 0.1

def simulate_seir(beta_trajectory, initial_conditions, max_days=200):
    """Simulate SEIR model with given beta trajectory and initial conditions"""
    S, E, I, R = initial_conditions
    
    history = {'I': [I], 'Beta': [beta_trajectory[0]]}
    
    for day in range(1, min(len(beta_trajectory), max_days)):
        beta = beta_trajectory[day]
        N = S + E + I + R
        
        new_exposed = beta * S * I 
        new_infectious = SIGMA * E
        new_recoveries = GAMMA * I
        
        S = max(S - new_exposed, 0)
        E = max(E + new_exposed - new_infectious, 0)
        I = max(I + new_infectious - new_recoveries, 0)
        R = max(R + new_recoveries, 0)
        
        history['I'].append(I)
        history['Beta'].append(beta)
        
        if I < 1e-7 and E < 1e-7:
            break
            
    return history

def load_beta_trajectories(data_dir, num_seeds):
    """Load beta values and initial conditions from all seeds"""
    trajectories = []
    initial_conditions_list = []
    for i in range(num_seeds):
        file_path = os.path.join(data_dir, f'seir_seed_{i}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            initial_conditions = df.iloc[0, :4].values.astype(float)
            initial_conditions_list.append(tuple(initial_conditions))
            beta_values = df['Beta'].values[1:]  
            trajectories.append(beta_values)

    return trajectories, initial_conditions_list

def calculate_infected_percentiles(seed_trajs):
    """Calculate percentiles for the infected population over time"""
    infected_data = np.array([traj['I'] for traj in seed_trajs])
    
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    percentile_values = np.percentile(infected_data, percentiles, axis=0)
    
    mean_infected = np.mean(infected_data, axis=0)
    median_infected = np.median(infected_data, axis=0)
    
    return mean_infected, median_infected, percentile_values

def plot_infected_percentiles(mean_infected, median_infected, percentile_values, infected_data, beta_data):
    """Plot the mean, median, percentiles, and infected population bundle along with beta values"""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    mean_infected = np.nan_to_num(mean_infected, nan=0) 
    ax1.plot(mean_infected, color='red', linewidth=2, label='Mean Infected')
    ax1.plot(median_infected, color='blue', linewidth=2, label='Median Infected')
    
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    percentile_colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
    for i, (percentile, color) in enumerate(zip(percentiles, percentile_colors)):
        ax1.plot(percentile_values[i], color=color, linestyle='--', label=f'{percentile}th percentile')
    
    infected_min = np.min(infected_data, axis=0)
    infected_max = np.max(infected_data, axis=0)
    ax1.fill_between(range(len(infected_min)), infected_min, infected_max, color='gray', alpha=0.3, label="Infected bundle")
    
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Infected population')
    ax1.set_title('Infected population and Beta values over time')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    for beta_traj in beta_data:
        ax2.plot(beta_traj, alpha=0.5, color='green', linewidth=1)
    ax2.set_ylabel('Beta Value')
    
    fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    '''CHANGE THE DIRECTORY'''
    data_dir = '../data/'
    '''CHANGE THE NUMBER OF SEEDS'''
    num_seeds = 1500

    seed_betas, seed_initials = load_beta_trajectories(data_dir, num_seeds)
    seed_trajs = [simulate_seir(betas, initial_conditions=init) for betas, init in zip(seed_betas, seed_initials)]
    infected_data = np.array([traj['I'] for traj in seed_trajs])
    mean_infected, median_infected, percentile_values = calculate_infected_percentiles(seed_trajs)
    plot_infected_percentiles(mean_infected, median_infected, percentile_values, infected_data, seed_betas)
