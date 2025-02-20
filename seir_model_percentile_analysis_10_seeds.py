import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Constants
GAMMA = 0.08
SIGMA = 0.1
START_DAY = 55
SEEDS = range(10)  # Seeds 0-9
DATA_DIR = "../data/"
BETA_FILE = "../data/percentile_beta_values.txt"
AVG_BETA_FILE = "../data/avg_beta_values.txt"
ORIGINAL_START_DAY = START_DAY  # When model predictions begin

def load_beta_trajectories(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    percentiles = list(map(int, lines[0].strip().split()))
    data = [list(map(float, line.strip().split())) for line in lines[1:]]
    return list(map(list, zip(*data))), percentiles

def load_avg_beta(file_path):
    with open(file_path, "r") as f:
        return [float(line.strip()) for line in f.readlines()[1:]]

def load_seed_data(file_path):
    df = pd.read_csv(file_path)
    df["day"] = np.arange(len(df))
    return df

def simulate_seir(beta_trajectory, start_day, initial_conditions, max_days=200):
    S, E, I, R = initial_conditions
    history = {"days": [start_day], "I": [I], "Beta": [beta_trajectory[0]]}
    
    for day in range(len(beta_trajectory)):
        if day >= max_days:
            break
            
        beta = beta_trajectory[day]
        new_exposed = beta * S * I
        new_infectious = SIGMA * E
        new_recoveries = GAMMA * I

        S = max(S - new_exposed, 0)
        E = max(E + new_exposed - new_infectious, 0)
        I = max(I + new_infectious - new_recoveries, 0)
        R = max(R + new_recoveries, 0)

        history["days"].append(start_day + day + 1)
        history["I"].append(I)
        history["Beta"].append(beta)

        # Note: this break may stop the simulation before all beta values are used
        if I < 1e-7 and E < 1e-7:
            break
            
    return history

beta_trajectories, percentiles = load_beta_trajectories(BETA_FILE)
avg_beta = load_avg_beta(AVG_BETA_FILE)

fig, axes = plt.subplots(5, 2, figsize=(18, 20))
plt.subplots_adjust(hspace=0.4, wspace=0.3)
axes = axes.flatten()

cmap = plt.get_cmap("viridis", len(percentiles))

for idx, seed in enumerate(SEEDS):
    ax = axes[idx]
    try:
        seed_file = f"{DATA_DIR}seir_seed_{seed}.csv"
        seed_data = load_seed_data(seed_file)
        
        if START_DAY >= len(seed_data):
            print(f"Skipping seed {seed} - insufficient data")
            ax.axis('off')
            continue

        seed_beta = seed_data["Beta"].values

        adjusted_beta_trajs = []
        for traj in beta_trajectories:
            adjusted_traj = np.concatenate([
                #seed_beta[:ORIGINAL_START_DAY],
                traj
            ])
            adjusted_beta_trajs.append(adjusted_traj)
        avg_beta_full = avg_beta
        #avg_beta_full = np.concatenate([
        #    seed_beta[:ORIGINAL_START_DAY],
        #    avg_beta
        #])

        initial_conditions = seed_data.iloc[START_DAY][["S", "E", "I", "R"]].values

        simulated_trajs = []
        for traj in adjusted_beta_trajs:
            sim_beta = traj[START_DAY:]
            history = simulate_seir(sim_beta, START_DAY, initial_conditions)
            simulated_trajs.append(history)
        
        avg_history = simulate_seir(avg_beta_full[START_DAY:], START_DAY, initial_conditions)

        ax.plot(seed_data["day"], seed_data["I"], '--', color="black", lw=2, label="Actual infections")

        for i, history in enumerate(simulated_trajs):
            color = cmap(i)
            ax.plot(history["days"], history["I"], '-', color=color, lw=2, alpha=0.3,
                    label=f'Prediction {percentiles[i]}%' if i == 0 else "")
        ax.plot(avg_history["days"], avg_history["I"], '-', color="red", lw=2, alpha=0.6, label="Average prediction")

        ax.axvline(START_DAY, color="magenta", linestyle='--', alpha=0.5, lw=1.5)
        ax.set_xlabel("Days")
        ax.set_ylabel("Infected", color="blue")
        ax.tick_params(axis='y', labelcolor="blue")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(seed_data["day"], seed_data["Beta"], '-', color="gray", lw=2, alpha=0.5, label="Actual Beta")

        for i, traj in enumerate(adjusted_beta_trajs):
            full_days = np.arange(len(traj))
            ax2.plot(full_days, traj, '--', color='gray', lw=1, alpha=0.4)

        full_days_avg = np.arange(len(avg_beta_full))
        ax2.plot(full_days_avg, avg_beta_full, '--', color="gray", lw=1, alpha=0.6)
        
        ax2.set_ylabel("Beta (log scale)", color="green")
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor="green")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax.set_title(f"Seed {seed} - Start Day {START_DAY}")

    except Exception as e:
        print(f"Error with seed {seed}: {str(e)}")
        ax.axis('off')

plt.tight_layout()
plt.show()
