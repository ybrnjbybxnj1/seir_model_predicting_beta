import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GAMMA = 0.08
SIGMA = 0.1
START_DAY = 55
SEED = 0
DATA_DIR = "../data/"
BETA_FILE = "../data/percentile_beta_values.txt"
AVG_BETA_FILE = "../data/avg_beta_values.txt"
SEED_FILE = f"{DATA_DIR}seir_seed_{SEED}.csv"

def load_beta_trajectories(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    percentiles = list(map(int, lines[0].strip().split()))
    data = [list(map(float, line.strip().split())) for line in lines[1:]]
    beta_trajectories = list(map(list, zip(*data)))
    return beta_trajectories, percentiles

def load_avg_beta(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()[1:]
    return [float(line.strip()) for line in lines]

def load_seed_data(file_path):
    df = pd.read_csv(file_path)
    df["day"] = np.arange(len(df))
    return df

def adjust_beta_trajectories(beta_trajectories, original_start_day, new_start_day):
    """
    Adjust beta trajectories to align with the new start day.
    """
    adjusted_trajectories = []
    for beta_traj in beta_trajectories:
        if new_start_day > original_start_day:
            adjusted_traj = beta_traj[new_start_day - original_start_day:]
        elif new_start_day < original_start_day:
            padding = [beta_traj[0]] * (original_start_day - new_start_day)
            adjusted_traj = padding + beta_traj
        else:
            adjusted_traj = beta_traj
        adjusted_trajectories.append(adjusted_traj)
    return adjusted_trajectories

def simulate_seir(beta_trajectory, start_day, initial_conditions, max_days=200):
    S, E, I, R = initial_conditions
    history = {"days": [start_day], "I": [I], "Beta": [beta_trajectory[start_day]]}

    for day in range(0, max_days):
        beta = beta_trajectory[day]
        new_exposed = beta * S * I
        new_infectious = SIGMA * E
        new_recoveries = GAMMA * I

        S = max(S - new_exposed, 0)
        E = max(E + new_exposed - new_infectious, 0)
        I = max(I + new_infectious - new_recoveries, 0)
        R = max(R + new_recoveries, 0)

        history["days"].append(start_day + day)
        history["I"].append(I)
        history["Beta"].append(beta)

        if I < 1e-7 and E < 1e-7:
            break

    return history

beta_trajectories, percentiles = load_beta_trajectories(BETA_FILE)
avg_beta = load_avg_beta(AVG_BETA_FILE)
seed_data = load_seed_data(SEED_FILE)

initial_conditions = seed_data.iloc[START_DAY][["S", "E", "I", "R"]].values
adjusted_beta_trajectories = adjust_beta_trajectories(beta_trajectories, original_start_day=50, new_start_day=START_DAY)

simulated_trajectories = [simulate_seir(beta, START_DAY, initial_conditions) for beta in adjusted_beta_trajectories]
avg_beta_trajectory = simulate_seir(avg_beta, START_DAY, initial_conditions)

plt.figure(figsize=(12, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(seed_data["day"], seed_data["I"], '--', color="black", linewidth=2, label="Actual infections")
cmap = plt.get_cmap("viridis", len(percentiles))
colors = cmap(range(len(percentiles)))

for i, (traj, p) in enumerate(zip(simulated_trajectories, percentiles)):
    color = colors[i]
    ax1.plot(traj["days"], traj["I"], linestyle="-", linewidth=2, color=color, label=f"SEIR {p}th", alpha=0.3)
    ax2.plot(traj["days"], traj["Beta"], linestyle="--", linewidth=1.5, color="lightgray")

ax1.plot(avg_beta_trajectory["days"], avg_beta_trajectory["I"], linestyle="-", linewidth=2, color="red", label="SEIR average beta", alpha=0.6)
ax2.plot(avg_beta_trajectory["days"], avg_beta_trajectory["Beta"], linestyle="--", linewidth=1.5, color="lightgray", label="Avg beta")
ax1.axvline(START_DAY, color="magenta", linestyle="--", linewidth=1.5, label="Prediction start", alpha=0.3)

ax2.plot(seed_data["day"], seed_data["Beta"], linestyle="-", linewidth=2, color="gray", label="Actual Beta", alpha=0.5)

ax1.set_xlabel("Days")
ax1.set_ylabel("Infected population", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True, alpha=0.3)

ax2.set_yscale("log")
ax2.set_yticks(ticks=[0.0001, 0.0002])
ax2.set_ylabel("Beta (log scale)", color="green")
ax2.tick_params(axis="y", labelcolor="green")

legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=f"{p}th percentile") for i, p in enumerate(percentiles)]
legend_elements.append(Line2D([0], [0], color="red", lw=2, label="Average beta"))
legend_elements.append(Line2D([0], [0], linestyle="--", color="red", lw=1.5, label="Prediction start"))
legend_elements.append(Line2D([0], [0], color="gray", lw=2, label="Actual beta"))

ax1.legend(handles=legend_elements, title="Percentile", bbox_to_anchor=(1.1, 1), loc="upper left")

plt.title(f"SEIR model predictions for seed {SEED} (Start day {START_DAY})")
plt.tight_layout()
plt.show()

