import matplotlib.pyplot as plt
import pandas as pd
import os

num_seeds = 30
model = 'seir'

path = f'../data/{model}_30_seeds_v0/'

cols = 2
rows = (num_seeds + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
axes = axes.flatten()

for seed in range(0, num_seeds):
    file_name = os.path.join(path, f'{model}_seed_{seed}.csv')
    df = pd.read_csv(file_name)

    ax = axes[seed]
    max_beta = df['Beta'].max()
    ax.plot(df['Beta'], color='b', label=f'Max Beta: {max_beta:.8f}')
    ax.set_xlabel('Day')
    ax.set_ylabel('Beta')
    ax.set_title(f'Beta for seed {seed}')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

