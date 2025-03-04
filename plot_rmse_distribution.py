import pandas as pd
import matplotlib.pyplot as plt

path='../path/rmse.csv'
data = pd.read_csv(path)

rmse_I = data.RMSE_I
rmse_Beta = data.RMSE_Beta

fig_box, ax1 = plt.subplots(figsize=(10, 6))
bp1 = ax1.boxplot(rmse_I, positions=[1], widths=0.6)
ax1.set_ylabel("Infections RMSE", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
    
ax2 = ax1.twinx()
bp2 = ax2.boxplot(rmse_Beta, positions=[2], widths=0.6)
ax2.set_ylabel("Beta RMSE", color='green')
ax2.tick_params(axis='y', labelcolor='green')
    
ax1.set_xlim(0.5, 2.5)
ax1.set_xticks([1,2])
ax1.set_xticklabels(["Infections RMSE", "Beta RMSE"])
ax1.set_title("RMSE distributions across all seeds")
plt.tight_layout()
plt.show()