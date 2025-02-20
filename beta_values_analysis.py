import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

path = '../data/' # CHANGE DIRECTORY NAME

all_data = []
for i in range(1500):
    filename = f'seir_seed_{i}.csv'
    filepath = os.path.join(path, filename)
    df = pd.read_csv(filepath)
    group = i // 5  
    smoothed_df = df
    all_data.append((group, smoothed_df))

variables = ['Beta']

num_groups = 30  
base_colors = plt.get_cmap('tab10').colors 
cmap = [base_colors[i % len(base_colors)] for i in range(num_groups)]  

group_sets = [range(0, 30)]

'''A SNIPPET OF CODE TO PLOT ALL BETA VALUES'''
for idx, group_range in enumerate(group_sets):
    plt.figure(figsize=(16, 8))

    for group, smoothed_df in all_data:
        if group in group_range:
            plt.plot(smoothed_df['Beta'], color=cmap[group], alpha=0.8, linewidth=1)

    legend_elements = [Line2D([0], [0], color=cmap[i], lw=2, label=f'Group {i}') for i in group_range]
    plt.legend(handles=legend_elements, title="Parameter group", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)

    plt.xlim(0, 250)

    plt.title(f'Smoothed beta over time (Groups {min(group_range)}-{max(group_range)})')
    plt.xlabel('Time step')
    plt.ylabel('Beta')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------------------------------------------------------
'''A SNIPPET OF CODE TO PLOT THE MEAN VALUE'''
group_sets = [range(0, 30)] 
for idx, group_range in enumerate(group_sets):
    plt.figure(figsize=(12, 6))

    beta_values_at_each_time = []

    # Plot each group's beta values and collect the data for averaging
    for group, smoothed_df in all_data:
        if group in group_range:
            plt.plot(smoothed_df['Beta'], color=cmap[group], alpha=0.8, linewidth=1)
            beta_values_at_each_time.append(smoothed_df['Beta'])

    # Calculate the mean of beta values at each time point (across all groups)
    if beta_values_at_each_time:
        avg_beta = pd.DataFrame(beta_values_at_each_time).mean(axis=0)
        print("Average beta values across all groups at each time point:")
        print(avg_beta.values.tolist())

        '''UNCOMMENT TO SAVE BETA VALUES'''
        #np.savetxt("avg_beta_values.txt", avg_beta.values, header="Beta")

        print("Average beta values saved to avg_beta_values.txt")
        plt.plot(avg_beta, color='black', linewidth=2, label='Average beta', zorder=3)  

    legend_elements = [Line2D([0], [0], color=cmap[i], lw=1, label=f'Group {i}') for i in group_range]
    legend_elements.append(Line2D([0], [0], color='black', lw=3, label='Average beta'))

    plt.legend(handles=legend_elements, title="Parameter group", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)

    plt.xlim(0, 100)  # Time step range

    '''UNCOMMENT TO SET BETA RANGE'''
    #plt.ylim(0, 0.0002)

    plt.title(f'Smoothed beta over time (Groups {min(group_range)}-{max(group_range)})')
    plt.xlabel('Time step')
    plt.ylabel('Beta')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------------------------------------------------
'''A SNIPPET OF CODE TO PLOT MEDIAN VALUE AND PERCENTILES'''
for idx, group_range in enumerate(group_sets):
    plt.figure(figsize=(12, 6))

    beta_values_at_each_time = []
    for group, smoothed_df in all_data:
        if group in group_range:
            plt.plot(smoothed_df['Beta'], color=cmap[group], alpha=0.2, linewidth=1)  
            beta_values_at_each_time.append(smoothed_df['Beta'])

    # Сalculate median and percentiles
    if beta_values_at_each_time:
        beta_df = pd.DataFrame(beta_values_at_each_time)

        avg_beta = beta_df.mean(axis=0)  
        median_beta = beta_df.median(axis=0)  

        percentiles = [10, 20, 30, 40, 60, 70, 80, 90]
        percentile_values = beta_df.quantile([p / 100 for p in percentiles], axis=0)

        #plt.plot(avg_beta, color='red', linewidth=3, label='Mean', zorder=3)  
        plt.plot(median_beta, color='blue', linewidth=3, linestyle='--', label='Медиана (Median)', zorder=3)  

        percentile_styles = ['dotted', 'dashdot', 'dashed', 'dashed', 'dashed', 'dashdot', 'dotted', 'dotted']
        percentile_colors = ['purple', 'darkorange', 'green', 'brown', 'brown', 'green', 'darkorange', 'purple']

        for i, p in enumerate(percentiles):
            plt.plot(
                percentile_values.loc[p / 100], 
                linestyle=percentile_styles[i], 
                color=percentile_colors[i], 
                linewidth=2, 
                alpha=0.8,
                label=f"{p} percentile"
            )

        #if percentile_values:
        # Save data to the files
            #np.savetxt("percentile_beta_values.txt", percentile_values.values.T, header="10 20 30 40 60 70 80 90")
            #print("Percentile values saved to percentile_beta_values.txt")
            #continue

    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='Mean'),
        Line2D([0], [0], color='blue', lw=3, linestyle='--', label='Median')
    ]
    for i, p in enumerate(percentiles):
        legend_elements.append(Line2D([0], [0], linestyle=percentile_styles[i], color=percentile_colors[i], lw=2, label=f"{p} percentile"))

    plt.legend(handles=legend_elements, title="Groups", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)

    plt.xlim(0, 250)  
    plt.title(f'Beta percentiles over time (Groups {min(group_range)}-{max(group_range)})')
    plt.xlabel('Timestamp')
    plt.ylabel('Beta')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------------------------------------------------------
'''A SNIPPET OF CODE TO PLOT ALL 1500 BETA VALUES'''
plt.figure(figsize=(16, 8))

#for group, smoothed_df in all_data:
    #plt.plot(smoothed_df['Beta'], color='blue', alpha=0.1, linewidth=1)

plt.xlim(0, 250)

plt.title('Smoothed beta over time (All 1500 Simulations)')
plt.xlabel('Time step')
plt.ylabel('Beta')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------------------------------------------------------

