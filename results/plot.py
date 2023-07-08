import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np

# Check if the CSV file is provided as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the path to the CSV file as a command-line argument.")
    sys.exit(1)

# Read the CSV file
filename = sys.argv[1]
df = pd.read_csv(filename)

# Get the number of rows and calculate the number of subplots needed
num_rows = len(df)

# Calculate the number of rows and columns for the subplots
num_cols_plot = min(4, num_rows)
num_rows_plot = int((num_rows + 3)/num_cols_plot)

# Define a color palette
colors = sns.color_palette("Set1")

# Create subplots
fig, axes = plt.subplots(nrows=num_rows_plot, ncols=num_cols_plot, figsize=(13, 3 * num_rows_plot))

subfig_used = [[False for _ in range(num_cols_plot)] for _ in range(num_rows_plot)]

use_v2=False

# Iterate over each row in the DataFrame
for i, row in df.iterrows():
    # Get the data for the current row
    matrixname = row['matrixname']
    types = row['type']
    if use_v2 :
        values = row[['scalar', '1rVc', '1rVc_v2', '2rVc', '2rVc_v2', '4rVc', '4rVc_v2']].astype(float)
    else:
        values = row[['scalar', '1rVc', '2rVc', '4rVc']].astype(float)

    print(matrixname)
    print(str(values))
    print(str(values.index))

    # Calculate the subplot indices for the current row
    row_index = i // num_cols_plot
    col_index = i % num_cols_plot
    
    if subfig_used[row_index][col_index]:
        raise('error')
    subfig_used[row_index][col_index] = True

    # Create a bar plot for the current row
    ax = axes[row_index, col_index] if num_rows_plot > 1 else axes[col_index]
    bars = ax.bar(values.index, values, color=colors)
    # ax.set_xlabel('Implementation')
    if col_index == 0:
        ax.set_ylabel('GFlops/s')
    ax.set_title(f'{matrixname} ({types})')
    
    # Set the major locator for the y-axis ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    
    # Calculate the maximum y-tick value
    max_value = max(values) * 1.2

    # Set the y-axis limits
    ax.set_ylim(top=max_value)
    
    # Format the y-axis tick labels to have consistent decimal precision
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    
    # Add numbers above the bars
    for idx, bar in enumerate(bars):
        if idx != 0:
            height = bar.get_height()
            speedup=values[idx]/values[0]
            ax.annotate(f'×{speedup:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom')

######################################################

# Iterate over each row in the DataFrame
for i, types in enumerate(['double','float']):
    break
    
    if use_v2:
        scalar=df[df['type'] == types]['scalar'].values.astype(float)
        v1rVc=df[df['type'] == types]['1rVc'].values.astype(float)
        v1rVc_v2=df[df['type'] == types]['1rVc_v2'].values.astype(float)
        v2rVc=df[df['type'] == types]['2rVc'].values.astype(float)
        v2rVc_v2=df[df['type'] == types]['2rVc_v2'].values.astype(float)
        v4rVc=df[df['type'] == types]['4rVc'].values.astype(float)
        v4rVc_v2=df[df['type'] == types]['4rVc_v2'].values.astype(float)
        
        values=[np.mean(scalar), np.mean(v1rVc), np.mean(v1rVc_v2), np.mean(v2rVc), np.mean(v2rVc_v2), np.mean(v4rVc), np.mean(v4rVc_v2)]
    else:
        scalar=df[df['type'] == types]['scalar'].values.astype(float)
        v1rVc=df[df['type'] == types]['1rVc'].values.astype(float)
        v2rVc=df[df['type'] == types]['2rVc'].values.astype(float)
        v4rVc=df[df['type'] == types]['4rVc'].values.astype(float)
        
        values=[np.mean(scalar), np.mean(v1rVc), np.mean(v2rVc), np.mean(v4rVc)]
    
    
    print('average ' + types)
    print(str(values))

    if types == 'double':
        row_index = 11
        col_index = 2
    else:
        row_index = 11
        col_index = 3
        
    subfig_used[row_index][col_index] = True

    # Create a bar plot for the current row
    ax = axes[row_index, col_index] if num_rows_plot > 1 else axes[col_index]
    bars = ax.bar(np.arange(len(values)), values, color=colors)
    #ax.set_xticks(range(len(values)))
    #ax.set_xticklabels([''] * len(values))
    # ax.set_xlabel('Implementation')
    if col_index == 0:
        ax.set_ylabel('GFlops/s')
    ax.set_title(f'Averages ({types})')
    
    # Set the major locator for the y-axis ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    
    # Calculate the maximum y-tick value
    max_value = max(values) * 1.2

    # Set the y-axis limits
    ax.set_ylim(top=max_value)
    
    # Format the y-axis tick labels to have consistent decimal precision
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    
    # Add numbers above the bars
    for idx, bar in enumerate(bars):
        if idx != 0:
            height = bar.get_height()
            speedup=values[idx]/values[0]
            ax.annotate(f'×{speedup:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom')

######################################################

# Remove any empty subplots
for row_index in range(num_rows_plot):
    for col_index in range(num_cols_plot):
        if not subfig_used[row_index][col_index]:
            fig.delaxes(axes[row_index, col_index])

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.savefig(filename + ".pdf", format="pdf")


#fig, ax = plt.subplots()
#for idx,label in enumerate(['Scalar', 'B(1,VEC_SIZE)', 'B(2,VEC_SIZE)', 'B(4,VEC_SIZE)']):# ['scalar', '1rVc', '2rVc', '4rVc']
#    plt.plot([], [],  color=colors[idx], marker='s', markerfacecolor=colors[idx], markeredgecolor=colors[idx], linestyle='-', label=label)
##ax.legend(ncol=2)
#plt.gca().set_axis_off()
#plt.savefig(filename + "_legend.pdf", format="pdf")

# Show the plot
#plt.show()

