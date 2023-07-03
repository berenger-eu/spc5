import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

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

# Iterate over each row in the DataFrame
for i, row in df.iterrows():
    # Get the data for the current row
    matrixname = row['matrixname']
    types = row['type']
    values = row[['scalar', '1rVc', '2rVc', '4rVc']].astype(float)

    # Calculate the subplot indices for the current row
    row_index = i // num_cols_plot
    col_index = i % num_cols_plot

    # Create a bar plot for the current row
    ax = axes[row_index, col_index] if num_rows_plot > 1 else axes[col_index]
    bars = ax.bar(values.index, values, color=colors)
    # ax.set_xlabel('Implementation')
    if col_index == 0:
        ax.set_ylabel('GFlops/s')
    ax.set_title(f'{matrixname} ({types})')
    
    # Set the major locator for the y-axis ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    
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

# Remove any empty subplots
# fig.delaxes(axes[row_index, col_index])

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.savefig(filename + ".pdf", format="pdf")

# Show the plot
#plt.show()

