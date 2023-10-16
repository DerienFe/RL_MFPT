import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
df = pd.read_csv("mm_metrics.csv")

# Initialize empty lists to store the average time and standard deviation for each method
avg_times = []
std_devs = []

# List of methods to consider
methods = ["Explore", "Bias", "Unbias", "Metadynamics"]

# Calculate the average time and standard deviation for each method
for method in methods:
    subset_df = df[df['Run Type'] == method]
    avg_time = subset_df['Time'].mean()  # Convert to nanoseconds
    std_dev = subset_df['Time'].std()  # Convert to nanoseconds
    avg_times.append(avg_time)
    std_devs.append(std_dev)

# Create the bar chart
x_pos = np.arange(len(methods))
plt.bar(x_pos, avg_times, yerr=std_devs, alpha=0.7, capsize=10, color=['r', 'g', 'b', 'y'])

# Label the bars
plt.xticks(x_pos, methods)

# Label the axes and provide a title
plt.xlabel('Method')
plt.ylabel('Average Run Time (ps)')
plt.title('Average Run Time for Each Method to Reach 7 Amstrong')

# Save the figure as a PNG file before showing it
plt.savefig("average_run_time_bar_chart.png")

# Show the plot
plt.show()
