import matplotlib.pyplot as plt
import numpy as np

# Sample data
steps = [1000, 2000, 3000]
latency_data = [
    [30, 25, 20],  # Data for each bar group representing training latency
    [25, 20, 15],
    [20, 15, 10]
]
running_time = [31, 35, 40]  # Sample data for running time

# Set up the bar chart
bar_width = 0.2  # Width of each bar
index = np.arange(len(steps))  # The x locations for the group of bars

fig, ax1 = plt.subplots()

# Colors for different bars
colors = ['gray', 'silver', 'lightgray']

# Plot each set of bars for training latency
for i, data in enumerate(latency_data):
    ax1.bar(index + i * bar_width, data, bar_width, color=colors[i], edgecolor='black', hatch='//')

# Set labels and title for training latency
ax1.set_xlabel("Step Number")
ax1.set_ylabel("Training Latency")
ax1.set_title("Training Latency and Running Time vs Step Number")
ax1.set_xticks(index + bar_width)
ax1.set_xticklabels(steps)

# Adding legend for training latency
ax1.legend(["Model 1", "Model 2", "Model 3"], loc="upper right")

# Create a secondary y-axis for running time
ax2 = ax1.twinx()
ax2.plot(index + bar_width, running_time, color="black", marker="o", linestyle="--", linewidth=2)
ax2.set_ylabel("Running Time")

# Manually set the range for the running time axis
ax2.set_ylim(30, 60)  # Set your desired range here (min_value, max_value)

# Adding legend for running time
ax2.legend(["Running Time"], loc="upper left")

# Save as PDF vector file
plt.savefig("training_latency_running_time_chart.pdf", format="pdf")
plt.show()