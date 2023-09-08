import numpy as np
import matplotlib.pyplot as plt

# Generating random example data
np.random.seed(0)
time_steps = 10
n_trajectories = 5

# Simulating trajectories using random walk
trajectories = np.cumsum(np.random.randn(time_steps, n_trajectories), axis=0)

# Calculating the mean and standard deviation at each time step
mean_trajectory = np.mean(trajectories, axis=1)
std_trajectory = np.std(trajectories, axis=1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(mean_trajectory, label='Average Trajectory', color='blue')
plt.fill_between(range(time_steps), 
                 mean_trajectory - std_trajectory, 
                 mean_trajectory + std_trajectory, 
                 color='blue', alpha=0.2, label='1 std deviation')
for i in range(n_trajectories):
    plt.plot(trajectories[:, i], color='gray', linestyle='--', alpha=0.5)

plt.title('Average Trajectory with Standard Deviation Shading')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()