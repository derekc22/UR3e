import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Generate trajectory points
num_points = 200
t = np.linspace(0, 1, num_points)

# Generate random waypoints and interpolate for smoothness
np.random.seed(42)  # For reproducible results

# Define workspace bounds for UR3e
x_bounds = [0.2, 0.5]
y_bounds = [-0.3, 0.3]
z_bounds = [0.4, 0.8]

# Generate fewer control points for smooth interpolation
control_points = 10
t_control = np.linspace(0, 1, control_points)

# Random control points
x_control = np.random.uniform(x_bounds[0], x_bounds[1], control_points)
y_control = np.random.uniform(y_bounds[0], y_bounds[1], control_points)
z_control = np.random.uniform(z_bounds[0], z_bounds[1], control_points)

# Smooth orientation changes
rx_control = np.random.uniform(-0.5, 0.5, control_points)
ry_control = np.random.uniform(-0.5, 0.5, control_points)
rz_control = np.random.uniform(-np.pi, np.pi, control_points)

# Create smooth interpolated trajectories
x_interp = interp1d(t_control, x_control, kind='cubic')
y_interp = interp1d(t_control, y_control, kind='cubic')
z_interp = interp1d(t_control, z_control, kind='cubic')
rx_interp = interp1d(t_control, rx_control, kind='cubic')
ry_interp = interp1d(t_control, ry_control, kind='cubic')
rz_interp = interp1d(t_control, rz_control, kind='cubic')

# Generate smooth trajectory
x = x_interp(t)
y = y_interp(t)
z = z_interp(t)
rx = rx_interp(t)
ry = ry_interp(t)
rz = rz_interp(t)

# Gripper control (0 = open, 1 = closed)
# g = np.zeros(num_points)  # Keep gripper open throughout trajectory
# g cycles through 0, 0.5, 1
g = np.tile([0, 0, 0], num_points // 3 + 1)[:num_points]


rx = np.tile([0, 0, 0], num_points // 3 + 1)[:num_points]
ry = np.tile([0, 0, 0], num_points // 3 + 1)[:num_points]
rz = np.tile([0, 0, 0], num_points // 3 + 1)[:num_points]


# Create DataFrame
df = pd.DataFrame({
    'x': x,
    'y': y,
    'z': z,
    'rx': rx,
    'ry': ry,
    'rz': rz,
    'g': g
})

# Save to CSV
df.to_csv("mujoco/trajectory.csv", index=False)
print(f"Generated smooth trajectory with {num_points} points saved to trajectory.csv")

