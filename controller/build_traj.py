import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
np.set_printoptions(
    linewidth=400,     # Wider output (default is 75)
    threshold=np.inf,  # Print entire array, no summarization with ...
    precision=8,       # Show more decimal places
    suppress=False     # Do not suppress small floating point numbers
)


def build_traj_l_point(trajectory_fpath: str,
                       start: np.ndarray, 
                       stop: np.ndarray = None) -> None:

    # Generate trajectory points
    num_points = 1000
    t = np.linspace(0, 1, num_points)
    
    if stop is None:
        stop = np.hstack([
            # np.random.uniform(-0.6, 0.6, size=2), np.random.uniform(0.1, 0.8), np.random.uniform(-np.pi, np.pi, size=3), 1
            np.random.uniform(-0.6, 0.6, size=2), np.random.uniform(0.7, 0.8), [0, 0, 0], 1
        ])
        
    traj = np.vstack([start, stop])
        
    # Generate control points
    t_control = [0, 1]
    
    # Create smooth interpolated trajectories
    x_interp = interp1d(t_control,  traj[:, 0], kind='linear')
    y_interp = interp1d(t_control,  traj[:, 1], kind='linear')
    z_interp = interp1d(t_control,  traj[:, 2], kind='linear')
    rx_interp = interp1d(t_control, traj[:, 3], kind='linear')
    ry_interp = interp1d(t_control, traj[:, 4], kind='linear')
    rz_interp = interp1d(t_control, traj[:, 5], kind='linear')
    g_interp =  interp1d(t_control, traj[:, 6], kind='linear')

    # Generate smooth trajectory    
    x = x_interp(t[1:])
    y = y_interp(t[1:])
    z = z_interp(t[1:])
    rx = rx_interp(t[1:])
    ry = ry_interp(t[1:])
    rz = rz_interp(t[1:])
    g = g_interp(t[1:])
    
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
    df.to_csv(trajectory_fpath, index=False)
    # print(f"Generated smooth trajectory with {num_points} points saved to {trajectory_fpath}")
    




def build_traj_l(trajectory_fpath: str) -> None:

    # Generate trajectory points
    num_points = 1000
    t = np.linspace(0, 1, num_points)

    # Generate random waypoints and interpolate for smoothness
    np.random.seed(42)  # For reproducible results

    # Define workspace bounds for UR3e
    x_bounds = [0.2, 0.5]
    y_bounds = [-0.3, 0.3]
    z_bounds = [0.4, 0.8]

    # Generate fewer control points for smooth interpolation
    num_control_points = 10
    t_control = np.linspace(0, 1, num_control_points)

    # Random control points
    x_control = np.random.uniform(x_bounds[0], x_bounds[1], num_control_points)
    y_control = np.random.uniform(y_bounds[0], y_bounds[1], num_control_points)
    z_control = np.random.uniform(z_bounds[0], z_bounds[1], num_control_points)

    # Smooth orientation changes
    rx_control = np.random.uniform(-0.5, 0.5, num_control_points)
    ry_control = np.random.uniform(-0.5, 0.5, num_control_points)
    rz_control = np.random.uniform(-np.pi, np.pi, num_control_points)

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
    df.to_csv(trajectory_fpath, index=False)
    print(f"Generated smooth trajectory with {num_points} points saved to {trajectory_fpath}")





def build_traj_j(trajectory_fpath: str) -> None:

    # Generate trajectory points
    num_points = 200
    t = np.linspace(0, 1, num_points)

    # Generate random waypoints and interpolate for smoothness
    np.random.seed(42)  # For reproducible results

    # Define workspace bounds for UR3e
    j1_bounds = [0.2, 0.5]
    j2_bounds = [-0.3, 0.3]
    j3_bounds = [0.4, 0.8]
    j4_bounds = [0.4, 0.8]
    j5_bounds = [0.4, 0.8]
    j6_bounds = [0.4, 0.8]

    # Generate fewer control points for smooth interpolation
    num_control_points = 10
    t_control = np.linspace(0, 1, num_control_points)

    # Random control points
    j1_control = np.random.uniform(j1_bounds[0], j1_bounds[1], num_control_points)
    j2_control = np.random.uniform(j2_bounds[0], j2_bounds[1], num_control_points)
    j3_control = np.random.uniform(j3_bounds[0], j3_bounds[1], num_control_points)
    j4_control = np.random.uniform(j4_bounds[0], j4_bounds[1], num_control_points)
    j5_control = np.random.uniform(j5_bounds[0], j5_bounds[1], num_control_points)
    j6_control = np.random.uniform(j6_bounds[0], j6_bounds[1], num_control_points)


    # Create smooth interpolated trajectories
    j1_interp = interp1d(t_control, j1_control, kind='cubic')
    j2_interp = interp1d(t_control, j2_control, kind='cubic')
    j3_interp = interp1d(t_control, j3_control, kind='cubic')
    j4_interp = interp1d(t_control, j4_control, kind='cubic')
    j5_interp = interp1d(t_control, j5_control, kind='cubic')
    j6_interp = interp1d(t_control, j6_control, kind='cubic')


    # Generate smooth trajectory
    j1 = j1_interp(t)
    j2 = j2_interp(t)
    j3 = j3_interp(t)
    j4 = j4_interp(t)
    j5 = j5_interp(t)
    j6 = j6_interp(t)


    # Gripper control (0 = open, 1 = closed)
    # g = np.zeros(num_points)  # Keep gripper open throughout trajectory
    # g cycles through 0, 0.5, 1
    g = np.tile([0, 0.5, 1], num_points // 3 + 1)[:num_points]



    # Create DataFrame
    df = pd.DataFrame({
        'j1': j1,
        'j2': j2,
        'j3': j3,
        'j4': j4,
        'j5': j5,
        'j6': j6,
        'g': g
    })

    # Save to CSV
    df.to_csv(trajectory_fpath, index=False)
    print(f"Generated smooth trajectory with {num_points} points saved to {trajectory_fpath}")





def build_gripless_traj_mug(start: np.ndarray, 
                     stop: np.ndarray) -> np.ndarray:

    # Generate trajectory points
    num_points = 1000
    t = np.linspace(0, 1, num_points)
    
    traj = np.vstack([start, stop])
            
    # Generate control points
    t_control = [0, 1]
    
    # Create smooth interpolated trajectories
    x_interp = interp1d(t_control,  traj[:, 0], kind='linear')
    y_interp = interp1d(t_control,  traj[:, 1], kind='linear')
    z_interp = interp1d(t_control,  traj[:, 2], kind='linear')
    rx_interp = interp1d(t_control, traj[:, 3], kind='linear')
    ry_interp = interp1d(t_control, traj[:, 4], kind='linear')
    rz_interp = interp1d(t_control, traj[:, 5], kind='linear')

    # Generate smooth trajectory    
    x = x_interp(t[1:])
    y = y_interp(t[1:])
    z = z_interp(t[1:])
    rx = rx_interp(t[1:])
    ry = ry_interp(t[1:])
    rz = rz_interp(t[1:])
    
    # print(x.shape)
    # exit()
    return np.vstack([
        x, y, z, rx, ry, rz   
    ]).T