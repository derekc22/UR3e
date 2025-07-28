import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from utils import *
import os
np.set_printoptions(
    linewidth=400,     # Wider output (default is 75)
    threshold=np.inf,  # Print entire array, no summarization with ...
    precision=8,       # Show more decimal places
    suppress=False     # Do not suppress small floating point numbers
)




def build_traj_l_hold(start: np.ndarray, 
                      hold: int,
                      trajectory_fpath: str = None) -> None:

    traj = np.repeat(np.expand_dims(start, axis=0), repeats=1000*hold, axis=0)
    
    if not trajectory_fpath:
        return traj
    
    save_traj(traj, trajectory_fpath, ctrl_mode="l")
    

def build_traj_l_pick_place(start: np.ndarray, 
                            destinations: list[np.ndarray],
                            hold: int,
                            trajectory_fpath: str = None) -> None:

    # FOR PD CONTROL, HOLD CAN BE NON-STANDARDIZED
    pick, place = destinations
    traj_pick = build_traj_l_point_custom(start, pick, hold=120)
    # traj_place = build_traj_l_point_custom(traj_pick[-1, :], place, hold=5)

    up = traj_pick[-1, :] + [0, 0, 0.2, 0, 0, 0, 1]
    traj_up = build_traj_l_point_custom(traj_pick[-1, :], up, hold=120)
    
    place += [0, 0, 0.025, 0, 0, 0, 0]
    traj_place = build_traj_l_point_custom(traj_up[-1, :], place, hold=120)
    
    end_drop = np.append(traj_place[-1, :-1], 0)
    traj_drop = build_traj_l_point_custom(traj_place[-1, :], end_drop, hold=120)
    
    traj = np.vstack([
        traj_pick,
        traj_up, 
        traj_place,
        traj_drop
    ])
    

    if not trajectory_fpath:
        return traj
    
    save_traj(traj, trajectory_fpath, ctrl_mode="l")




def build_traj_l_pick_move_place(start: np.ndarray, 
                                destinations: list[np.ndarray],
                                hold: int,
                                trajectory_fpath: str = None) -> None:

    # FOR PID CONTROL, HOLD MUST BE STANDARDIZED
    # pick, place = destinations
    # traj_pick = build_traj_l_point_custom(start, pick, hold)
    # traj_move = build_traj_l(traj_pick[-1, :], hold)
    # traj_place = build_traj_l_point_custom(traj_move[-1, :], place, hold)
    
    # drop = np.append(traj_place[-1, :-1], 0)
    # traj_drop = build_traj_l_point_custom(traj_place[-1, :], drop, hold)

    # FOR PD CONTROL, HOLD CAN BE NON-STANDARDIZED
    pick, place = destinations
    traj_pick = build_traj_l_point_custom(start, pick, hold)
    
    traj_move = build_traj_l(traj_pick[-1, :], hold=120)
    
    traj_place = build_traj_l_point_custom(traj_move[-1, :], place, hold=50)
    
    drop = np.append(traj_place[-1, :-1], 0)
    traj_drop = build_traj_l_point_custom(traj_place[-1, :], drop, hold=5)
    
    traj = np.vstack([
        traj_pick, 
        traj_move,
        traj_place,
        traj_drop
    ])
    
    save_traj(traj, trajectory_fpath, ctrl_mode="l")



def build_traj_l_point_custom(start: np.ndarray, 
                              stop: np.ndarray,
                              hold: int) -> None:

    # Generate trajectory points
    # num_points = 5
    num_points = 30
    t = np.linspace(0, 1, num_points+1)
            
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
    # Slice at t[1:] to avoid holding at the start point    
    x = x_interp(t[1:])
    y = y_interp(t[1:])
    z = z_interp(t[1:])
    rx = rx_interp(t[1:])
    ry = ry_interp(t[1:])
    rz = rz_interp(t[1:])
    g = g_interp(t[1:])
    
    # g = np.linspace(start[-1], stop[-1], num_points)
    
    return np.repeat(np.vstack([
        x, y, z, rx, ry, rz, g]).T, 
        repeats=hold, axis=0
    )

    


    

def build_traj_l_point(start: np.ndarray, 
                       hold: int,
                       trajectory_fpath: str = None) -> None:

    # Generate trajectory points
    num_points = 500
    t = np.linspace(0, 1, num_points+1)

    stop = np.hstack([ np.random.uniform(-0.6, 0.6, size=2), np.random.uniform(0.7, 0.8), [0, 0, 0], 1])

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
    # Slice at t[1:] to avoid holding at the start point    
    x = x_interp(t[1:])
    y = y_interp(t[1:])
    z = z_interp(t[1:])
    rx = rx_interp(t[1:])
    ry = ry_interp(t[1:])
    rz = rz_interp(t[1:])
    g = g_interp(t[1:])
    
    # g = np.linspace(0, 1, num_points)

    traj = np.repeat(np.vstack(
        [x, y, z, rx, ry, rz, g]).T,
        repeats=hold, axis=0
    )

    if not trajectory_fpath:
        return traj
    
    save_traj(traj, trajectory_fpath, ctrl_mode="l")




def build_traj_l(start: np.ndarray, 
                 hold: int,
                 trajectory_fpath: str = None) -> None:

    # Generate trajectory points
    num_points = 500
    t = np.linspace(0, 1, num_points)

    # Generate random waypoints and interpolate for smoothness
    # np.random.seed(42)  # For reproducible results
    np.random.seed(49)  # For reproducible results

    # Define workspace bounds for UR3e
    x_bounds = [0.2, 0.5]
    y_bounds = [-0.3, 0.3]
    z_bounds = [0.4, 0.8]
    rx_bounds = [-0.1, 0.1]
    ry_bounds = [-0.1, 0.1]
    rz_bounds = [-np.pi/4, np.pi/4]

    # Generate fewer control points for smooth interpolation
    num_control_points = 10
    t_control = np.linspace(0, 1, num_control_points+1)

    # Random control points
    x_control = np.concatenate([start[0:1], np.random.uniform(x_bounds[0], x_bounds[1], num_control_points)])
    y_control = np.concatenate([start[1:2], np.random.uniform(y_bounds[0], y_bounds[1], num_control_points)])
    z_control = np.concatenate([start[2:3], np.random.uniform(z_bounds[0], z_bounds[1], num_control_points)])

    # Smooth orientation changes
    rx_control = np.concatenate([start[3:4], np.random.uniform(rx_bounds[0], rx_bounds[1], num_control_points)])
    ry_control = np.concatenate([start[4:5], np.random.uniform(ry_bounds[0], ry_bounds[1], num_control_points)])
    rz_control = np.concatenate([start[5:6], np.random.uniform(rz_bounds[0], rz_bounds[1], num_control_points)])

    # Create smooth interpolated trajectories
    x_interp = interp1d(t_control, x_control, kind='cubic')
    y_interp = interp1d(t_control, y_control, kind='cubic')
    z_interp = interp1d(t_control, z_control, kind='cubic')
    rx_interp = interp1d(t_control, rx_control, kind='cubic')
    ry_interp = interp1d(t_control, ry_control, kind='cubic')
    rz_interp = interp1d(t_control, rz_control, kind='cubic')

    # Generate smooth trajectory
    # Slice at t[1:] to avoid holding at the start point
    x = x_interp(t)
    y = y_interp(t)
    z = z_interp(t)
    rx = rx_interp(t)
    ry = ry_interp(t)
    rz = rz_interp(t)

    rx = np.full(num_points, -1.209)
    ry = np.full(num_points, -1.209)
    rz = np.full(num_points, 1.209)
    
    # Gripper control (0 = open, 1 = closed)
    # g cycles through 0, 0.5, 1
    # g = np.tile([0, 0, 0], num_points // 3 + 1)[:num_points]
    # g = np.tile([0, 0.5, 1], num_points // 3 + 1)[:num_points]
    g = np.tile([1, 1, 1], num_points // 3 + 1)[:num_points]


    traj = np.repeat(np.vstack(
        [x, y, z, rx, ry, rz, g]).T,
        repeats=hold, axis=0
    )

    if not trajectory_fpath:
        return traj
    
    save_traj(traj, trajectory_fpath, ctrl_mode="l")






def build_traj_j(start: np.ndarray, 
                 hold: int,
                 trajectory_fpath: str = None) -> None:

    # Generate trajectory points
    num_points = 500
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
    t_control = np.linspace(0, 1, num_control_points+1)

    # Random control points
    j1_control = np.concatenate([start[0:1], np.random.uniform(j1_bounds[0], j1_bounds[1], num_control_points)])
    j2_control = np.concatenate([start[1:2], np.random.uniform(j2_bounds[0], j2_bounds[1], num_control_points)])
    j3_control = np.concatenate([start[2:3], np.random.uniform(j3_bounds[0], j3_bounds[1], num_control_points)])
    j4_control = np.concatenate([start[3:4], np.random.uniform(j4_bounds[0], j4_bounds[1], num_control_points)])
    j5_control = np.concatenate([start[4:5], np.random.uniform(j5_bounds[0], j5_bounds[1], num_control_points)])
    j6_control = np.concatenate([start[5:6], np.random.uniform(j6_bounds[0], j6_bounds[1], num_control_points)])


    # Create smooth interpolated trajectories
    j1_interp = interp1d(t_control, j1_control, kind='cubic')
    j2_interp = interp1d(t_control, j2_control, kind='cubic')
    j3_interp = interp1d(t_control, j3_control, kind='cubic')
    j4_interp = interp1d(t_control, j4_control, kind='cubic')
    j5_interp = interp1d(t_control, j5_control, kind='cubic')
    j6_interp = interp1d(t_control, j6_control, kind='cubic')


    # Generate smooth trajectory
    # Slice at t[1:] to avoid holding at the start point
    j1 = j1_interp(t)
    j2 = j2_interp(t)
    j3 = j3_interp(t)
    j4 = j4_interp(t)
    j5 = j5_interp(t)
    j6 = j6_interp(t)


    # Gripper control (0 = open, 1 = closed)
    # g = np.zeros(num_points)  # Keep gripper open throughout trajectory
    # g cycles through 0, 0.5, 1
    g = np.tile([0, 0, 0], num_points // 3 + 1)[:num_points]
    # g = np.tile([0, 0.5, 1], num_points // 3 + 1)[:num_points]



    # # Create DataFrame
    # df = pd.DataFrame({
    #     'j1': j1,
    #     'j2': j2,
    #     'j3': j3,
    #     'j4': j4,
    #     'j5': j5,
    #     'j6': j6,
    #     'g': g
    # })

    # # Save to CSV
    # df.to_csv(trajectory_fpath, index=False)
    # print(f"Generated smooth trajectory with {num_points} points saved to {trajectory_fpath}")
    

    traj = np.repeat(np.vstack(
        [j1, j2, j3, j4, j5, j6, g]).T,
        repeats=hold, axis=0
    )

    if not trajectory_fpath:
        return traj
    
    save_traj(traj, trajectory_fpath, ctrl_mode="j")



# @DEPRECATED
def build_gripless_traj_gym(start: np.ndarray, 
                            stop: np.ndarray,
                            hold: int) -> np.ndarray:

    # Generate trajectory points
    num_points = 100
    t = np.linspace(0, 1, num_points+1)
    
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
    # Slice at t[1:] to avoid holding at the start point    
    x = x_interp(t[1:])
    y = y_interp(t[1:])
    z = z_interp(t[1:])
    rx = rx_interp(t[1:])
    ry = ry_interp(t[1:])
    rz = rz_interp(t[1:])
    

    return np.repeat(np.vstack(
        [x, y, z, rx, ry, rz]).T,
        repeats=hold, axis=0
    )





def save_traj(traj: np.ndarray,
              trajectory_fpath: str,
              ctrl_mode: str) -> None:
    
    columns = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'g'] if ctrl_mode == 'j' else ['x', 'y', 'z', 'rx', 'ry', 'rz', 'g']
    df = pd.DataFrame(traj, columns=columns)

    # Save to CSV
    os.makedirs("controller/data/", exist_ok=True)
    df.to_csv(trajectory_fpath, index=False)
    print(f"Generated smooth trajectory saved to {trajectory_fpath}")