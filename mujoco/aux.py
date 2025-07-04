
import numpy as np
import yaml
import shutil
import os
from datetime import datetime
from mujoco.plot import plot_trajectory_l, plot_trajectory_j, plot_2d_trajectory, plot_3d_trajectory
from utils import axis_angle_to_R


def cleanup(traj_target, traj_true, T, 
            trajectory_fpath, log_fpath, yml,
            ctrl_mode, **kwargs
            ):

    dtn = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_fpath = log_fpath + dtn
    os.makedirs(save_fpath, exist_ok=True)
    
    if ctrl_mode == "l":
        pos_errs, rot_errs = (kwargs.get("pos_errs"), kwargs.get("rot_errs"))
        plot_trajectory_l(traj_target, traj_true, pos_errs, rot_errs, T, save_fpath)
        plot_3d_trajectory(traj_target, traj_true, pos_errs, save_fpath)
        plot_2d_trajectory(traj_target, traj_true, pos_errs, save_fpath)
    else:
        qpos_errs = kwargs.get("qpos_errs")
        plot_trajectory_j(traj_target, traj_true, qpos_errs, T, save_fpath)
        
    with open(f"{save_fpath}/log.yml", 'w') as f: yaml.dump(yml, f)
    shutil.copy(trajectory_fpath, save_fpath)



def load_trajectory_file(trajectory_fpath, ctrl_mode):
    traj = np.genfromtxt(trajectory_fpath, delimiter=',', skip_header=1).reshape(-1, 7)
    

    if ctrl_mode == "l":
        # euler angles are specified as roll, pitch, yaw in the csv. 
        # euler_to_R(yaw, pitch, roll) is expected. Thus, flip it with negative slicing
        euler_angles = traj[:, 3:6] 
        
        rot_matrix = np.apply_along_axis(
            axis_angle_to_R, 
            axis=1, arr=euler_angles
        )
            
        return np.concatenate(
            [traj[:, 0:3], rot_matrix, traj[:, 6:]],
            axis=1
        )
    
    return traj
        
    
    


def build_interpolated_trajectory(n, hold, trajectory_fpath, ctrl_mode):
    
    traj = np.concatenate([
        np.zeros(shape=(1, 13)),
        load_trajectory_file(trajectory_fpath, ctrl_mode)
    ], axis=0)

    nrow = traj.shape[0]

    # Interpolation factors: [1/(n+1), 2/(n+1), ..., n/(n+1)]
    alphas = np.linspace(0, 1, n + 2)[1:-1]  # exclude 0 and 1

    # Temporary list to hold the final trajectory
    result = []

    for i in range(nrow - 1):
        start = traj[i]
        end = traj[i + 1]

        # Append the start point 'hold' times
        if i > 0: result.extend([start] * hold)
        else: result.extend([start])

        # Interpolated points between start and end (not held)
        interpolated_rows = start + (end - start)[None, :] * alphas[:, None]
        result.extend(interpolated_rows)

    # Append the last waypoint 'hold' times
    result.extend([traj[-1]] * hold)

    return np.vstack(result)




def build_trajectory(hold, trajectory_fpath, ctrl_mode):

    # traj = load_trajectory_file()
    # if hold > 1:
    #     traj = np.repeat(traj, hold, axis=0)
            
    return np.repeat(load_trajectory_file(trajectory_fpath, ctrl_mode), hold, axis=0)