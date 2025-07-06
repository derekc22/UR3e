
import numpy as np
import yaml
import shutil
import os
from datetime import datetime
from controller.plot import *


def cleanup(traj_target: np.array, 
            traj_true: np.array,
            ctrls: np.array,
            actuator_frc: np.array,
            trajectory_fpath: str, 
            log_fpath: str, 
            yml: dict,
            ctrl_mode: str, 
            **kwargs: any) -> None:

    dtn = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_fpath = log_fpath + dtn
    os.makedirs(save_fpath, exist_ok=True)
    
    if ctrl_mode in ("l", "l_task"):
        pos_errs, rot_errs = (kwargs.get("pos_errs"), kwargs.get("rot_errs"))
        plot_trajectory_l(traj_target, traj_true, pos_errs, rot_errs, save_fpath)
        plot_3d_trajectory(traj_target, traj_true, pos_errs, save_fpath)
        plot_2d_trajectory(traj_target, traj_true, pos_errs, save_fpath)
    else:
        qpos_errs = kwargs.get("qpos_errs")
        plot_trajectory_j(traj_target, traj_true, qpos_errs, save_fpath)
        
    plot_ctrl(ctrls, actuator_frc, save_fpath)
        
    with open(f"{save_fpath}/log.yml", 'w') as f: yaml.dump(yml, f)
    shutil.copy(trajectory_fpath, save_fpath)



def load_trajectory_file(trajectory_fpath: str) -> np.array:
    return np.genfromtxt(
        trajectory_fpath, 
        delimiter=',', 
        skip_header=1).reshape(-1, 7)

        
    
    


def build_interpolated_trajectory(n: int, 
                                  hold: int, 
                                  trajectory_fpath: str) -> np.array:
    
    traj = np.concatenate([
        np.zeros(shape=(1, 13)),
        load_trajectory_file(trajectory_fpath)
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




def build_trajectory(hold: int, 
                     trajectory_fpath: str) -> np.array:
    return np.repeat(
        load_trajectory_file(trajectory_fpath),
        hold, axis=0)