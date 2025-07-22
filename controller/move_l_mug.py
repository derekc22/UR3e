import mujoco
import numpy as np
import matplotlib
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf)
matplotlib.use('Agg')  # Set backend to non-interactive
from controller.controller_utils import (
    get_task_space_state, pid_task_ctrl
)
from utils import (
    load_model, get_joint_torques,
)
from controller.build_traj import build_traj_l_point
from controller.aux import load_trajectory, plot_plots, get_dtn
from gymnasium_env.gymnasium_env_utils import (
    get_mug_xpos, reset_mug_stochastic, 
    init_collision_cache, get_robot_collision
)
import yaml
from scipy.spatial.transform import Rotation as R
import time
    


def main():
    r, R = (0, 30)
    
    model_path = "assets/main.xml"
    trajectory_fpath = "controller/data/traj_l_mug.csv"
    config_path = "controller/config/config_l_mug.yml"
    log_fpath = "controller/logs/logs_l_mug/"
    ctrl_mode = "l_mug"
    
    with open(config_path, "r") as f: yml = yaml.safe_load(f)
    pos_gains = { k:np.diag(v) for k, v in yml["pos"].items() } 
    rot_gains = { k:np.diag(v) for k, v in yml["rot"].items() } 
    hold = yml["hold"]
    
    m, d = load_model(model_path)
    
    viewer = mujoco.viewer.launch_passive(m, d)
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    
    cc = init_collision_cache(m)
    
    while r < R:
        reset_mug_stochastic(m, d)

        stop = np.hstack([
            get_mug_xpos(m, d), [0, 0, 0], 1
            # get_mug_xpos(m, d), np.random.uniform(-np.pi, np.pi, size=3), 1
        ])
        build_traj_l_point(trajectory_fpath, get_task_space_state(m, d), stop)
        # build_traj_l_point(trajectory_fpath, get_task_space_state(m, d))
        
        traj_target = load_trajectory(hold, trajectory_fpath)
        T = traj_target.shape[0]
        traj_true = np.zeros_like(traj_target)

        pos_errs = np.zeros(shape=(T, 3))
        rot_errs = np.zeros(shape=(T, 3))
        ctrls = np.zeros(shape=(T, m.nu))
        actuator_frc = np.zeros(shape=(T, m.nu))

        tot_pos_errs = np.zeros(shape=(3, ))
        tot_rot_errs = np.zeros(shape=(3, ))
        
        for t in range(T):
            viewer.sync()
            
            if t % hold == 0: 
                tot_pos_errs.fill(0)
                tot_rot_errs.fill(0)
            
            u = pid_task_ctrl(t, m, d, traj_target[t, :], pos_gains, rot_gains, pos_errs, rot_errs, tot_pos_errs, tot_rot_errs)
            d.ctrl = u
            ctrls[t] = u
            
            mujoco.mj_step(m, d)
            
            traj_true[t] = get_task_space_state(m, d)
            actuator_frc[t] = get_joint_torques(d)
            
            # print(f"pos_target: {traj_target[t, :3]}, pos_true: {traj_true[t, :3]}, pos_err: {pos_errs[t, :]}")
            # print(f"rot_target: {traj_target[t, 3:6]}, rot_true: {traj_true[t, 3:6]}, rot_err: {rot_errs[t, :]}")
            # print("------------------------------------------------------------------------------------------")
            
            print(get_robot_collision(m, d, cc))
            
        time.sleep(1)
        # plot_plots(traj_target, traj_true, ctrls, actuator_frc, ctrl_mode, log_fpath=log_fpath, pos_errs=pos_errs, rot_errs=rot_errs)
        r += 1
            
            

        
        
    
        
    
    
if __name__ == "__main__":
    
    main()