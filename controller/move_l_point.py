import mujoco
import numpy as np
import matplotlib
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf)
matplotlib.use('Agg')  # Set backend to non-interactive
from controller.controller_utils import (
    get_task_space_state,
    grip_ctrl, update_errs, update_tot_errs
)
from utils import (
    load_model, reset, get_site_id,
    get_site_xpos, get_site_R, get_joint_torques,
    get_jnt_ranges
)
from controller.move_l_task import ctrl
from controller.build_traj import build_traj_l_point
from controller.aux import load_trajectory, cleanup
import yaml
from scipy.spatial.transform import Rotation as R
import time



def main():
    
    model_path = "assets/ur3e_2f85.xml"
    trajectory_fpath = "controller/data/traj_l_point.csv"
    config_path = "controller/config/config_l_point.yml"
    log_fpath = "controller/logs/logs_l_point/"
    ctrl_mode = "l_point"

    with open(config_path, "r") as f: yml = yaml.safe_load(f)
    pos_gains = { k:np.diag(v) for k, v in yml["pos"].items() } 
    rot_gains = { k:np.diag(v) for k, v in yml["rot"].items() } 
    hold = yml["hold"]

    # ur3e  = 6  nq, 6  nv, 6 nu
    # 2f85  = 8  nq, 8  nv, 1 nu
    # total = 14 nq, 14 nv, 7 nu
    m, d = load_model(model_path)
    
    mujoco.mj_step(m, d) # step the simulation to initialize it
    print(get_task_space_state(m, d))
    build_traj_l_point(trajectory_fpath, get_task_space_state(m, d))
    traj_target = load_trajectory(hold, trajectory_fpath)
    T = traj_target.shape[0]
    traj_true = np.zeros_like(traj_target)

    jnt_ranges = get_jnt_ranges(m) # NEEDED IN THIS CASE?
    pos_errs = np.zeros(shape=(T, 3))
    rot_errs = np.zeros(shape=(T, 3))
    grip_errs = np.zeros(shape=(T, 1)) # NEEDED?
    ctrls = np.zeros(shape=(T, m.nu))
    actuator_frc = np.zeros(shape=(T, m.nu))

    tot_pos_errs = np.zeros(shape=(3, ))
    tot_rot_errs = np.zeros(shape=(3, ))

    viewer = mujoco.viewer.launch_passive(m, d)
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            
    reset(m, d)
    save_flag = True
    
    try:
        for t in range(T):
            viewer.sync()
            
            if t % hold == 0: 
                # print(t)
                tot_pos_errs.fill(0)
                tot_rot_errs.fill(0)
            
            u = ctrl(t, m, d, traj_target[t, :], pos_gains, rot_gains, pos_errs, rot_errs, tot_pos_errs, tot_rot_errs)
            d.ctrl = u
            ctrls[t] = u
            
            mujoco.mj_step(m, d)
            
            traj_true[t] = get_task_space_state(m, d)
            actuator_frc[t] = get_joint_torques(d)
            
            # print(ctrls[:3, :] - actuator_frc[:3, :])
            
            # print(f"pos_target: {traj_target[t, :3]}, pos_true: {traj_true[t, :3]}, pos_err: {pos_errs[t, :]}")
            # print(f"rot_target: {traj_target[t, 3:6]}, rot_true: {traj_true[t, 3:6]}, rot_err: {rot_errs[t, :]}")
            # print(f"grip_target: {traj_target[t, -1]}")
            # print("------------------------------------------------------------------------------------------")
            
            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        save_flag = False 
        raise(e)
    finally: 
        viewer.close()
        if save_flag: 
            cleanup(traj_target, traj_true, ctrls, actuator_frc, trajectory_fpath, log_fpath, yml, ctrl_mode, pos_errs=pos_errs, rot_errs=rot_errs)

        
        
    
        
    
    
    
if __name__ == "__main__":
    
    main()