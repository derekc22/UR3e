import mujoco
import numpy as np
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf)
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
from controller.controller_utils import (
    get_arm_qpos, get_joint_space_state,
    pd_ctrl, grip_ctrl, update_errs)
from utils import (
    load_model, reset, get_joint_torques
)
from gen_traj import gen_traj_j
from controller.aux import build_trajectory, build_interpolated_trajectory, cleanup
import yaml


    


def ctrl(t: int, 
         m: mujoco.MjModel, 
         d: mujoco.MjData,
         traj_t: np.array,
         qpos_gains: dict,
         qpos_errs: np.array,
         jnt_ranges: np.array) -> np.array:
    
    qpos_u = pd_ctrl(t, m, d, traj_t[:-1], qpos_gains, get_qpos_err, qpos_errs, jnt_ranges)    
    grip_u = grip_ctrl(m, traj_t[-1])
    
    return np.hstack([
        qpos_u,
        grip_u
    ])
    





def get_qpos_err(t: int, 
                 m: mujoco.MjModel, 
                 d: mujoco.MjData, 
                 qpos_target: np.array,
                 qpos_errs: np.array) -> np.array:
    
    qpos_delta = qpos_target - get_arm_qpos(d)
    update_errs(t, qpos_errs, qpos_delta)
    return qpos_delta
    









def main():
    
    model_path = "assets/ur3e_2f85.xml"
    trajectory_fpath = "controller/data/traj_j.csv"
    config_fpath = "controller/config/config_j.yml"
    log_fpath = "controller/logs/logs_j/"
    ctrl_mode = "j"
    num_ur3e_joints = 6

    with open(config_fpath, "r") as f: yml = yaml.safe_load(f)
    qpos_gains = { k:np.diag(v) for k, v in yml["qpos"].items() } 
    hold = yml["hold"]
    n = yml["n"]

    # ur3e  = 6  nq, 6  nv, 6 nu
    # 2f85  = 8  nq, 8  nv, 1 nu
    # total = 14 nq, 14 nv, 7 nu
    m, d = load_model(model_path)

    gen_traj_j()
    traj_target = build_interpolated_trajectory(n, hold, trajectory_fpath) if n else build_trajectory(hold, trajectory_fpath)
    T = traj_target.shape[0]
    traj_true = np.zeros_like(traj_target)

    jnt_ranges = m.jnt_range[:num_ur3e_joints]
    qpos_errs = np.zeros(shape=(T, num_ur3e_joints))
    ctrls = np.zeros(shape=(T, m.nu))
    actuator_frc = np.zeros(shape=(T, m.nu))

    viewer = mujoco.viewer.launch_passive(m, d)
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            
    reset(m, d)
    save_flag = True

    try:
        for t in range (T):
            viewer.sync()
    
            u = ctrl(t, m, d, traj_target[t, :], qpos_gains, qpos_errs, jnt_ranges)
            d.ctrl = u
            ctrls[t] = u
            
            mujoco.mj_step(m, d)
            
            traj_true[t] = get_joint_space_state(m, d)
            actuator_frc[t] = get_joint_torques(d)
            
            # print(ctrls[:3, :] - actuator_frc[:3, :])

            # print(f"qpos_target: {traj_target[t, :7]}, pos_true: {traj_true[t, :7]}, pos_err: {qpos_errs[t, :]}")
            # print(f"grip_target: {traj_target[t, -1]}")
            # print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            
            # time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        save_flag = False 
        raise(e)
    finally: 
        viewer.close()
        if save_flag: 
            cleanup(traj_target, traj_true, ctrls, actuator_frc, trajectory_fpath, log_fpath, yml, ctrl_mode, qpos_errs=qpos_errs)

        
        
    
        
    
    
    
if __name__ == "__main__":
    
    main()