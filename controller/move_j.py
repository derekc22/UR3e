import mujoco
import numpy as np
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf)
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
from controller.controller_utils import get_joint_space_state, pd_joint_ctrl, grip_ctrl, update_errs
from utils import load_model, reset, get_ur3e_qpos, get_joint_torques
from controller.build_traj import build_traj_j
from controller.aux import load_trajectory, cleanup
import yaml


    
def ctrl(t: int, 
         m: mujoco.MjModel, 
         d: mujoco.MjData,
         traj_t: np.ndarray,
         qpos_gains: dict,
         qpos_errs: np.ndarray) -> np.ndarray:
    
    qpos_u = pd_joint_ctrl(t, m, d, traj_t[:-1], qpos_gains, get_joint_delta, qpos_errs)    
    grip_u = grip_ctrl(m, traj_t[-1])
    
    return np.hstack([
        qpos_u,
        grip_u
    ])
    

def get_joint_delta(t: int, 
                    m: mujoco.MjModel, 
                    d: mujoco.MjData, 
                    qpos_target: np.ndarray,
                    qpos_errs: np.ndarray) -> np.ndarray:
    
    qpos_delta = qpos_target - get_ur3e_qpos(d)
    update_errs(t, qpos_errs, qpos_delta)
    return qpos_delta
    


def main():
    
    model_path = "assets/ur3e_2f85.xml"
    trajectory_fpath = "controller/data/traj_j.csv"
    config_fpath = "controller/config/config_j.yml"
    log_fpath = "controller/logs/logs_j/"
    ctrl_mode = "j"

    with open(config_fpath, "r") as f: yml = yaml.safe_load(f)
    qpos_gains = { k:np.diag(v) for k, v in yml["qpos"].items() } 
    hold = yml["hold"]

    # ur3e  = 6  nq, 6  nv, 6 nu
    # 2f85  = 8  nq, 8  nv, 1 nu
    # total = 14 nq, 14 nv, 7 nu
    m, d = load_model(model_path)
    reset(m, d)

    build_traj_j(get_joint_space_state(d), hold, trajectory_fpath)
    traj_target = load_trajectory(trajectory_fpath)
    T = traj_target.shape[0]
    traj_true = np.zeros_like(traj_target)

    qpos_errs = np.zeros(shape=(T, 6))
    ctrls = np.zeros(shape=(T, m.nu))
    actuator_frc = np.zeros(shape=(T, m.nu))

    viewer = mujoco.viewer.launch_passive(m, d)
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            
    save_flag = True

    try:
        for t in range(T):
            viewer.sync()
    
            u = ctrl(t, m, d, traj_target[t, :], qpos_gains, qpos_errs)
            d.ctrl = u
            ctrls[t] = u
            
            mujoco.mj_step(m, d)
            
            traj_true[t] = get_joint_space_state(d)
            actuator_frc[t] = get_joint_torques(d)

            # print(f"qpos_target: {traj_target[t, :7]}, pos_true: {traj_true[t, :7]}, pos_err: {qpos_errs[t, :]}")
            # print(f"grip_target: {traj_target[t, -1]}")
            # print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            
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