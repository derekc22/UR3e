import mujoco
import numpy as np
import matplotlib
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf)
matplotlib.use('Agg')  # Set backend to non-interactive
from controller.controller_utils import get_task_space_state, pd_joint_ctrl, grip_ctrl, get_pos_err, get_rot_err
from utils import load_model, reset, get_site_id, get_joint_torques
from controller.build_traj import build_traj_l
from controller.aux import load_trajectory, cleanup
import yaml
from scipy.spatial.transform import Rotation as R

    

def ctrl(t: int,
         m: mujoco.MjModel, 
         d: mujoco.MjData,
         traj_t: np.ndarray,
         pos_gains: dict,
         rot_gains: dict,
         pos_errs: np.ndarray,
         rot_errs: np.ndarray) -> np.ndarray:
    
    pos_u = pd_joint_ctrl(t, m, d, traj_t[:3], pos_gains, get_pos_joint_delta, pos_errs)    
    rot_u = pd_joint_ctrl(t, m, d, traj_t[3:6], rot_gains, get_rot_joint_delta, rot_errs)  # rotation matrix
    grip_u = grip_ctrl(m, traj_t[-1])
    
    return np.hstack([
        pos_u + rot_u,
        grip_u
    ])


def get_pos_joint_delta(t: int, 
                        m: mujoco.MjModel, 
                        d: mujoco.MjData, 
                        xpos_target: np.ndarray,
                        pos_errs: np.ndarray) -> np.ndarray:
    
    xpos_err = get_pos_err(t, m, d, xpos_target, pos_errs)    

    # Get arm joints and their velocity addresses
    ur3e_joint_indices = np.arange(6) # num_ur3e_joints

    # Compute full Jacobian and extract columns for arm joints
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, None, get_site_id(m, "tcp"))
    jacp_arm = jacp[:, ur3e_joint_indices]  # Extract relevant columns (3x6)

    jacp_pinv = np.linalg.pinv(jacp_arm)

    # Compute joint angle updates
    theta_delta = jacp_pinv @ xpos_err
    return theta_delta
    

def get_rot_joint_delta(t: int, 
                        m: mujoco.MjModel, 
                        d: mujoco.MjData, 
                        xrot_target: np.ndarray,
                        rot_errs: np.ndarray) -> np.ndarray:
    
    xrot_err = get_rot_err(t, m, d, xrot_target, rot_errs)

    # Get arm joints and their velocity addresses
    ur3e_joint_indices = np.arange(6) # num_ur3e_joints

    # Compute full Jacobian and extract columns for arm joints
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, None, jacr, get_site_id(m, "tcp"))
    jacr_arm = jacr[:, ur3e_joint_indices]  # Extract relevant columns (3x6)

    jacr_pinv = np.linalg.pinv(jacr_arm)

    # Compute joint angle updates

    theta_delta = jacr_pinv @ xrot_err
    return theta_delta
    








def main():
    
    model_path = "assets/ur3e_2f85.xml"
    trajectory_fpath = "controller/data/traj_l.csv"
    config_path = "controller/config/config_l.yml"
    log_fpath = "controller/logs/logs_l/"
    ctrl_mode = "l"

    with open(config_path, "r") as f: yml = yaml.safe_load(f)
    pos_gains = { k:np.diag(v) for k, v in yml["pos"].items() } 
    rot_gains = { k:np.diag(v) for k, v in yml["rot"].items() } 
    hold = yml["hold"]

    # ur3e  = 6  nq, 6  nv, 6 nu
    # 2f85  = 8  nq, 8  nv, 1 nu
    # total = 14 nq, 14 nv, 7 nu
    m, d = load_model(model_path)
    reset(m, d)

    build_traj_l(get_task_space_state(m, d), hold, trajectory_fpath)
    traj_target = load_trajectory(trajectory_fpath)
    T = traj_target.shape[0]
    traj_true = np.zeros_like(traj_target)

    pos_errs = np.zeros(shape=(T, 3))
    rot_errs = np.zeros(shape=(T, 3))
    grip_errs = np.zeros(shape=(T, 1)) # NEEDED ???
    ctrls = np.zeros(shape=(T, m.nu))
    actuator_frc = np.zeros(shape=(T, m.nu))
    
    viewer = mujoco.viewer.launch_passive(m, d)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            
    save_flag = True

    try:
        for t in range(T):
            viewer.sync()
    
            u = ctrl(t, m, d, traj_target[t, :], pos_gains, rot_gains, pos_errs, rot_errs)
            d.ctrl = u
            ctrls[t] = u
            
            mujoco.mj_step(m, d)
            
            traj_true[t] = get_task_space_state(m, d)
            actuator_frc[t] = get_joint_torques(d)
            
            # print(f"pos_target: {traj_target[t, :3]}, pos_true: {traj_true[t, :3]}, pos_err: {pos_errs[t, :]}")
            # print(f"rot_target: {traj_target[t, 3:6]}, rot_true: {traj_true[t, 3:6]}, rot_err: {rot_errs[t, :]}")
            # print(f"grip_target: {traj_target[t, -1]}")
            # print("------------------------------------------------------------------------------------------")

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