import mujoco
import numpy as np
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf)
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
from utils import (
    load_model, reset,
    get_xpos, get_xrot, get_task_space_state,
    pd_ctrl, grip_ctrl, update_errs, get_joint_torques)
from gen_traj import gen_trajL
from controller.aux import build_trajectory, build_interpolated_trajectory, cleanup
import yaml
from scipy.spatial.transform import Rotation as R

    




def ctrl(t: int,
         m: mujoco.MjModel, 
         d: mujoco.MjData,
         traj_t: np.array,
         pos_gains: dict,
         rot_gains: dict,
         pos_errs: np.array,
         rot_errs: np.array,
         jnt_ranges: np.array,) -> np.array:
    
    pos_u = pd_ctrl(t, m, d, traj_t[:3], pos_gains, get_pos_err, pos_errs, jnt_ranges)    
    rot_u = pd_ctrl(t, m, d, traj_t[3:6], rot_gains, get_rot_err, rot_errs, jnt_ranges)  # rotation matrix

    grip_u = grip_ctrl(m, traj_t[-1])
    
    return np.hstack([
        pos_u + rot_u,
        # rot_u, 
        grip_u
    ])




def get_pos_err(t: int, 
                m: mujoco.MjModel, 
                d: mujoco.MjData, 
                xpos_target: np.array,
                pos_errs: np.array) -> np.array:
    
    sensor_site_2f85, xpos_2f85 = get_xpos(m, d, "right_pad1_site")
    
    # Compute 3D cartesian position error
    xpos_delta = xpos_target - xpos_2f85
    update_errs(t, pos_errs, xpos_delta)

    # Get arm joints and their velocity addresses
    ur3e_joint_indices = np.arange(6) # num_ur3e_joints

    # Compute full Jacobian and extract columns for arm joints
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, None, sensor_site_2f85)
    jacp_arm = jacp[:, ur3e_joint_indices]  # Extract relevant columns (3x6)

    jacp_pinv = np.linalg.pinv(jacp_arm)

    # Compute joint angle updates
    theta_delta = jacp_pinv @ xpos_delta
    return theta_delta # theta_delta
    
    
    

def get_rot_err(t: int, 
                m: mujoco.MjModel, 
                d: mujoco.MjData, 
                xrot_target: np.array,
                rot_errs: np.array) -> np.array:
    
    sensor_site_2f85, xrot_2f85 = get_xrot(m, d, "right_pad1_site") 
    xrot_2f85 = xrot_2f85.reshape(3, 3) # Rgc
     
    #########################################################################################################################
    # Compute rotational error (Quaternion approach)
    
    # Convert current and target rotation matrices to quaternions
    q = R.from_matrix(xrot_2f85).as_quat()   # [x, y, z, w]
    q_d = R.from_rotvec(xrot_target).as_quat()
    # Compute the inverse of the current quaternion
    q_inv = R.from_quat(q).inv()
    # Compute the relative quaternion: q_err = q_d * q⁻¹
    q_err = R.from_quat(q_d) * q_inv
    # Convert to rotation vector (axis-angle)
    xrot_delta = q_err.as_rotvec()
    update_errs(t, rot_errs, xrot_delta)

    # Get arm joints and their velocity addresses
    ur3e_joint_indices = np.arange(6) # num_ur3e_joints

    # Compute full Jacobian and extract columns for arm joints
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, None, jacr, sensor_site_2f85)
    jacr_arm = jacr[:, ur3e_joint_indices]  # Extract relevant columns (3x6)

    jacr_pinv = np.linalg.pinv(jacr_arm)

    # Compute joint angle updates

    theta_delta = jacr_pinv @ xrot_delta
    return theta_delta
    








def main():
    
    model_path = "assets/ur3e_2f85.xml"
    trajectory_fpath = "controller/data/traj_l.csv"
    config_path = "controller/config/config_l.yml"
    log_fpath = "controller/logs/logs_l/"
    ctrl_mode = "L"
    num_ur3e_joints = 6

    with open(config_path, "r") as f: yml = yaml.safe_load(f)
    pos_gains = { k:np.diag(v) for k, v in yml["pos"].items() } 
    rot_gains = { k:np.diag(v) for k, v in yml["rot"].items() } 
    hold = yml["hold"]
    n = yml["n"]

    # ur3e  = 6  nq, 6  nv, 6 nu
    # 2f85  = 8  nq, 8  nv, 1 nu
    # total = 14 nq, 14 nv, 7 nu
    m, d = load_model(model_path)

    gen_trajL()
    traj_target = build_interpolated_trajectory(n, hold, trajectory_fpath) if n else build_trajectory(hold, trajectory_fpath)
    T = traj_target.shape[0]
    traj_true = np.zeros_like(traj_target)

    jnt_ranges = m.jnt_range[:num_ur3e_joints]
    pos_errs = np.zeros(shape=(T, 3))
    rot_errs = np.zeros(shape=(T, 3))
    grip_errs = np.zeros(shape=(T, 1)) # NEEDED ???
    ctrls = np.zeros(shape=(T, m.nu))
    actuator_frc = np.zeros(shape=(T, m.nu))
    
    viewer = mujoco.viewer.launch_passive(m, d)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            
    reset(m, d)
    save_flag = True

    try:
        for t in range (T):
            viewer.sync()
    
            u = ctrl(t, m, d, traj_target[t, :], pos_gains, rot_gains, pos_errs, rot_errs, jnt_ranges)
            d.ctrl = u
            ctrls[t] = u
            
            mujoco.mj_step(m, d)
            
            traj_true[t] = get_task_space_state(m, d)
            actuator_frc[t] = get_joint_torques(d)
            
            # print(f"pos_target: {traj_target[t, :3]}, pos_true: {traj_true[t, :3]}, pos_err: {pos_errs[t, :]}")
            print(f"rot_target: {traj_target[t, 3:6]}, rot_true: {traj_true[t, 3:6]}, rot_err: {rot_errs[t, :]}")
            # print(f"grip_target: {traj_target[t, -1]}")
            print("------------------------------------------------------------------------------------------")
            
            # time.sleep(0.01)

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
    

    # Compute rotational error (o3)
    
    # R_err = xrot_target @ xrot_2f85.T                 # desired-to-current rotation
    # skew = 0.5 * (R_err - R_err.T)
    # xrot_delta =  np.array([skew[2, 1], skew[0, 2], skew[1, 0]])