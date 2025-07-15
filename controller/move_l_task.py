import mujoco
import numpy as np
import matplotlib
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf)
matplotlib.use('Agg')  # Set backend to non-interactive
from controller.controller_utils import (
    get_task_space_state,
    grip_ctrl, update_errs, update_tot_errs)
from utils import (
    load_model, reset,
    get_site_xpos, get_site_xquat, get_site_xrot, get_joint_torques
)
from scipy.spatial.transform import Rotation as R
from gen_traj import gen_traj_l
from controller.aux import build_trajectory, build_interpolated_trajectory, cleanup
import yaml
import time

    





def ctrl(t: int, 
         m: mujoco.MjModel, 
         d: mujoco.MjData, 
         traj_target: np.array, 
         pos_gains: dict, 
         rot_gains: dict,
         pos_errs: np.array,
         rot_errs: np.array,
         tot_pos_errs: np.array,
         tot_rot_errs: np.array) -> np.array:

    sensor_site_2f85, xpos_2f85 = get_site_xpos(m, d, "right_pad1_site")
    _, xrot_2f85 = get_site_xrot(m, d, "right_pad1_site") 
    xrot_2f85 = xrot_2f85.reshape(3, 3)
        
    # Position error
    xpos_delta = traj_target[:3] - xpos_2f85
    update_errs(t, pos_errs, xpos_delta)
    update_tot_errs(tot_pos_errs, xpos_delta)
    
    # Orientation error (axis-angle)
    xrot_target = traj_target[3:6]

    # Quaternion approach
    # Convert current and target rotation matrices to quaternions
    q = R.from_matrix(xrot_2f85).as_quat()   # [x, y, z, w]
    q_d = R.from_rotvec(xrot_target).as_quat()
    # q_d = R.from_matrix(xrot_target).as_quat()
    # Compute the inverse of the current quaternion
    q_inv = R.from_quat(q).inv()
    # Compute the relative quaternion: q_err = q_d * q⁻¹
    q_err = R.from_quat(q_d) * q_inv
    # Convert to rotation vector (axis-angle)
    xrot_delta = q_err.as_rotvec()
    update_errs(t, rot_errs, xrot_delta)
    update_tot_errs(tot_rot_errs, xrot_delta)

    # Compute full geometric Jacobian (6x6 for arm joints)
    jac = np.zeros((6, m.nv))
    mujoco.mj_jacSite(m, d, jac[:3], jac[3:], sensor_site_2f85)
    jac_arm = jac[:, :6]
    
    # Task-space PD force
    kp_pos, kd_pos, ki_pos = pos_gains.values()
    kp_rot, kd_rot, ki_rot = rot_gains.values()
    dt = m.opt.timestep
    
    # in task space (?)
    u_pos = kp_pos @ xpos_delta - kd_pos @ (jac_arm[:3] @ d.qvel[:6]) + ki_pos @ tot_pos_errs * dt
    u_rot = kp_rot @ xrot_delta - kd_rot @ (jac_arm[3:] @ d.qvel[:6]) + ki_rot @ tot_rot_errs * dt
    u = np.hstack([u_pos, u_rot])
    
    # Compute joint torques (gravity compensation + task force)
    # Convert from task space to joint space using inverse jacobian (?)
    pos_rot_u = jac_arm.T @ u + d.qfrc_bias[:6]
    grip_u = grip_ctrl(m, traj_target[-1])
    
    return np.hstack([
        pos_rot_u, 
        grip_u
    ])







def main():
    
    model_path = "assets/ur3e_2f85.xml"
    trajectory_fpath = "controller/data/traj_l.csv"
    config_path = "controller/config/config_l_task.yml"
    log_fpath = "controller/logs/logs_l_task/"
    ctrl_mode = "l_task"
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

    gen_traj_l()
    traj_target = build_interpolated_trajectory(n, hold, trajectory_fpath) if n else build_trajectory(hold, trajectory_fpath)
    # print(traj_target.shape)
    T = traj_target.shape[0]
    traj_true = np.zeros_like(traj_target)

    jnt_ranges = m.jnt_range[:num_ur3e_joints] # NEEDED IN THIS CASE?
    pos_errs = np.zeros(shape=(T, 3))
    rot_errs = np.zeros(shape=(T, 3))
    grip_errs = np.zeros(shape=(T, 1)) # NEEDED?
    ctrls = np.zeros(shape=(T, m.nu))
    actuator_frc = np.zeros(shape=(T, m.nu))

    tot_pos_errs = np.zeros(shape=(3, ))
    tot_rot_errs = np.zeros(shape=(3, ))

    viewer = mujoco.viewer.launch_passive(m, d)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            
    reset(m, d)
    save_flag = True
    
    try:
        for t in range (T):
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
            
            # time.sleep(0.001)

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
    



# Skew-symmetric approach
# R_err = xrot_target @ xrot_2f85.T
# skew = 0.5 * (R_err - R_err.T)
# xrot_delta = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

# Cross product approach
# R_err = xrot_target @ xrot_2f85.T
# xrot_delta = 0.5 * (np.cross(xrot_2f85[:, 0], xrot_target[:, 0]) +
            #  np.cross(xrot_2f85[:, 1], xrot_target[:, 1]) +
            #  np.cross(xrot_2f85[:, 2], xrot_target[:, 2]))

# Logarithmic map approach
# from scipy.linalg import logm
# R_err = xrot_target @ xrot_2f85.T
# rot_skew = logm(R_err)
# xrot_delta = np.array([rot_skew[2, 1], rot_skew[0, 2], rot_skew[1, 0]])