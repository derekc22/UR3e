import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from typing import Callable
from utils.utils import *


################################################################################################################################################################################################################################################
# General
################################################################################################################################################################################################################################################


def get_pos_err(t: int, 
                m: mujoco.MjModel, 
                d: mujoco.MjData, 
                xpos_target: np.ndarray,
                pos_errs: np.ndarray) -> np.ndarray:

    xpos_2f85 = get_site_xpos(m, d, "tcp")

    # Position error
    xpos_err = xpos_target - xpos_2f85
    update_errs(t, pos_errs, xpos_err)
    
    # print(f"pos_target: {xpos_target}, pos_true: {xpos_2f85}, pos_err: {xpos_err}")
    
    return xpos_err


def get_rot_err(t: int, 
                m: mujoco.MjModel, 
                d: mujoco.MjData, 
                xrot_target: np.ndarray,
                rot_errs: np.ndarray) -> np.ndarray:
    
    # Quaternion approach
    # Convert current and target rotation matrices to quaternions
    q = get_site_R(m, d, "tcp")
    q_d = R.from_rotvec(xrot_target)
    # Compute the inverse of the current quaternion
    q_inv = q.inv()
    # Compute the relative quaternion: q_err = q_d * q⁻¹
    q_err = q_d * q_inv
    # Convert to rotation vector (axis-angle)
    xrot_err = q_err.as_rotvec()
    update_errs(t, rot_errs, xrot_err)
    
    return xrot_err
    


def update_errs(t: int, 
                errs: np.ndarray, 
                err: np.ndarray) -> None:
    errs[t, :] = err

def update_tot_errs(tot_errs: np.ndarray,
                    err: np.ndarray) -> None:
    tot_errs[:] += err
    


################################################################################################################################################################################################################################################
# Task Space
################################################################################################################################################################################################################################################


def pid_task_ctrl(t: int, 
                  m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  traj_target: np.ndarray, 
                  pos_gains: dict, 
                  rot_gains: dict,pos_errs: 
                  np.ndarray,rot_errs: 
                  np.ndarray,tot_pos_errs: 
                  np.ndarray,tot_rot_errs: 
                  np.ndarray) -> np.ndarray:


    xpos_err = get_pos_err(t, m, d, traj_target[:3], pos_errs)
    xrot_err = get_rot_err(t, m, d, traj_target[3:6], rot_errs)
    
    update_tot_errs(tot_pos_errs, xpos_err)
    update_tot_errs(tot_rot_errs, xrot_err)

    # Compute full geometric Jacobian (6x6 for arm joints)
    jac = np.zeros((6, m.nv))
    mujoco.mj_jacSite(m, d, jac[:3], jac[3:], get_site_id(m, "tcp"))
    jac_arm = jac[:, :6]
    
    # Task-space PD force
    num_ur3e_joints = 6
    kp_pos, kd_pos, ki_pos = pos_gains.values()
    kp_rot, kd_rot, ki_rot = rot_gains.values()
    dt = m.opt.timestep
    
    # in task space (?)
    u_pos = kp_pos @ xpos_err - kd_pos @ (jac_arm[:3] @ d.qvel[:6]) #+ ki_pos @ tot_pos_errs * dt
    u_rot = kp_rot @ xrot_err - kd_rot @ (jac_arm[3:] @ d.qvel[:6]) #+ ki_rot @ tot_rot_errs * dt
    u = np.hstack([u_pos, u_rot])
    
    # Compute joint torques (gravity compensation + task force)
    # Convert from task space to joint space using inverse jacobian (?)
    pos_rot_u = jac_arm.T @ u + d.qfrc_bias[:6]
    grip_u = grip_ctrl(m, traj_target[-1])
    
    ctrl_ranges = get_ctrl_ranges(m)[:num_ur3e_joints]
    u = np.clip(
        u, 
        ctrl_ranges[:, 0],  # Lower bounds of ctrl_range
        ctrl_ranges[:, 1]   # Upper bounds of ctrl_range
    )
    
    return np.hstack([
        pos_rot_u, 
        grip_u
    ])





################################################################################################################################################################################################################################################
# Joint Space
################################################################################################################################################################################################################################################


def pd_joint_ctrl(t: int, 
                  m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  traj_target: np.ndarray, 
                  gains: dict, 
                  joint_delta_func: Callable,
                  errs: np.ndarray) -> np.ndarray:
    
    theta_delta = joint_delta_func(t, m, d, traj_target, errs)
    
    num_ur3e_joints = 6
    kp, kd = gains.values()

    # Get current state
    joint_angles = d.qpos[:6]
    joint_vels = d.qvel[:6]
        
    # Compute target angle with clamping
    joint_target_angles = joint_angles + theta_delta
    
    jnt_ranges = get_jnt_ranges(m)[:num_ur3e_joints]
    joint_target_angles = np.clip(
        joint_target_angles, 
        jnt_ranges[:, 0],  # Lower bounds of jnt_range
        jnt_ranges[:, 1]   # Upper bounds of jnt_range
    )
    
    # PD torque calculation
    errs = joint_target_angles - joint_angles
    
    u = kp @ errs + kd @ -joint_vels # pd

    ctrl_ranges = get_ctrl_ranges(m)[:num_ur3e_joints]
    u = np.clip(
        u, 
        ctrl_ranges[:, 0],  # Lower bounds of ctrl_range
        ctrl_ranges[:, 1]   # Upper bounds of ctrl_range
    )

    return u



################################################################################################################################################################################################################################################
# Other
################################################################################################################################################################################################################################################




def get_gravity_compensation(d: mujoco.MjData) -> np.ndarray:
    """Returns gravity compensation torque for arm joints"""
    return d.qfrc_bias[:6]


def grip_ctrl(m: mujoco.MjData, 
              traj_t: np.ndarray) -> np.ndarray:
    ctrl_range = m.actuator_ctrlrange[-1] # 'fingers_actuator' is the last actuator
    return traj_t * ctrl_range[1] 




def get_task_space_state(m: mujoco.MjModel, 
                         d: mujoco.MjData) -> np.ndarray:
    
    xpos_2f85 = get_site_xpos(m, d, "tcp")
    xrot_2f85 = get_site_xrotvec(m, d, "tcp")
    grip_2f85 = get_boolean_grasp_contact(d) #get_grasp_contact(d) 

    return np.hstack([
        xpos_2f85, xrot_2f85, grip_2f85
    ])



def get_joint_space_state(d: mujoco.MjData) -> np.ndarray:
    
    qpos_ur3e = get_ur3e_qpos(d)
    grip_2f85 = get_boolean_grasp_contact(d) #get_grasp_contact(d) 

    return np.hstack([
        qpos_ur3e, grip_2f85
    ])