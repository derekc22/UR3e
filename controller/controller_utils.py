import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from typing import Callable
from utils import (
    get_site_xpos, get_site_xquat, get_site_xrot,
    get_grasp_force)


################################################################################################################################################################################################################################################
# ure3_2f85.xml
################################################################################################################################################################################################################################################







# def pd_ctrl(t, m, d, 
#             target_traj, 
#             err_func,
#             gains, 
#             tot_joint_errs,
#             ):
    
#     theta_delta = err_func(t, m, d, target_traj)
#     dt = m.opt.timestep
    
#     num_ur3e_joints = 6
#     u = np.zeros((num_ur3e_joints, ))
    
#     kp, kd, ki = gains.values()
    
#     for i in range(num_ur3e_joints):  # 6 arm joints

#         # Get current state
#         curr_joint_angle = d.qpos[i]
#         curr_joint_vel = d.qvel[i]
        
#         # Compute target angle with clamping
#         curr_joint_target_angle = curr_joint_angle + theta_delta[i]
#         joint_range = m.jnt_range[i]

#         if joint_range[0] < joint_range[1]:  # Check valid limits
#             curr_joint_target_angle = np.clip(
#                 curr_joint_target_angle, 
#                 joint_range[0], joint_range[1]
#             )
        
#         # PD torque calculation
#         err = curr_joint_target_angle - curr_joint_angle
#         update_joint_errs(i, tot_joint_errs, err)
        
#         ui = kp[i] * err + kd[i] * -curr_joint_vel # pd
#         # ui = kp[i] * err + kd[i] * -curr_joint_vel + ki[i]*tot_joint_errs[i]*dt # pid
#         u[i] = ui

#     return u


def pd_ctrl(t: int, 
            m: mujoco.MjModel, 
            d: mujoco.MjData, 
            traj_target: np.array, 
            gains: dict, 
            err_func: Callable,
            errs: np.array,
            jnt_ranges: np.array) -> np.array:
    
    theta_delta = err_func(t, m, d, traj_target, errs)
    
    num_ur3e_joints = 6
    u = np.zeros((num_ur3e_joints, ))
    
    kp, kd = gains.values()

    # Get current state
    joint_angles = d.qpos[:6]
    joint_vels = d.qvel[:6]
        
    # Compute target angle with clamping
    joint_target_angles = joint_angles + theta_delta
    
    joint_target_angles = np.clip(
        joint_target_angles, 
        jnt_ranges[:, 0],  # Lower bounds of joint_range
        jnt_ranges[:, 1]   # Upper bounds of joint_range
    )
    
    # PD torque calculation
    errs = joint_target_angles - joint_angles
    
    u = kp @ errs + kd @ -joint_vels # pd

    return u



def update_errs(t: int, 
                errs: np.array, 
                err: np.array) -> None:
    errs[t] = err

def update_tot_errs(tot_errs: np.array,
                    err: np.array) -> None:
    tot_errs[:] += err








def get_gravity_compensation(m: mujoco.MjModel,
                             d: mujoco.MjData) -> np.array:
    """Returns gravity compensation torque for arm joints"""
    return d.qfrc_bias[:6]


def grip_ctrl(m: mujoco.MjData, 
              traj_t: np.array) -> np.array:
    ctrl_range = m.actuator_ctrlrange[-1] # 'fingers_actuator' is the last actuator
    return traj_t * ctrl_range[1] 




def get_task_space_state(m: mujoco.MjModel, 
                         d: mujoco.MjData) -> np.array:
    
    _, xpos_2f85 = get_site_xpos(m, d, "right_pad1_site")
    _, xrot_2f85 = get_site_xrot(m, d, "right_pad1_site")
    xrot_2f85 = R.from_matrix(xrot_2f85.reshape(3, 3)).as_rotvec()
    grip_2f85 = get_grasp_force(d)[0:1] #get_grip_ctrl(d)

    return np.hstack([
        xpos_2f85, xrot_2f85, grip_2f85
    ])

def get_joint_space_state(m: mujoco.MjModel,
                          d: mujoco.MjData) -> np.array:
    
    qpos_ur3e = get_arm_qpos(d)
    grip_2f85 = get_grasp_force(d)[0:1] #get_grip_ctrl(d)

    return np.hstack([
        qpos_ur3e, grip_2f85
    ])