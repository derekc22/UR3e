import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from typing import Callable
from utils import *


################################################################################################################################################################################################################################################
# ure3_2f85.xml
################################################################################################################################################################################################################################################


def pd_ctrl(t: int, 
            m: mujoco.MjModel, 
            d: mujoco.MjData, 
            traj_target: np.array, 
            gains: dict, 
            err_func: Callable,
            errs: np.array) -> np.ndarray:
    
    theta_delta = err_func(t, m, d, traj_target, errs)
    
    num_ur3e_joints = 6
    u = np.zeros((num_ur3e_joints, ))
    
    kp, kd = gains.values()

    # Get current state
    joint_angles = d.qpos[:6]
    joint_vels = d.qvel[:6]
        
    # Compute target angle with clamping
    joint_target_angles = joint_angles + theta_delta
    
    jnt_ranges = get_jnt_ranges(m)[:num_ur3e_joints]
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
    errs[t, :] = err

def update_tot_errs(tot_errs: np.array,
                    err: np.array) -> None:
    tot_errs[:] += err








def get_gravity_compensation(d: mujoco.MjData) -> np.ndarray:
    """Returns gravity compensation torque for arm joints"""
    return d.qfrc_bias[:6]


def grip_ctrl(m: mujoco.MjData, 
              traj_t: np.array) -> np.ndarray:
    ctrl_range = m.actuator_ctrlrange[-1] # 'fingers_actuator' is the last actuator
    return traj_t * ctrl_range[1] 




def get_task_space_state(m: mujoco.MjModel, 
                         d: mujoco.MjData) -> np.ndarray:
    
    xpos_2f85 = get_site_xpos(m, d, "right_pad1_site")
    xrot_2f85 = get_site_xrotvec(m, d, "right_pad1_site")
    grip_2f85 = get_binary_grasp_contact(d) #get_grasp_contact(d) #get_finger_torque(d)

    return np.hstack([
        xpos_2f85, xrot_2f85, grip_2f85
    ])



def get_joint_space_state(d: mujoco.MjData) -> np.ndarray:
    
    qpos_ur3e = get_ur3e_qpos(d)
    grip_2f85 = get_binary_grasp_contact(d) #get_grasp_contact(d) #get_finger_torque(d)

    return np.hstack([
        qpos_ur3e, grip_2f85
    ])