from typing import Tuple, List
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

def get_joint_torques(d):
    joints = [
        "shoulder_pan_actuatorfrc",
        "shoulder_lift_actuatorfrc",
        "elbow_actuatorfrc",
        "wrist_1_actuatorfrc",
        "wrist_2_actuatorfrc",
        "wrist_3_actuatorfrc",
        "fingers_actuatorfrc"
    ]
    # return { joint : float(d.sensor(joint).data[0]) for joint in joints }
    return np.array([ d.sensor(joint).data[0] for joint in joints ])


def get_grasp_force(d):
    contact_pads = [
        "right_pad1_contact",
        # "right_pad2_contact",
        "left_pad1_contact",
        # "left_pad2_contact",
    ]
    # return { pad : float(d.sensor(pad).data[0]) for pad in contact_pads }
    return np.array([ d.sensor(pad).data[0] for pad in contact_pads ])


def get_ghost_pos(m):
    return m.body(name='ghost').pos

def get_mug_qpos(d):
    return d.qpos[:7]

def get_robot_qpos(d):
    return d.qpos[7:]

def get_robot_qvel(d):
    return d.qvel[6:]

def get_gripper_pos(m, d):
    return d.xpos[m.body("left_pad").id]
    


# def euler_to_quaternion(roll, pitch, yaw):
#     """Convert Euler angles (radians) to a unit quaternion."""
#     cy = np.cos(yaw * 0.5)
#     sy = np.sin(yaw * 0.5)
#     cp = np.cos(pitch * 0.5)
#     sp = np.sin(pitch * 0.5)
#     cr = np.cos(roll * 0.5)
#     sr = np.sin(roll * 0.5)

#     qw = cr * cp * cy + sr * sp * sy
#     qx = sr * cp * cy - cr * sp * sy
#     qy = cr * sp * cy + sr * cp * sy
#     qz = cr * cp * sy - sr * sp * cy

#     return [qw, qx, qy, qz]

def euler_to_quaternion(euler_angles):
    """Convert Euler angles (radians) to a unit quaternion."""
    
    roll, pitch, yaw = euler_angles
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])


def euler_to_R(euler_angles):
    
    """Convert Euler angles (radians) to ZYX (yaw-pitch-roll) rotation matrix."""
    # Note, angles are passed as roll, pitch, yaw, but R is constructued as R = RZ @ RY @ RX (roll first, pitch second, yaw last)
    
    roll, pitch, yaw = euler_angles
    
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    
    return np.array([
      [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
      [-sp, cp*sr, cp*cr]  
    ]).flatten()
    
    
def R_to_euler(R):

    if abs(R[2, 0]) != 1:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        pitch = np.pi / 2 if R[2, 0] == -1 else -np.pi / 2
        roll = 0
        yaw = np.arctan2(R[0, 1], R[0, 2]) if R[2, 0] == -1 else np.arctan2(-R[0, 1], -R[0, 2])
    return np.array([roll, pitch, yaw])  # XYZ order





def get_xpos(m, d, site):
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    return site_id, d.site(site_id).xpos

# def get_xquat(m, d, site):
#     site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
#     # xmat = d.site(site_id).xmat.copy()
#     xmat = np.array(d.site(site_id).xmat).reshape(3, 3)
#     return site_id, R.from_matrix(xmat).as_quat()

def get_xrot(m, d, site):
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    return site_id, d.site(site_id).xmat


def get_grip_ctrl(d):    
    # print(d.sensor("fingers_actuatorfrc").data)
    return d.sensor("fingers_actuatorfrc").data # actuatorfrc (0, 255) ???