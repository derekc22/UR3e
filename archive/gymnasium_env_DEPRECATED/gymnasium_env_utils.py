import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from typing import Callable
from utils.utils import *



################################################################################################################################################################################################################################################
# main.xml
################################################################################################################################################################################################################################################




###################### SHORTHANDS ############################

def get_ur3e_qpos(d: mujoco.MjData) -> np.array:
    return d.qpos[:6]


def get_ur3e_qvel(d: mujoco.MjData) -> np.array:
    return d.qvel[:6]


def get_robot_qpos(d: mujoco.MjData) -> np.array:
    return d.qpos[:14]


def get_robot_qvel(d: mujoco.MjData) -> np.array:
    return d.qvel[:14]


def get_mug_qpos(d: mujoco.MjData) -> np.array:
    return d.qpos[14:21]


def get_2f85_xpos(m: mujoco.MjModel,
                  d: mujoco.MjData) -> np.array:
    xpos_2f85 = get_site_xpos(m, d, "right_pad1_site")
    return xpos_2f85


def get_ghost_xpos(m: mujoco.MjModel, 
                   d: mujoco.MjData) -> np.array:
    ghost_xpos = get_body_xpos(m, d, "ghost")
    return ghost_xpos


def get_mug_xpos(m: mujoco.MjModel, 
                 d: mujoco.MjData) -> np.array:
    mug_xpos = get_body_xpos(m, d, "fish")
    return mug_xpos


















def get_2f85_home2(m, d):

    # Helper function to apply a transformation (rotation + translation)
    def transform_point(pos, quat, point):
        rot = R.from_quat(quat)
        return rot.apply(point) + pos

    # Transformation chain from base to right_pad1_site
    transforms = [
        # Each tuple is (position, quaternion) in world coordinates
        ((0, 0, 0.152), (0, 0, 0, 1)),  # shoulder_link
        ((0, 0.12, 0), (0, 0.707107, 0, 0.707107)),  # upper_arm_link
        ((0, -0.093, 0.244), (0, 0, 0, 1)),  # forearm_link
        ((0, 0, 0.213), (0, 0.707107, 0, 0.707107)),  # wrist_1_link
        ((0, 0.104, 0), (0, 0, 0, 1)),  # wrist_2_link
        ((0, 0, 0.085), (0, 0, 0, 1)),  # wrist_3_link
        ((0, 0.082, 0), (0, 0, -0.707107, 0.707107)),  # robotiq_base_mount
        ((0, 0, 0.0038), (0, 0, -1, 1)),  # gripper_base
        ((0, 0.0306011, 0.054904), (0, 0, 0, 1)),  # right_driver
        ((0, 0.0315, -0.0041), (0, 0, 0, 1)),  # right_coupler
        ((0, 0.055, 0.0375), (0, 0, 0, 1)),  # right_follower
        ((0, -0.0189, 0.01352), (0, 0, 0, 1)),  # right_pad
    ]

    # Local position of the site within the right_pad body
    site_local_pos = np.array([0, -0.0025, 0.0185])

    # Initial global transformation
    global_pos = np.array([0, 0, 0])
    global_quat = R.from_quat([0, 0, 0, 1])  # Identity rotation

    # Apply transformations sequentially
    for pos, quat in transforms:
        global_pos = transform_point(global_pos, global_quat.as_quat(), pos)
        global_quat = global_quat * R.from_quat(quat)

    # Final transformation: apply local site position
    site_global_pos = transform_point(global_pos, global_quat.as_quat(), site_local_pos)

    return site_global_pos





def get_2f85_home(m, d):
    
    # Helper function to apply a transformation (rotation + translation)
    def transform_point(pos, quat, point):
        rot = R.from_quat(quat)
        return rot.apply(point) + pos

    bodies = [
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
        "robotiq_base_mount",
        "gripper_base",
        "right_driver",
        "right_coupler",
        "right_follower",
        "right_pad",
    ]
        
    transforms = []
    for body in bodies:
        # print(body)
        _, pos = get_body_xpos(m, d, body)
        _, quat = get_body_xquat(m, d, body)
        transforms.append( ( tuple(pos), tuple(quat) ) )
        
    # Local position of the site within the right_pad body
    _, site_local_pos = get_site_xpos(m, d, "right_pad1_site")

    # Initial global transformation
    global_pos = np.array([0, 0, 0])
    global_quat = R.from_quat([0, 0, 0, 1])  # Identity rotation

    # Apply transformations sequentially
    for pos, quat in transforms:
        global_pos = transform_point(global_pos, global_quat.as_quat(), pos)
        global_quat = global_quat * R.from_quat(quat)

    # Final transformation: apply local site position
    site_global_pos = transform_point(global_pos, global_quat.as_quat(), site_local_pos)

    return site_global_pos
