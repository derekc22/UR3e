import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from typing import Callable
from utils.utils import *


def get_mug_toppled(m: mujoco.MjModel, 
                    d: mujoco.MjData) -> np.ndarray:
    init_qpos, _ = get_init(m, mode="deterministic", keyframe="home")
    init_mug_z = init_qpos[-5]
    dx, dy, _ = get_body_size(m, "fish")
    # print(get_mug_xpos(m, d)[-1], init_mug_z, max(dx, dy)/2)
    # 0.045-0.01989224457978381
    return init_mug_z - get_mug_xpos(m, d)[-1] > max(dx, dy)/2


def get_mug_qpos(d: mujoco.MjData) -> np.ndarray:
    return d.qpos[14:21]


def get_ghost_xpos(m: mujoco.MjModel, 
                   d: mujoco.MjData) -> np.ndarray:
    return get_body_xpos(m, d, "ghost")


# def get_mug_xpos(m: mujoco.MjModel, 
#                  d: mujoco.MjData) -> np.ndarray:
#     return get_body_xpos(m, d, "fish")

def get_mug_xpos(m: mujoco.MjModel, 
                 d: mujoco.MjData) -> np.ndarray:
    return get_site_xpos(m, d, "handle_site")


def get_init(m: mujoco.MjModel, 
             mode: str,
             keyframe: str) -> tuple[np.ndarray, np.ndarray]:
    init_qpos = m.keyframe(keyframe).qpos
    init_qvel = m.keyframe(keyframe).qvel
    
    if mode == "stochastic":
        noise = np.hstack([
            np.zeros(m.nq-7), # [0:13]
            np.random.uniform(low=-0.02, high=0.02, size=1), # x [13:14]
            np.random.uniform(low=-0.4, high=0.3, size=1), # y [13:14]
            # np.random.uniform(low=-0.02, high=0.02, size=1), # [13:14]
            np.zeros(5), # [14:21]
        ])    
        return (init_qpos + noise, init_qvel)
    
    return (init_qpos, init_qvel)


def reset_with_mug(m: mujoco.MjModel, 
                   d: mujoco.MjData, 
                   mode: str,
                   keyframe: str) -> None:

    init_qpos, init_qvel = get_init(m, mode=mode, keyframe=keyframe)    
    mujoco.mj_resetData(m, d) 
    d.qpos = init_qpos
    d.qvel = init_qvel
    mujoco.mj_step(m, d)






# def init_collision_cache(m: mujoco.MjModel) -> tuple:
#     # ---------- names you may need to update when renaming ------------
#     gripper_names = {
#         # "left_driver", "right_driver",
#         # "left_spring_link", "right_spring_link",
#         # "left_follower", "right_follower",
#         "left_pad", "right_pad",
#         # "left_pad1", "right_pad1",        # (these are not bodies dumbass)
#         # "left_pad2", "right_pad2",        # (these are not bodies dumbass)   
#         # "left_silicone_pad", "right_silicone_pad",
#     }
#     arm_names = {
#         "robot_base", #"shoulder_link", #"upper_arm_link",
#         "forearm_link", "wrist_1_link", "wrist_2_link",
#         "wrist_3_link", "gripper_base",
#         *gripper_names,
#     }
#     # ---------- convert body names to integer body-ids -----------------
#     gripper_bodies = { get_body_id(m, gripper) for gripper in gripper_names }
#     arm_bodies = { get_body_id(m, arm) for arm in arm_names }
#     table_body = get_body_id(m, "table")
    
#     return (gripper_bodies, arm_bodies, table_body)


# def get_block_grasp_state(m: mujoco.MjModel,
#                     d: mujoco.MjData) -> int:
    
#     one_finger_grip = False
    
#     for k in range(d.ncon):
#         c = d.contact[k]
#         b1 = m.geom_bodyid[c.geom1]
#         b2 = m.geom_bodyid[c.geom2]
        
#         if ((b1 in ("left_pad1_contact", "right_pad1_contact") and b2 == "fish" or 
#             b2 in ("left_pad1_contact", "right_pad1_contact") and b1 == "fish") and
#             not one_finger_grip):
#             one_finger_grip = True
#         elif ((b1 in ("left_pad1_contact", "right_pad1_contact") and b2 == "fish" or 
#               b2 in ("left_pad1_contact", "right_pad1_contact") and b1 == "fish") and
#               one_finger_grip):
#             return 1
#     return 0


def get_block_grasp_state(m: mujoco.MjModel, 
                          d: mujoco.MjData) -> int:
    
    contact_pads = set()
    left_pad_id = get_body_id(m, "left_pad")
    right_pad_id = get_body_id(m, "right_pad")
    fish_id = get_body_id(m, "fish")
    
    for k in range(d.ncon):
        c = d.contact[k]
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]
        
        for pad in (left_pad_id, right_pad_id):
            # print(get_body_name(m, b1))
            # print(get_body_name(m, b2))
            if (pad in (b1, b2)) and (fish_id in (b1, b2)):
                contact_pads.add(pad)
                
    # return int(len(contact_pads) == 2)
    return len(contact_pads)

    


def init_collision_cache(m: mujoco.MjModel) -> tuple[set, set, int]:

    gripper_root_id = get_body_id(m, "robotiq_base_mount")
    gripper_bodies = { gripper_root_id, *get_children_deep(m, gripper_root_id) }
    
    arm_root_id = get_body_id(m, "robot_base")
    arm_bodies = { arm_root_id, *get_children_deep(m, arm_root_id) }
    
    table_body = get_body_id(m, "table")
    
    return (gripper_bodies, arm_bodies, table_body)


def get_self_collision(m: mujoco.MjModel,
                       d: mujoco.MjData,
                       collision_cache: tuple[set, set, int]) -> int:
    """
    Return 1 if any arm or gripper body (excluding gripper-to-gripper pairs)
    touches the table or another arm/gripper body.  Otherwise return 0.
    gripper-to-gripper contacts are ignored so that other routines can still
    inspect them.
    """

    gripper_bodies, arm_bodies, _ = collision_cache

    # ------------------------------------------------------------
    # Inspect active contacts
    # ------------------------------------------------------------
    for k in range(d.ncon):
        c = d.contact[k]
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]

        # detect arm–arm contact
        if (b1 in arm_bodies and b2 in arm_bodies):
            # ignore pure gripper–gripper contact
            if b1 in gripper_bodies and b2 in gripper_bodies:
                continue
            return 1
    return 0

def get_table_collision(m: mujoco.MjModel,
                        d: mujoco.MjData,
                        collision_cache: tuple[set, set, int]) -> int:
    """
    Return 1 if any arm or gripper body (excluding gripper-to-gripper pairs)
    touches the table or another arm/gripper body.  Otherwise return 0.
    gripper-to-gripper contacts are ignored so that other routines can still
    inspect them.
    """

    gripper_bodies, _, table_body = collision_cache

    # ------------------------------------------------------------
    # Inspect active contacts
    # ------------------------------------------------------------
    for k in range(d.ncon):
        c = d.contact[k]
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]

        # ignore pure gripper–gripper contact
        # if b1 in gripper_bodies and b2 in gripper_bodies:
            # continue

        # detect gripper–table contact
        if (b1 in gripper_bodies and b2 == table_body) or (b2 in gripper_bodies and b1 == table_body):
            return 1
    return 0

#@working
# def get_robot_collision(m: mujoco.MjModel,
#                         d: mujoco.MjData,
#                         collision_cache: tuple[set, set, int]) -> int:
#     """
#     Return 1 if any arm or gripper body (excluding gripper-to-gripper pairs)
#     touches the table or another arm/gripper body.  Otherwise return 0.
#     gripper-to-gripper contacts are ignored so that other routines can still
#     inspect them.
#     """

#     gripper_bodies, arm_bodies, table_body = collision_cache

#     # ------------------------------------------------------------
#     # Inspect active contacts
#     # ------------------------------------------------------------
#     for k in range(d.ncon):
#         c = d.contact[k]
#         b1 = m.geom_bodyid[c.geom1]
#         b2 = m.geom_bodyid[c.geom2]

#         # ignore pure gripper–gripper contact
#         if b1 in gripper_bodies and b2 in gripper_bodies:
#             continue

#         # detect arm–arm or arm–table contact
#         if (b1 in arm_bodies and b2 in arm_bodies):
#         # if ((b1 in arm_bodies and b2 in arm_bodies) or
#         #     (b1 in arm_bodies and b2 == table_body) or
#         #     (b2 in arm_bodies and b1 == table_body)):
#             print("collision detected: ", get_body_name(m, b1), get_body_name(m, b2))
#             return 1
#             # exit()
#     return 0



# collision_cache = {}
# def arm_collision(model: mujoco.MjModel,
#                   data: mujoco.MjData) -> int:
#     """
#     Return 1 if any arm or gripper body (excluding gripper-to-gripper pairs)
#     touches the table or another arm/gripper body.  Otherwise return 0.
#     gripper-to-gripper contacts are ignored so that other routines can still
#     inspect them.
#     """
#     # ------------------------------------------------------------
#     # Build and memoise lookup tables the first time we see a model
#     # ------------------------------------------------------------
#     cache_entry = collision_cache.get(id(model))
#     if cache_entry is None:
#         # ---------- names you may need to update when renaming ------------
#         gripper_names = {
#             "left_driver", "right_driver",
#             "left_spring_link", "right_spring_link",
#             "left_follower", "right_follower",
#             "left_pad", "right_pad",
#             "left_pad1", "right_pad1",        # include sub-pads if present
#             "left_pad2", "right_pad2",
#             "left_gripper_pad", "right_gripper_pad",
#         }
#         arm_names = {
#             "robot_base", "shoulder_link", "upper_arm_link",
#             "forearm_link", "wrist_1_link", "wrist_2_link",
#             "wrist_3_link", "gripper_base",
#             *gripper_names,
#         }
#         # ---------- convert body names to integer body-ids -----------------
#         gripper_bodies = {
#             mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
#             for n in gripper_names
#             # if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) != -1
#         }
#         arm_bodies = {
#             mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
#             for n in arm_names
#             if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) != -1
#         }
#         table_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
#         if table_body == -1:
#             raise ValueError("Body named 'table' not found in model.")
#         cache_entry = (gripper_bodies, arm_bodies, table_body)
#         collision_cache[id(model)] = cache_entry

#     gripper_bodies, arm_bodies, table_body = cache_entry

#     # ------------------------------------------------------------
#     # Inspect active contacts
#     # ------------------------------------------------------------
#     for k in range(data.ncon):
#         c = data.contact[k]
#         b1 = model.geom_bodyid[c.geom1]
#         b2 = model.geom_bodyid[c.geom2]

#         # ignore pure gripper–gripper contact
#         if b1 in gripper_bodies and b2 in gripper_bodies:
#             continue

#         # detect arm–arm or arm–table contact
#         if ((b1 in arm_bodies and b2 in arm_bodies) or
#             (b1 in arm_bodies and b2 == table_body) or
#             (b2 in arm_bodies and b1 == table_body)):
#             return 1
#     return 0















"""def get_2f85_home2(m, d):

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
        pos = get_body_xpos(m, d, body)
        quat = get_body_xquat(m, d, body)
        transforms.append( ( tuple(pos), tuple(quat) ) )
        
    # Local position of the site within the right_pad body
    site_local_pos = get_site_xpos(m, d, "right_pad1_site")

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
"""