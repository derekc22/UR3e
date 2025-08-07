import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from typing import Callable


###################### INITIALIZATION ############################

def load_model(model_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    return m, d


def reset(m: mujoco.MjModel, 
          d: mujoco.MjData, 
          keyframe: str) -> None:
    init_qpos = m.keyframe(keyframe).qpos
    init_qvel = m.keyframe(keyframe).qvel
    mujoco.mj_resetData(m, d) 
    d.qpos = init_qpos
    d.qvel = init_qvel
    # mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)


###################### BODY, SITE, JOINT ID/NAME ############################

def get_body_id(m: mujoco.MjModel,
                body: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)


def get_site_id(m: mujoco.MjModel,
                site: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)


def get_joint_id(m: mujoco.MjModel, 
                 joint: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)


def get_joint_name(m: mujoco.MjModel, 
                   id: str) -> int:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, id)


def get_body_name(m: mujoco.MjModel,
                  id: int) -> str:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, id)


def get_site_name(m: mujoco.MjModel,
                  id: int) -> str:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, id)


def get_joint_name(m: mujoco.MjModel,
                   id: int) -> str:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, id)





###################### BODY XPOS, XQUAT, R, XMAT, XROTVEC ############################

def get_body_xpos(m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  body: str) -> np.ndarray:
    body_id = get_body_id(m, body)
    return d.xpos[body_id]


def get_body_xquat(m: mujoco.MjModel, 
                   d: mujoco.MjData,
                   body: str) -> np.ndarray:
    xmat = get_body_xmat(m, d, body).reshape(3, 3)
    return R.from_matrix(xmat).as_quat()


def get_body_R(m: mujoco.MjModel, 
                     d: mujoco.MjData,
                     body: str) -> R:
    """Return scipy Rotation object"""
    xmat = get_body_xmat(m, d, body).reshape(3, 3)
    return R.from_matrix(xmat)


def get_body_xmat(m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  body: str) -> np.ndarray:
    body_id = get_body_id(m, body)
    return d.xmat[body_id]


def get_body_xrotvec(m: mujoco.MjModel, 
                     d: mujoco.MjData, 
                     body: str) -> np.ndarray:
    xmat = get_body_xmat(m, d, body).reshape(3, 3)
    return R.from_matrix(xmat).as_rotvec()


###################### SITE XPOS, XQUAT, R, XMAT, XROTVEC ############################

def get_site_xpos(m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  site: str) -> np.ndarray:
    site_id = get_site_id(m, site)
    return d.site(site_id).xpos


def get_site_xquat(m: mujoco.MjModel, 
                   d: mujoco.MjData,
                   site: str) -> np.ndarray:
    xmat = get_site_xmat(m, d, site).reshape(3, 3)
    return R.from_matrix(xmat).as_quat()


def get_site_R(m: mujoco.MjModel, 
                     d: mujoco.MjData,
                     site: str) -> R:
    """Return scipy Rotation object"""
    xmat = get_site_xmat(m, d, site).reshape(3, 3)
    return R.from_matrix(xmat)


def get_site_xmat(m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  site: str) -> np.ndarray:
    site_id = get_site_id(m, site)
    return d.site(site_id).xmat


def get_site_xrotvec(m: mujoco.MjModel, 
                     d: mujoco.MjData, 
                     site: str) -> np.ndarray:
    xmat = get_site_xmat(m, d, site).reshape(3, 3)
    return R.from_matrix(xmat).as_rotvec()


###################### SITE XPOS, XQUAT, R, XMAT, XROTVEC ############################

def get_site_velp(m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  site: str) -> np.ndarray:
    return get_site_vel(m, d, site)[3:]


def get_site_velr(m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  site: str) -> np.ndarray:
    return get_site_vel(m, d, site)[:3]


def get_site_vel(m: mujoco.MjModel, 
                 d: mujoco.MjData, 
                 site: str) -> tuple[np.ndarray]:
    
    vel = np.zeros(6)
    site_id = get_site_id(m, site)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, site_id, vel, flg_local=0)
    # fig_local=0 returns the velocity vector in the world frame
    # fig_local=1 return the velocity vector in the site's local frame

    return vel # [ωx, ωy, ωz, vx, vy, vz] 

###################### DATA ############################

def get_body_size(m: mujoco.MjModel, 
                  body: str) -> np.ndarray:
    return m.geom_size[m.geom_bodyid == get_body_id(m, body)][0]


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
    return np.array([ d.sensor(joint).data[0] for joint in joints ])


# @DEPRECATED
# def get_grasp_contact(d):
#     contact_pads = [
#         "right_pad1_contact",
#         # "right_pad2_contact",
#         "left_pad1_contact",
#         # "left_pad2_contact",
#     ]
#     # return { pad : float(d.sensor(pad).data[0]) for pad in contact_pads }
#     return np.array([ d.sensor(pad).data[0] for pad in contact_pads ])


# @DEPRECATED
# def get_boolean_grasp_contact_single(d: mujoco.MjData) -> bool:
#     contact_threshold = 0.1
#     return get_grasp_contact(d) > contact_threshold


# @DEPRECATED
# def get_grasp_contact_single(d: mujoco.MjData) -> float:
#     # boolean [0, oscillates between 6.29 to 12.75 for some reason] = [no contact, contact]
#     return d.sensor("left_pad1_contact").data[0]


def get_boolean_grasp_contact(d: mujoco.MjData) -> bool:
    contact_threshold = 0.1
    return get_grasp_contact(d) > (contact_threshold, contact_threshold)


def get_grasp_contact(d: mujoco.MjData) -> tuple[float, float]:
    # boolean [0, oscillates between 6.29 to 12.75 for some reason] = [no contact, contact]
    return (d.sensor("left_pad1_contact").data[0], d.sensor("right_pad1_contact").data[0])

# @DEPRECATED
# def get_finger_torque(d: mujoco.MjData) -> np.ndarray: 
#     # continuous (-0.28418, 1.3) = (no contact, full contact)
#     return d.sensor("fingers_actuatorfrc").data


def get_jnt_range(m: mujoco.MjModel,
                  joint: str = None) -> np.ndarray:
    return m.jnt_range[get_joint_id(m, joint)]


def get_jnt_ranges(m: mujoco.MjModel) -> np.ndarray:
    return m.jnt_range


def get_ctrl_ranges(m: mujoco.MjModel) -> np.ndarray:
    return m.actuator_ctrlrange


def get_children_deep(m: mujoco.MjModel, 
                      parent_id: int) -> list:
    children = []
    for i in range(m.nbody):
        if m.body_parentid[i] == parent_id:
            children.append(i)
            children.extend(get_children_deep(m, i))  # recursive call
    return children


def get_children_shallow(m: mujoco.MjModel,
                         parent_id: int) -> list:
    children = []
    for i in range(m.nbody):
        if m.body_parentid[i] == parent_id:
            children.append(i)
    return children


###################### SHORTHANDS ############################

def get_ur3e_qpos(d: mujoco.MjData) -> np.ndarray:
    return d.qpos[:6]


def get_ur3e_qvel(d: mujoco.MjData) -> np.ndarray:
    return d.qvel[:6]


def get_robot_qpos(d: mujoco.MjData) -> np.ndarray:
    return d.qpos[:14]


def get_robot_qvel(d: mujoco.MjData) -> np.ndarray:
    return d.qvel[:14]


def get_2f85_xpos(m: mujoco.MjModel,
                  d: mujoco.MjData) -> np.ndarray:
    return get_site_xpos(m, d, "tcp")

def get_2f85_xvelp(m: mujoco.MjModel,
                   d: mujoco.MjData) -> np.ndarray:
    return get_site_velp(m, d, "tcp")

def get_fingers_rel_dist(m: mujoco.MjModel,
                         d: mujoco.MjData) -> np.ndarray:
    return np.linalg.norm(
        get_site_xpos(m, d, "left_pad1_site") - 
        get_site_xpos(m, d, "right_pad1_site")
    )

def get_finger_jnt_disp(d: mujoco.MjData) -> np.ndarray:
    # 6 = "right_spring_link", 10 = "left_spring_link" 
    # Assume initial qpos = 0 for these joints
    # return np.array([d.qpos[6], d.qpos[10]])
    return d.qpos[6] # rad

def get_finger_jnt_vel(d: mujoco.MjData) -> np.ndarray:
    # 6 = "right_spring_link", 10 = "left_spring_link" 
    return d.qvel[6] # rad/s