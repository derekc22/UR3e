import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from typing import Callable



def load_model(model_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    return m, d


def reset(m: mujoco.MjModel, 
          d: mujoco.MjData, 
          keyframe: str = 'home') -> None:
    # print(m.keyframe("home"))
    # exit()
    init_qpos = m.keyframe(keyframe).qpos
    # print(init_qpos)
    # exit()
    init_qvel = m.keyframe(keyframe).qvel
    # mujoco.mj_resetData(m, d) 
    d.qpos = init_qpos
    d.qvel = init_qvel
    # mujoco.mj_step(m, d)



def get_body_xpos(m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  body: str) -> tuple[int, np.array]:
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return body_id, d.xpos[body_id]


def get_body_xquat(m: mujoco.MjModel, 
                   d: mujoco.MjData,
                   body: str) -> tuple[int, np.array]:
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    xmat = np.array(d.xmat[body_id]).reshape(3, 3)
    return body_id, R.from_matrix(xmat).as_quat()


def get_body_xrot(m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  body: str) -> tuple[int, np.array]:
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return body_id, d.xmat[body_id]


def get_site_xpos(m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  site: str) -> tuple[int, np.array]:
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    return site_id, d.site(site_id).xpos


def get_site_xquat(m: mujoco.MjModel, 
                   d: mujoco.MjData,
                   site: str) -> tuple[int, np.array]:
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    xmat = np.array(d.site(site_id).xmat).reshape(3, 3)
    return site_id, R.from_matrix(xmat).as_quat()


def get_site_xrot(m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  site: str) -> tuple[int, np.array]:
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    return site_id, d.site(site_id).xmat


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

def get_grasp_force(d):
    contact_pads = [
        "right_pad1_contact",
        # "right_pad2_contact",
        "left_pad1_contact",
        # "left_pad2_contact",
    ]
    # return { pad : float(d.sensor(pad).data[0]) for pad in contact_pads }
    return np.array([ d.sensor(pad).data[0] for pad in contact_pads ])



def get_grip_ctrl(d: mujoco.MjData) -> np.array: 
    # print(d.sensor("fingers_actuatorfrc").data)
    return d.sensor("fingers_actuatorfrc").data # actuatorfrc (0, 255) ???



