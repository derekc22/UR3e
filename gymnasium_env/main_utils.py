import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from typing import Callable

################################################################################################################################################################################################################################################
# main.xml
################################################################################################################################################################################################################################################


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
    



################################################################################################################################################################################################################################################
# ure3_2f85.xml
################################################################################################################################################################################################################################################


def get_arm_qpos(d: mujoco.MjData) -> np.array:
    return d.qpos[:6]



def load_model(model_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    
    return m, d





def reset(m: mujoco.MjModel, 
          d: mujoco.MjData, 
          intialization: str = 'home') -> None :
    init_qp = np.array(m.keyframe(intialization).qpos)
    mujoco.mj_resetData(m, d) 
    d.qpos[:] = init_qp
    mujoco.mj_step(m, d)



def get_xpos(m: mujoco.MjModel, 
             d: mujoco.MjData, 
             site: str) -> tuple[int, np.array]:
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    return site_id, d.site(site_id).xpos

# def get_xquat(m, d, site):
#     site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
#     # xmat = d.site(site_id).xmat.copy()
#     xmat = np.array(d.site(site_id).xmat).reshape(3, 3)
#     return site_id, R.from_matrix(xmat).as_quat()

def get_xrot(m: mujoco.MjModel, 
             d: mujoco.MjData, 
             site: str) -> tuple[int, np.array]:
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    return site_id, d.site(site_id).xmat


def get_grip_ctrl(d):    
    # print(d.sensor("fingers_actuatorfrc").data)
    return d.sensor("fingers_actuatorfrc").data # actuatorfrc (0, 255) ???



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
    
    _, xpos_2f85 = get_xpos(m, d, "right_pad1_site")
    _, xrot_2f85 = get_xrot(m, d, "right_pad1_site")
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