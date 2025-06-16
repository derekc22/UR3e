from typing import Tuple, List
import numpy as np

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
    


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (radians) to a unit quaternion."""
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

    return [qw, qx, qy, qz]