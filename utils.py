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



################################################################################################################################################################################################################################################
# ure3_2f85.xml
################################################################################################################################################################################################################################################


def get_arm_qpos(d):
    return d.qpos[:6]



def load_model(model_path):    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    
    return m, d






# def pd_ctrl(t, m, d, 
#             target_traj, 
#             err_func,
#             gains, 
#             tot_joint_errs,
#             ):
    
#     theta_delta = err_func(t, m, d, target_traj)
#     dt = m.opt.timestep
    
#     num_joints = 6
#     u = np.zeros((num_joints, ))
    
#     kp, kd, ki = gains.values()
    
#     for i in range(num_joints):  # 6 arm joints

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

def pd_ctrl(t, m, d, 
            traj_target, 
            err_func,
            gains, 
            joint_ranges,
            tot_joint_errs,
            ):
    
    theta_delta = err_func(t, m, d, traj_target)
    dt = m.opt.timestep
    
    num_joints = 6
    u = np.zeros((num_joints, ))
    
    kp, kd, ki = gains.values()

    # Get current state
    joint_angles = d.qpos[:6]
    joint_vels = d.qvel[:6]
        
    # Compute target angle with clamping
    joint_target_angles = joint_angles + theta_delta
    
    joint_target_angles = np.clip(
        joint_target_angles, 
        joint_ranges[:, 0],  # Lower bounds of joint_range
        joint_ranges[:, 1]   # Upper bounds of joint_range
    )
    
    # PD torque calculation
    errs = joint_target_angles - joint_angles
    update_joint_errs(tot_joint_errs, errs)
    
    u = kp @ errs + kd @ -joint_vels # pd
    # ui = kp @ errs + kd @ -joint_vels + ki @ tot_joint_errs * dt # pid. 
    # this pid control is actually incorrect because tot_joint_errs should be flushed after each new, non-holding trajectory step
    # that is, once a particular 'hold' is over and the next trajectory step is reached, tot_joint_errs should be reset to zero
    # this current implementation accumulates the joint errors over the entire trajectory (across all unique trajectory steps) which is not correct
    # fix later

    return u



def update_errs(t, errs, err):
    errs[t] = err

# def update_joint_errs(i, tot_joint_errs, joint_err):
#     tot_joint_errs[i] += joint_err

def update_joint_errs(tot_joint_errs, joint_errs):
    tot_joint_errs += joint_errs



def reset(m, d, intialization='home'):
    init_qp = np.array(m.keyframe(intialization).qpos)
    mujoco.mj_resetData(m, d) 
    d.qpos[:] = init_qp
    mujoco.mj_step(m, d)



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





def axis_angle_to_R(axis_angle):
    theta = np.linalg.norm(axis_angle)
    if theta == 0:
        return np.eye(3).flatten()

    u = axis_angle / theta
    ux, uy, uz = u

    K = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])

    I = np.eye(3)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    return R.flatten()



def R_to_axis_angle(R):
    # Ensure R is a numpy array
    R = np.array(R)
    angle = np.arccos((np.trace(R) - 1) / 2)

    if np.isclose(angle, 0):
        return np.zeros(3)

    if np.isclose(angle, np.pi):
        # Special case: angle is 180°, axis can be ambiguous
        # Extract axis from diagonal elements
        R_plus = (R + np.eye(3)) / 2
        axis = np.sqrt(np.maximum(np.diagonal(R_plus), 0))
        axis = axis / np.linalg.norm(axis)
        return angle * axis

    rx = (R[2,1] - R[1,2]) / (2 * np.sin(angle))
    ry = (R[0,2] - R[2,0]) / (2 * np.sin(angle))
    rz = (R[1,0] - R[0,1]) / (2 * np.sin(angle))
    axis = np.array([rx, ry, rz])

    return angle * axis





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



def get_gravity_compensation(m, d):
    """Returns gravity compensation torque for arm joints"""
    return d.qfrc_bias[:6]


def grip_ctrl(m, traj_t):
    ctrl_range = m.actuator_ctrlrange[-1] # 'fingers_actuator' is the last actuator
    return traj_t * ctrl_range[1] 



def R_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion (x, y, z, w)
    
    Parameters:
        R : numpy.ndarray
            A 3x3 rotation matrix

    Returns:
        tuple: Quaternion (x, y, z, w)
    """
    assert R.shape == (3, 3), "Rotation matrix must be 3x3"
    
    trace = np.trace(R)

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S = 4 * w
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * x
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * y
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * z
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return (x, y, z, w)



def get_3D_state(m, d):
    
    _, xpos_2f85 = get_xpos(m, d, "right_pad1_site")
    _, xrot_2f85 = get_xrot(m, d, "right_pad1_site")
    xrot_2f85 = R.from_matrix(xrot_2f85.reshape(3, 3)).as_rotvec()
    grip_2f85 = get_grasp_force(d)[0:1] #get_grip_ctrl(d)

    return np.concatenate([
        xpos_2f85, xrot_2f85, grip_2f85
    ])