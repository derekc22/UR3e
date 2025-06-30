import numpy as np
import time
import cvxopt
from cvxopt import solvers
# import osqp
from scipy import sparse
# import pyqpoases
import mujoco
np.set_printoptions(suppress=True, precision=2)
import matplotlib.pyplot as plt

################## functions #####################

class PD:
    def __init__(self, model):
        self.joint_data = _initJointData(model)
        self.toe_sensor_data = _initToeData(model)

        self.amp_x = 0.5 #5
        self.amp_z = 0.5 #5
        self.freq = 3.2 #0.8

        self.kp = [
            0.10, # Hip Yaw
            10.0, # Hip Roll
            1.20, # Hip Pitch
            6.50, # Knee
            0.50, # Ankle
        ]

        self.kd = [
            1.00, # Hip Yaw
            20.0, # Hip Roll
            0.30, # Hip Pitch
            1.20, # Knee
            0.10  # Ankle
        ]




def _initJointData(m):
    joint_names = [
        f"l_hip_yaw_joint", f"l_hip_roll_joint", f"l_hip_pitch_joint", f"l_knee_joint", f"l_ankle_joint",
        "r_hip_yaw_joint", "r_hip_roll_joint", "r_hip_pitch_joint", "r_knee_joint", "r_ankle_joint",
    ]
    joint_data = {}
    for joint_name in joint_names:
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        joint_data[joint_name] = {
            "joint_id": jnt_id,
            "qpos_adr": m.jnt_qposadr[jnt_id],
            "qvel_adr": m.jnt_dofadr[jnt_id],
        }
    return joint_data



def _initToeData(m):
    toe_names = [f"l_toe_sensor", "r_toe_sensor"]
    toe_sensor_data = {
        toe_name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, toe_name) for toe_name in toe_names 
    }
    return toe_sensor_data


def _generateSinusoidalLegTrajectory(t, side, toe_current_pos, pd):
    phase_shift = 0.0 if side == 'l' else np.pi
    toe_des_x = pd.amp_x * np.sin(2 * np.pi * pd.freq * t + phase_shift)
    toe_des_y = toe_current_pos[1]
    toe_des_z = -0.85 + pd.amp_z * np.sin(2 * np.pi * pd.freq * t + phase_shift)
    toe_desired_pos = np.array([toe_des_x, toe_des_y, toe_des_z])

    return toe_desired_pos


def jointPDControl(t, model, data, pd):
    """Inverse kinematics foot trajectory tracking with joint PD control"""

    torques = np.zeros((10, ))

    for side in ['l', 'r']:

        toe_sensor_site_id = pd.toe_sensor_data[f"{side}_toe_sensor"]
        toe_current_pos = data.site(toe_sensor_site_id).xpos.copy()

        # Trajectory generation
        toe_desired_pos = _generateSinusoidalLegTrajectory(t, side, toe_current_pos, pd)

        # Compute 3D cartesian position error
        delta_x = toe_desired_pos - toe_current_pos
        # print(f"err_left: {delta_x}, curr_left: {toe_current_pos}, des_left: {toe_desired_pos}",  end='') if side == 'l' else print(f"; err_right: {delta_x}, curr_right: {toe_current_pos}, des_right: {toe_desired_pos}")
        # print(f"err_left: {delta_x}, des_left: {toe_desired_pos}",  end='') if side == 'l' else print(f"; err_right: {delta_x}, des_right: {toe_desired_pos}")

        # Get leg joints and their velocity addresses
        curr_joints = [ pd.joint_data[joint] for joint in pd.joint_data if joint[0] == side ]
        toe_vel_adrs = [ curr_joint['qvel_adr'] for curr_joint in curr_joints ]

        # Compute full Jacobian and extract columns for leg joints
        Jt_full = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, Jt_full, None, toe_sensor_site_id)
        Jt_leg = Jt_full[:, toe_vel_adrs]  # Extract relevant columns (3x5)

        # Damped Least Squares pseudo-inverse
        J_JT = Jt_leg @ Jt_leg.T
        damping = 0.1
        regularization = damping**2 * np.eye(3)
        J_pinv = Jt_leg.T @ np.linalg.inv(J_JT + regularization)

        # Compute joint angle updates
        delta_theta = J_pinv @ delta_x


        # Apply PD control to each joint
        for i, curr_joint in enumerate(curr_joints):       

            # Get current state
            current_angle = data.qpos[curr_joint['qpos_adr']]
            current_vel = data.qvel[curr_joint['qvel_adr']]
            
            # Compute desired angle with clamping
            desired_angle = current_angle + delta_theta[i]
            joint_range = model.jnt_range[curr_joint['joint_id']]

            if joint_range[0] < joint_range[1]:  # Check valid limits
                desired_angle_clipped = np.clip(desired_angle, joint_range[0], joint_range[1])
            
            # PD torque calculation
            pos_err = desired_angle_clipped - current_angle
            torque = pd.kp[i] * pos_err + pd.kd[i] * -current_vel
            
            j = i if side == 'l' else i + 5
            torques[j] = torque
    
    return torques



if __name__ == "__main__":

    ts = np.linspace(0, 10, 10000)
    traj = np.zeros(shape=(3, ts.shape[0]))

    for i, t in enumerate(ts):
        pd_params = type('', (), {'amp_x': 5, 'amp_z': 5, 'freq': 0.8})()
        curr_traj = _generateSinusoidalLegTrajectory(t, side='l', toe_current_pos=[0, 0, 0], pd=pd_params)
        traj[:, i] = curr_traj

    plt.plot(ts, traj[0, :], label='x')
    plt.plot(ts, traj[1, :], label='y')
    plt.plot(ts, traj[2, :], label='z')
    plt.legend()
    plt.show()

