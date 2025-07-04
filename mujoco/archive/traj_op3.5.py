import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
from utils import euler_to_R, R_to_euler, get_xpos, get_xrot, get_grip_ctrl, get_grasp_force
matplotlib.use('Agg')  # Set backend to non-interactive
import yaml
from datetime import datetime
import os
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf)
from plot import plot_trajectory, plot_2d_trajectory, plot_3d_trajectory
import shutil
from scipy.spatial.transform import Rotation as R

    
def get_state(m, d):
    
    _, pos_2f85 = get_xpos(m, d, "right_pad1_site")
    _, rot_2f85 = get_xrot(m, d, "right_pad1_site")
    grip_2f85 = get_grasp_force(d)[0:1] #get_grip_ctrl(d)

    return np.concatenate([
        pos_2f85, rot_2f85, grip_2f85
    ])




def load_model(model_path):    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    
    return m, d



def load_trajectory_file():
    traj = np.genfromtxt(trajectory_path, delimiter=',', skip_header=1).reshape(-1, 7)
    
    # euler angles are specified as roll, pitch, yaw in the csv. 
    # euler_to_R(yaw, pitch, roll) is expected. Thus, flip it with negative slicing
    euler_angles = traj[:, 3:6] 
    
    rot_matrix = np.apply_along_axis(
        euler_to_R, 
        axis=1, arr=euler_angles
    )
        
    return np.concatenate(
        [traj[:, 0:3], rot_matrix, traj[:, 6:]],
        axis=1
    )
    
    
    
    


def build_interpolated_trajectory(n, hold=1):
    
    traj = np.concatenate([
        np.zeros(shape=(1, 13)),
        load_trajectory_file()
    ], axis=0)

    nrow = traj.shape[0]

    # Interpolation factors: [1/(n+1), 2/(n+1), ..., n/(n+1)]
    alphas = np.linspace(0, 1, n + 2)[1:-1]  # exclude 0 and 1

    # Temporary list to hold the final trajectory
    result = []

    for i in range(nrow - 1):
        start = traj[i]
        end = traj[i + 1]

        # Append the start point 'hold' times
        if i > 0: result.extend([start] * hold)
        else: result.extend([start])

        # Interpolated points between start and end (not held)
        interpolated_rows = start + (end - start)[None, :] * alphas[:, None]
        result.extend(interpolated_rows)

    # Append the last waypoint 'hold' times
    result.extend([traj[-1]] * hold)

    return np.vstack(result)




def build_trajectory(hold=1):

    # traj = load_trajectory_file()
    # if hold > 1:
    #     traj = np.repeat(traj, hold, axis=0)
            
    return np.repeat(load_trajectory_file(), hold, axis=0)
    

    
    


def ctrl(t, m, d, traj_i):
    # pos_u = pd_ctrl(t, m, d, traj_i[:3], pos_err, pos_gains, tot_pos_joint_errs)    
    rot_u = pd_ctrl(t, m, d, traj_i[3:12], rot_err, rot_gains, tot_rot_joint_errs)  # rotation matrix

    grip_u = grip_ctrl(m, traj_i[-1])
    
    return np.hstack([
        # pos_u + rot_u, 
        rot_u, 
        grip_u
    ])


def grip_ctrl(m, traj_i):
    ctrl_range = m.actuator_ctrlrange[-1] # 'fingers_actuator' is the last actuator
    return traj_i*ctrl_range[1] 


def pos_err(t, m, d, xpos_target):
    
    sensor_site_2f85, xpos_2f85 = get_xpos(m, d, "right_pad1_site")
    
    # Compute 3D cartesian position error
    xpos_delta = xpos_target - xpos_2f85
    
    update_errs(t, pos_errs, xpos_delta)

    # Get arm joints and their velocity addresses
    ur3e_joint_indices = np.arange(num_joints)

    # Compute full Jacobian and extract columns for arm joints
    Jp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, Jp, None, sensor_site_2f85)
    Jp_arm = Jp[:, ur3e_joint_indices]  # Extract relevant columns (3x6)

    Jp_pinv = np.linalg.pinv(Jp_arm)

    # Compute joint angle updates
    return Jp_pinv @ xpos_delta # theta_delta
    
    
    

def rot_err(t, m, d, xrot_target):
    
    # sensor_site_2f85, xrot_2f85 = get_xrot(m, d, "right_pad1_site")    
    # xrot_target = xrot_target.reshape(3, 3)
    # xrot_2f85 = xrot_2f85.reshape(3, 3)
    
    sensor_site_2f85, xrot_2f85 = get_xrot(m, d, "right_pad1_site") 
    xrot_2f85 = xrot_2f85.reshape(3, 3)
    # convert xrot_target from body frame-relative to global frame-relative by pre-multiplying by xrot_2f85
    
    xrot_target = xrot_2f85 @ xrot_target.reshape(3, 3) 
    
    # #########################################################################################################################
    # # Compute rotational error (original)
    
    # R_err = xrot_target @ xrot_2f85.T
    
    # # Compute the rotation angle
    # cos_theta = (np.trace(R_err) - 1) / 2
    # cos_theta = np.clip(cos_theta, -1, 1)  # Avoid numerical issues
    # theta = np.arccos(cos_theta)

    # # Compute the rotation axis (skew-symmetric part)
    # axis = np.array([
    #     R_err[2, 1] - R_err[1, 2],
    #     R_err[0, 2] - R_err[2, 0],
    #     R_err[1, 0] - R_err[0, 1]
    # ])

    # # Normalize axis if theta is not near zero
    # if theta > 1e-5: axis = axis / (2 * np.sin(theta))
    # else: axis = axis / 2 # Small angle approximation

    # xrot_delta = theta * axis  # Orientation error vector
    
        
    # #########################################################################################################################

    
    #########################################################################################################################
    # Compute rotational error (deepseek)
    

    # Compute rotational error
    # # Calculate relative rotation: from current to target frame
    # R_err = xrot_target @ xrot_2f85.T
    
    # # Extract skew-symmetric part (logarithmic map)
    # skew_sym = (R_err - R_err.T) / 2.0
    
    # # Convert skew-symmetric matrix to rotation vector (axis-angle)
    # xrot_delta = np.array([skew_sym[2, 1], 
    #                        skew_sym[0, 2], 
    #                        skew_sym[1, 0]])
    
    
    #########################################################################################################################
    
    #########################################################################################################################
    # Compute rotational error (o4 mini-high)
    

    # R_err = xrot_target @ xrot_2f85.T
    # # ensure numerical safety
    # cos_theta = (np.trace(R_err) - 1.0) / 2.0
    # cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # theta = np.arccos(cos_theta)
    # if np.abs(theta) < 1e-6:
    #     xrot_delta = np.zeros(3)
    # else:
    #     # extract the axis ∝ [R_err(3,2)-R_err(2,3), …]
    #     axis = np.array([
    #         R_err[2,1] - R_err[1,2],
    #         R_err[0,2] - R_err[2,0],
    #         R_err[1,0] - R_err[0,1]
    #     ]) / (2.0 * np.sin(theta))
    #     xrot_delta = axis * theta


    #########################################################################################################################
    
    #########################################################################################################################
    # Compute rotational error (o3)
    
    R_err = xrot_target @ xrot_2f85.T                 # desired-to-current rotation
    skew = 0.5 * (R_err - R_err.T)
    xrot_delta =  np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

    #########################################################################################################################

    #########################################################################################################################
    # Compute rotational error (4o)
    
    # Compute relative rotation matrix (target * current.T)
    # R_err = xrot_target @ xrot_2f85.T

    # Convert to rotation vector (axis-angle representation)
    # xrot_delta = R.from_matrix(R_err).as_rotvec()

    #########################################################################################################################


    # update_errs(t, rot_errs, R_err.flatten())
    # update_errs(t, rot_errs, (np.linalg.pinv(xrot_2f85) @ R_err).flatten())
    # update_errs(t, rot_errs, (xrot_2f85.T @ R_err).flatten())
    
    R_err = xrot_target @ (xrot_2f85.T @ xrot_2f85.T) 
    # print(np.linalg.pinv(xrot_2f85)@   ( xrot_2f85@xrot_target @ xrot_2f85.T  )     ).flatten())

    # Get arm joints and their velocity addresses
    ur3e_joint_indices = np.arange(num_joints)

    # Compute full Jacobian and extract columns for arm joints
    Jr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, None, Jr, sensor_site_2f85)
    Jr_arm = Jr[:, ur3e_joint_indices]  # Extract relevant columns (3x6)

    Jr_pinv = np.linalg.pinv(Jr_arm)

    # Compute joint angle updates

    return Jr_pinv @ xrot_delta # theta_delta
    



def pd_ctrl(t, m, d, target, err_func, gains, tot_joint_errs):
    
    theta_delta = err_func(t, m, d, target)
    
    u = np.zeros((num_joints, ))
    
    kp, kd, ki = gains.values()
    
    for i in range(num_joints):  # 6 joints

        # Get current state
        curr_joint_angle = d.qpos[i]
        curr_joint_vel = d.qvel[i]
        
        # Compute target angle with clamping
        curr_joint_target_angle = curr_joint_angle + theta_delta[i]
        joint_range = m.jnt_range[i]

        if joint_range[0] < joint_range[1]:  # Check valid limits
            curr_joint_target_angle = np.clip(
                curr_joint_target_angle, 
                joint_range[0], joint_range[1]
            )
        
        # PD torque calculation
        err = curr_joint_target_angle - curr_joint_angle
        # torque = kp[i] * err + kd[i] * -curr_joint_vel # pd
        
        update_joint_errs(i, tot_joint_errs, err)
        torque = kp[i] * err + kd[i] * -curr_joint_vel + ki[i]*tot_joint_errs[i]*dt # pid
        u[i] = torque

    return u


def update_errs(t, errs, err):
    errs[t] = err

def update_joint_errs(i, tot_joint_errs, joint_err):
    tot_joint_errs[i] += joint_err




def cleanup():

    dtn = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(log_path+dtn, exist_ok=True)
    
    plot_trajectory(traj_target, traj_true, pos_errs, rot_errs, T, dtn)
    plot_3d_trajectory(traj_target, traj_true, pos_errs, dtn)
    plot_2d_trajectory(traj_target, traj_true, pos_errs, dtn)
        
    with open(f"{log_path + dtn}/log.yml", 'w') as f: yaml.dump(yml, f)
    shutil.copy(trajectory_path, log_path + dtn)



model_path = "assets/ur3e_2f85.xml"
trajectory_path = "mujoco/trajectory.csv"
config_path = "mujoco/config.yml"
log_path = "mujoco/logs/"


with open(config_path, "r") as f: yml = yaml.safe_load(f)
pos_gains = yml["pos"]
rot_gains = yml["rot"]
hold = yml["hold"]
n = yml["n"]

# ur3e  = 6  nq, 6  nv, 6 nu
# 2f85  = 8  nq, 8  nv, 1 nu
# total = 14 nq, 14 nv, 7 nu
m, d = load_model(model_path)
dt = m.opt.timestep
num_joints = 6

traj_target = build_interpolated_trajectory(n=n, hold=hold) if n else build_trajectory(hold=hold)
T = traj_target.shape[0]
traj_true = np.zeros_like(traj_target)

pos_errs = np.zeros(shape=(T, 3))
rot_errs = np.zeros(shape=(T, 9))
grip_errs = np.zeros(shape=(T, 1))
tot_pos_joint_errs = np.zeros(num_joints)
tot_rot_joint_errs = np.zeros(num_joints)



def reset(m, d, intialization='home'):
    init_qp = np.array(m.keyframe(intialization).qpos)
    mujoco.mj_resetData(m, d) 
    d.qpos[:] = init_qp
    mujoco.mj_step(m, d)

def main():
    
    viewer = mujoco.viewer.launch_passive(m, d)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            
    reset(m, d)
    
    try:
        for t in range (T):
    
            d.ctrl = ctrl(t, m, d, traj_target[t, :])

            mujoco.mj_step(m, d)
            viewer.sync()
            
            traj_true[t] = get_state(m, d)
            
            # print(f"pos_target: {traj_target[t, :3]}, pos_true: {traj_true[t, :3]}, pos_err: {pos_errs[t, :]}")
            print(f"rot_target: {R_to_euler(traj_target[t, 3:12].reshape(3, 3))}, rot_true: {R_to_euler(traj_true[t, 3:12].reshape(3, 3))}, rot_err: {R_to_euler(rot_errs[t, :].reshape(3, 3))}")
            # print(f"grip_target: {traj_target[t, -1]}")
            print("------------------------------------------------------------------------------------------")
            
            # time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    
    finally:
        viewer.close()
        cleanup()

        
        
    
        
    
    
    
if __name__ == "__main__":
    
    main()