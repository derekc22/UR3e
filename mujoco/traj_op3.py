import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
from utils import euler_to_R, R_to_euler, get_xpos, get_xrot
matplotlib.use('Agg')  # Set backend to non-interactive
import yaml
np.set_printoptions(precision=3, linewidth=300)


    
def get_state(m, d):
    
    _, pos_2f85 = get_xpos(m, d, "right_pad1_site")
    _, rot_2f85 = get_xrot(m, d, "right_pad1_site")

    grip_2f85 = np.array([0]) # UPDATE THIS ONCE GRIPPER IS IMPLEMENTED

    return np.concatenate([
        pos_2f85, rot_2f85, grip_2f85
    ])


def plot_trajectory(traj_target, traj):
    
    t = np.arange(traj_target.shape[0])
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 8)) # width, height
    axes = axes.flatten()

    # pos plotting
    pos_2f85_target = traj_target[:, :3]
    pos_2f85_traj = traj[:, :3]
    

    for i in range(3):
        ax = axes[i]
        j = i
        ax.plot(t, pos_2f85_target[:, j],'C0', label="target")
        ax.plot(t, pos_2f85_traj[:, j],'C1', label="true")
        ax.plot(t, pos_errs[:, j],'C2', label="err")
        ax.set_title(["x", "y", "z"][j])
        # axes[j].grid(True)
        ax.set_xlim(left=0)
        # if i == 2:
        ax.legend(loc='upper right')
    
    
    # convert codebase to jax backend and use jax.vmap instead
    rot_2f85_target = np.array([R_to_euler(R) for R in traj_target[:, 3:12].reshape(T, 3, 3)])
    rot_2f85_traj = np.array([R_to_euler(R) for R in traj[:, 3:12].reshape(T, 3, 3)])
    
    for i in range(3, 6):
        ax = axes[i]
        j = i-3
        ax.plot(t, rot_2f85_target[:, j],'C0', label="target")
        ax.plot(t, rot_2f85_traj[:, j],'C1', label="true")
        ax.plot(t, rot_errs[:, j],'C2', label="err")
        ax.set_title(["roll", "pitch", "yaw"][j])
        # axes[j].grid(True)
        ax.set_xlim(left=0)
        # if i == 5:
        ax.legend(loc='upper right')



    mean_pos_errs = np.mean(pos_errs, axis=0)
    mean_rot_errs = np.mean(rot_errs, axis=0)
    title = (
        f"mean_xpos_err:  {mean_pos_errs[0]:.3g}, "
        f"mean_ypos_err:  {mean_pos_errs[1]:.3g}, "
        f"mean_zpos_err:  {mean_pos_errs[2]:.3g}, "
        f"mean_roll_err:  {mean_rot_errs[0]:.3g}, "
        f"mean_pitch_err: {mean_rot_errs[1]:.3g}, "
        f"mean_yaw_err:   {mean_rot_errs[2]:.3g}"
    )
    plt.suptitle(title)
    
    axes[7].set_visible(False)
    axes[8].set_visible(False)
    plt.tight_layout()    
    plt.savefig("./plots.jpg")
        
        
    
    
    


def load_model(model_path):    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    
    return m, d



def load_trajectory_file():
    traj = np.genfromtxt('mujoco/trajectory.csv', delimiter=',', skip_header=1).reshape(-1, 7)
    
    euler_angles = traj[:, 3:6]
    
    rot_matrix = np.apply_along_axis(
        euler_to_R, 
        axis=1, arr=euler_angles
    )
        
    return np.concatenate(
        [traj[:, 0:3], rot_matrix, traj[:, 6:7]],
        axis=1
    )
    
    
    
    


def build_discretized_trajectory(n, hold=1):
    
    traj = np.concatenate([
        np.zeros(shape=(1, 13)),
        load_trajectory_file()
    ], axis=0)

    num_rows, num_cols = traj.shape

    # Interpolation factors: [1/(n+1), 2/(n+1), ..., n/(n+1)]
    alphas = np.linspace(0, 1, n + 2)[1:-1]  # exclude 0 and 1

    # Temporary list to hold the final trajectory
    result = []

    for i in range(num_rows - 1):
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
    pos_torques = pd_ctrl(t, m, d, traj_i[:3], pos_err, pos_gains, tot_pos_joint_errs)    
    rot_torques = pd_ctrl(t, m, d, traj_i[3:12], rot_err, rot_gains, tot_rot_joint_errs)  # rotation matrix
    
    return pos_torques + rot_torques



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
    
    sensor_site_2f85, xrot_2f85 = get_xrot(m, d, "right_pad1_site")
    
    xrot_target = xrot_target.reshape(3, 3)
    xrot_2f85 = xrot_2f85.reshape(3, 3)
    
    #########################################################################################################################
    # Compute rotational error
    
    R_err = xrot_target @ xrot_2f85.T
    
    # Compute the rotation angle
    cos_theta = (np.trace(R_err) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)  # Avoid numerical issues
    theta = np.arccos(cos_theta)

    # Compute the rotation axis (skew-symmetric part)
    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ])

    # Normalize axis if theta is not near zero
    if theta > 1e-5: axis = axis / (2 * np.sin(theta))
    else: axis = axis / 2 # Small angle approximation

    xrot_delta = theta * axis  # Orientation error vector
    
    
    update_errs(t, rot_errs, R_err.flatten())
    
    #########################################################################################################################

    
    # print(f"rot_curr:{xrot_2f85}, rot_target:{xrot_target}, rot_err:{xrot_delta}")
    # print(f"rot_err:{xrot_delta}")


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
    
    torques = np.zeros((num_joints, ))
    
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
        torques[i] = torque

    return torques


def update_errs(t, errs, err):
    errs[t] = err

def update_joint_errs(i, tot_joint_errs, joint_err):
    tot_joint_errs[i] += joint_err




with open("mujoco/gains2.yml", "r") as f: yml = yaml.safe_load(f)
    
pos_gains = yml["pos"]
rot_gains = yml["rot"]

# ur3e  = 6  nq, 6  nv, 6 nu
# 2f85  = 8  nq, 8  nv, 1 nu
# total = 14 nq, 14 nv, 7 nu
model_path = "assets/ur3e_2f85.xml"
m, d = load_model(model_path)
dt = m.opt.timestep
num_joints = 6

# traj_target = build_trajectory(hold=200)
traj_target = build_discretized_trajectory(n=100, hold=100)
T = traj_target.shape[0]

pos_errs = np.zeros(shape=(T, 3))
rot_errs = np.zeros(shape=(T, 9))
tot_pos_joint_errs = np.zeros(num_joints)
tot_rot_joint_errs = np.zeros(num_joints)

def main():
    
    viewer = mujoco.viewer.launch_passive(m, d)
            
    traj_true = np.zeros_like(traj_target)
    
    t = 0
    while t < T:
 
        d.ctrl[:-1] = ctrl(t, m, d, traj_target[t, :])

        mujoco.mj_step(m, d)
        viewer.sync()
        
        traj_true[t] = get_state(m, d)
        
        print(f"pos_target:{traj_target[t, :3]}, pos_true:{traj_true[t, :3]}, pos_err: {pos_errs[t, :]}")
        print(f"rot_target:{R_to_euler(traj_target[t, 3:12].reshape(3, 3))}, rot_true:{R_to_euler(traj_true[t, 3:12].reshape(3, 3))}, rot_err: {R_to_euler(rot_errs[t, :].reshape(3, 3))}")
        print("------------------------------------------------------------------------------------------")
        
        t += 1
        time.sleep(0.01)

    plot_trajectory(traj_target, traj_true)
        
        
    
        
    
    
    
if __name__ == "__main__":
    
    main()