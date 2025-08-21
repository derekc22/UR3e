import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
from controller.controller_func import euler_to_R, get_xpos, get_xrot
matplotlib.use('Agg')  # Set backend to non-interactive
import yaml
np.set_printoptions(precision=3)


with open("mujoco/gains2.yml", "r") as f:
    yml = yaml.safe_load(f)
    
kp = yml["pos"]["kp"]
kd = yml["pos"]["kd"]
ki = yml["pos"]["ki"]


# def get_state(m, d):
    
#     sensor_site_2f85 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_pad1_site")
#     pos_2f85 = d.site(sensor_site_2f85).xpos.copy()
    
#     rot_2f85 = np.array([0, 0, 0]) # UPDATE THIS ONCE ROTATION IS IMPLEMENTED
    
#     grip_2f85 = np.array([0]) # UPDATE THIS ONCE GRIPPER IS IMPLEMENTED
    
#     return np.concatenate([
#         pos_2f85, rot_2f85, grip_2f85
#     ])
    
    
def get_state(m, d):
    
    pos_2f85 = get_xpos(m, d, "right_pad1_site")

    rot_2f85 = get_xrot(m, d, "right_pad1_site")

    grip_2f85 = np.array([0]) # UPDATE THIS ONCE GRIPPER IS IMPLEMENTED

    return np.concatenate([
        pos_2f85, rot_2f85, grip_2f85
    ])


def plot_trajectory(traj_target, traj):
    
    err = traj_target - traj
    t = np.arange(err.shape[0])
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 8)) # width, height
    axes = axes.flatten()

    # pos plotting
    pos_2f85_target = traj_target[:, :3]
    pos_2f85_traj = traj[:, :3]
    pos_2f85_err = err[:, :3]

    for i in range(3):
        ax = axes[i]
        ax.plot(t, pos_2f85_target[:, i],'C0')
        ax.plot(t, pos_2f85_traj[:, i],'C1')
        ax.plot(t, pos_2f85_err[:, i],'C2')
        ax.set_title(["x", "y", "z"][i])
        # axes[i].grid(True)
        ax.set_xlim(left=0)
        
        
    # for i in range(3, 6):
        # # rot plotting
        # pass


    axes[7].set_visible(False)
    axes[8].set_visible(False)
    plt.tight_layout()    
    plt.savefig("plots.jpg")
        
        
    
    
    


def load_model(model_path):
    # ur3e  = 6  nq, 6  nv, 6 nu
    # 2f85  = 8  nq, 8  nv, 1 nu
    # total = 14 nq, 14 nv, 7 nu
    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    
    return m, d



def load_trajectory_file():
    traj = np.genfromtxt('mujoco/trajectory.csv', delimiter=',', skip_header=1).reshape(-1, 7)
    
    euler_angles = traj[:, 3:6]
    # print(euler_angles)
    # exit()
    
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
        np.zeros(shape=(1, 7)),
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

    traj = load_trajectory_file()
    
    if hold > 1:
        traj = np.repeat(traj, hold, axis=0)
        
    return traj
    

    
    
    
# def pd_ctrl(t, m, d, traj_i):
#     pos_torques = pos_pd_ctrl(t, m, d, traj_i[:3])
#     rot_torques = rot_pd_ctrl(m, d, traj_i[3:7])  # 7: quaternion
    
#     return pos_torques + rot_torques


def ctrl(t, m, d, traj_i):
    pos_torques = pd_ctrl(t, m, d, traj_i[:3], pos_err)    
    rot_torques = pd_ctrl(t, m, d, traj_i[3:12], rot_err)  # rotation matrix
    
    return pos_torques + rot_torques



def pos_err(m, d, xpos_target):
    
    sensor_site_2f85, xpos_2f85 = get_xpos(m, d, "right_pad1_site")
    
    # Compute 3D cartesian position error
    xpos_delta = xpos_target - xpos_2f85
    print(f"pos_curr:{xpos_2f85}, pos_target:{xpos_target}, pos_err:{xpos_delta}")


    # Get arm joints and their velocity addresses
    ur3e_joint_indices = np.arange(6)

    # Compute full Jacobian and extract columns for arm joints
    Jp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, Jp, None, sensor_site_2f85)
    Jp_arm = Jp[:, ur3e_joint_indices]  # Extract relevant columns (3x6)

    Jp_pinv = np.linalg.pinv(Jp_arm)
    # J_JT = Jp_arm @ Jp_arm.T
    # damping = 0.1
    # regularization = damping**2 * np.eye(3)
    # Jp_pinv = Jp_arm.T @ np.linalg.inv(J_JT + regularization)

    # Compute joint angle updates
    return Jp_pinv @ xpos_delta # theta_delta
    
    
    
# def pos_pd_ctrl(t, m, d, xpos_target):
    
#     theta_delta = pos_err(t, m, d, xpos_target)
    
#     torques = np.zeros((6, ))
    
#     for i in range(torques.shape[1]):  

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
#         pos_err = curr_joint_target_angle - curr_joint_angle
#         # torque = kp[i] * pos_err + kd[i] * -curr_joint_vel # pd
#         torque = kp[i] * pos_err + kd[i] * -curr_joint_vel + ki[i]*pos_err*t # pid
        
#         torques[i] = torque

#     return torques


def rot_err(m, d, xrot_target):
    
    sensor_site_2f85, xrot_2f85 = get_xrot(m, d, "right_pad1_site")
    
    # Compute rotational error
    # 
    
    
    
    print(f"rot_curr:{xrot_2f85}, rot_target:{xrot_target}, rot_err:{xrot_delta}")


    # Get arm joints and their velocity addresses
    ur3e_joint_indices = np.arange(6)

    # Compute full Jacobian and extract columns for arm joints
    Jr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, None, Jr, sensor_site_2f85)
    Jr_arm = Jr[:, ur3e_joint_indices]  # Extract relevant columns (3x6)

    Jr_pinv = np.linalg.pinv(Jr_arm)
    # J_JT = Jr_arm @ Jr_arm.T
    # damping = 0.1
    # regularization = damping**2 * np.eye(3)
    # Jr_pinv = Jr_arm.T @ np.linalg.inv(J_JT + regularization)

    # Compute joint angle updates
    print(Jr_pinv.shape)
    print(xrot_delta.shape)
    return Jr_pinv @ xrot_delta # theta_delta
    
    
    
# def rot_pd_ctrl(t, m, d, xquat_target):
    
#     theta_delta = pos_err(t, m, d, xquat_target)
    
#     torques = np.zeros((6, ))
    
#     for i in range(torques.shape[1]):  

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
#         pos_err = curr_joint_target_angle - curr_joint_angle
#         # torque = kp[i] * pos_err + kd[i] * -curr_joint_vel # pd
#         torque = kp[i] * pos_err + kd[i] * -curr_joint_vel + ki[i]*pos_err*t # pid
        
#         torques[i] = torque

#     return torques



def pd_ctrl(t, m, d, target, err_func):
    
    theta_delta = err_func(m, d, target)
    
    torques = np.zeros((6, ))
    
    for i in range(torques.shape[0]):  

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
        torque = kp[i] * err + kd[i] * -curr_joint_vel + ki[i]*err*t # pid
        
        torques[i] = torque

    return torques







def main():
        
    model_path = "assets/ur3e_2f85.xml"
    m, d = load_model(model_path)
    
    traj_target = build_trajectory(hold=1)

    # n = 100
    # traj_target = build_discretized_trajectory(n, hold=200)
    
    N = traj_target.shape[0]
    
    traj_true = np.zeros_like(traj_target)
    
    viewer = mujoco.viewer.launch_passive(m, d)
    
    start = time.time()
    t = time.time() - start
    
    # i = 0
    # while t < 30: #np.inf:
    #     t = time.time() - start
    
    #     if i < N:
    #         d.ctrl[:-1] = pd_ctrl(t, m, d, traj_target[i, :])
    #         traj_true[i] = get_state(m, d)
    #         i += 1
    #     else:
    #         d.ctrl[:-1] = pd_ctrl(t, m, d, traj_target[-1, :])
    #         traj_target = np.append(traj_target, [traj_target[-1]], axis=0)
    #         traj_true = np.append(traj_true, [get_state(m, d)], axis=0)
        
    #     time.sleep(0.01)
        
    #     mujoco.mj_step(m, d)
    #     viewer.sync()
        
    i = 0
    while i < N:
 
        d.ctrl[:-1] = ctrl(t, m, d, traj_target[i, :])
        traj_true[i] = get_state(m, d)
        i += 1
        
        time.sleep(0.01)
        
        mujoco.mj_step(m, d)
        viewer.sync()
        
    plot_trajectory(traj_target, traj_true)
        
        
    
        
    
    
    
if __name__ == "__main__":
    
    main()