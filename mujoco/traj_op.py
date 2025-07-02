import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
import yaml
np.set_printoptions(precision=3)

# kp = [
#     20, # Shoulder Pan
#     150, # Shoulder Lift
#     50, # Elbow
#     50, # Wrist 1
#     50, # Wrist 2
#     50, # Wrist 3
# ]

# kd = [
#     5.0, # Shoulder Pan
#     15.0, # Shoulder Lift
#     1.00, # Elbow
#     1.00, # Wrist 1
#     1.00, # Wrist 2
#     1.00, # Wrist 3
# ]

# ki = [
#     5.0, # Shoulder Pan
#     17.0, # Shoulder Lift
#     1.00, # Elbow
#     1.00, # Wrist 1
#     1.00, # Wrist 2
#     1.00, # Wrist 3
# ]


with open("mujoco/gains.yml", "r") as f:
    yml = yaml.safe_load(f)
    
kp = yml["kp"]
kd = yml["kd"]
ki = yml["ki"]


def get_state(m, d):
    
    sensor_site_2f85 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_pad1_site")
    pos_2f85 = d.site(sensor_site_2f85).xpos.copy()
    
    rot_2f85 = np.array([0, 0, 0]) # UPDATE THIS ONCE ROTATION IS IMPLEMENTED
    
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


# def discretize_trajectory(traj_target, n, hold=1):
    
#     num_rows, num_cols = traj_target.shape

#     # Number of new rows to be inserted between each pair
#     total_new_rows = (num_rows - 1) * (n + 1) + 1
#     interpolated_traj = np.empty((total_new_rows, num_cols))

#     # Interpolation factors: [0, 1/(n+1), 2/(n+1), ..., 1]
#     alphas = np.linspace(0, 1, n + 2)[1:-1]  # exclude start (0) and end (1)

#     row_idx = 0
#     for i in range(num_rows - 1):
#         start = traj_target[i]
#         end = traj_target[i + 1]

#         # Add the original row
#         interpolated_traj[row_idx] = start
#         row_idx += 1

#         # Broadcast interpolate
#         interpolated_rows = start + (end - start)[None, :] * alphas[:, None]
#         interpolated_traj[row_idx:row_idx + n] = interpolated_rows
#         row_idx += n

#     # Add the final row
#     interpolated_traj[row_idx] = traj_target[-1]

#     return interpolated_traj



def build_discretized_trajectory(n, hold=1):
    
    # traj = np.genfromtxt('mujoco/trajectory.csv', delimiter=',', skip_header=1).reshape(-1, 7)
    
    traj = np.concatenate([
        np.zeros(shape=(1, 7)),
        np.genfromtxt('mujoco/trajectory.csv', delimiter=',', skip_header=1).reshape(-1, 7)
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


# def build_trajectory():
#     # x1, y1, z1, rx1, ry1, rz1, g1 = (1, 2, 3, 0, 0, 0, 0)
#     # x2, y2, z2, rx2, ry2, rz2, g2 = (2, 3, 4, 5, 6, 7, 0)
#     # return np.array([
#     #     [x1, y1, z1, rx1, ry1, rz1, g1],
#     #     [x2, y2, z2, rx2, ry2, rz2, g2],
#     # ])
    
#     return np.concatenate([
#             np.zeros(shape=(1, 7)),
#             np.genfromtxt('mujoco/trajectory.csv', delimiter=',', skip_header=1).reshape(-1, 7)
#         ], axis=0)
    

def build_trajectory(hold=1):

    traj = np.genfromtxt('mujoco/trajectory.csv', delimiter=',', skip_header=1).reshape(-1, 7)
    
    if hold > 1:
        traj = np.repeat(traj, hold, axis=0)
        
    return traj
    

    
    
    
def pd_ctrl(t, m, d, traj_i):

    return (
        pos_pd_ctrl(t, m, d, traj_i[:3])    
        # rot_pd_ctrl(m, d, traj_i[3:6])
        # grip_pd_ctrl(m, d, traj_i[7])
    )
    
    
def pos_pd_ctrl(t, m, d, xpos_target):

    sensor_site_2f85 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_pad1_site")
    xpos_2f85 = d.site(sensor_site_2f85).xpos.copy()
    
    # Compute 3D cartesian position error
    x_delta = xpos_target - xpos_2f85
    print(f"curr:{xpos_2f85}, target:{xpos_target}, err:{x_delta}")


    # Get leg joints and their velocity addresses
    ur3e_joint_indices = np.arange(6)

    # Compute full Jacobian and extract columns for leg joints
    Jp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, Jp, None, sensor_site_2f85)
    Jp_arm = Jp[:, ur3e_joint_indices]  # Extract relevant columns (3x6)

    Jp_pinv = np.linalg.pinv(Jp_arm)
    # J_JT = Jp_arm @ Jp_arm.T
    # damping = 0.1
    # regularization = damping**2 * np.eye(3)
    # Jp_pinv = Jp_arm.T @ np.linalg.inv(J_JT + regularization)

    # Compute joint angle updates
    theta_delta = Jp_pinv @ x_delta
    
    
    torques = np.zeros((6, ))
    
    for i in range(len(ur3e_joint_indices)):  

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
        pos_err = curr_joint_target_angle - curr_joint_angle
        # torque = kp[i] * pos_err + kd[i] * -curr_joint_vel # pd
        torque = kp[i] * pos_err + kd[i] * -curr_joint_vel + ki[i]*pos_err*t # pid
        
        torques[i] = torque

    return torques


def main():
        
    model_path = "assets/ur3e_2f85.xml"
    m, d = load_model(model_path)
    
    # traj_target = build_trajectory(hold=500)
    
    n = 100
    traj_target = build_discretized_trajectory(n, hold=200)
    
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
 
        d.ctrl[:-1] = pd_ctrl(t, m, d, traj_target[i, :])
        traj_true[i] = get_state(m, d)
        i += 1
        
        time.sleep(0.01)
        
        mujoco.mj_step(m, d)
        viewer.sync()
        
    plot_trajectory(traj_target, traj_true)
        
        
    
        
    
    
    
if __name__ == "__main__":
    
    main()