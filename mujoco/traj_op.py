import mujoco
import numpy as np
import time
np.set_printoptions(precision=3)

kp = [
    20, # Shoulder Pan
    150, # Shoulder Lift
    150, # Elbow
    100, # Wrist 1
    100, # Wrist 2
    100, # Wrist 3
]

kd = [
    5.0, # Shoulder Pan
    5.0, # Shoulder Lift
    1.00, # Elbow
    1.00, # Wrist 1
    1.00, # Wrist 2
    1.00, # Wrist 3
]

ki = [
    5.0, # Shoulder Pan
    10.0, # Shoulder Lift
    1.00, # Elbow
    1.00, # Wrist 1
    1.00, # Wrist 2
    1.00, # Wrist 3
]


def get_state(m, d):
    
    sensor_site_2f85 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_pad1_site")
    xpos_2f85 = d.site(sensor_site_2f85).xpos.copy()
    
    return np.array([
        xpos_2f85, 
    ])
    


def plot_trajectory(traj_target, traj_true):
    
    err = traj_target - traj_true


def load_model(model_path):
    # ur3e  = 6  nq, 6  nv, 6 nu
    # 2f85  = 8  nq, 8  nv, 1 nu
    # total = 14 nq, 14 nv, 7 nu
    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    
    return m, d


def discretize_trajectory(traj_target, n):
    
    num_rows, num_cols = traj_target.shape

    # Number of new rows to be inserted between each pair
    total_new_rows = (num_rows - 1) * (n + 1) + 1
    interpolated_traj = np.empty((total_new_rows, num_cols))

    # Interpolation factors: [0, 1/(n+1), 2/(n+1), ..., 1]
    alphas = np.linspace(0, 1, n + 2)[1:-1]  # exclude start (0) and end (1)

    row_idx = 0
    for i in range(num_rows - 1):
        start = traj_target[i]
        end = traj_target[i + 1]

        # Add the original row
        interpolated_traj[row_idx] = start
        row_idx += 1

        # Broadcast interpolate
        interpolated_rows = start + (end - start)[None, :] * alphas[:, None]
        interpolated_traj[row_idx:row_idx + n] = interpolated_rows
        row_idx += n

    # Add the final row
    interpolated_traj[row_idx] = traj_target[-1]

    return interpolated_traj
    

def load_trajectory(n):
    # x1, y1, z1, rx1, ry1, rz1, g1 = (1, 2, 3, 0, 0, 0, 0)
    # x2, y2, z2, rx2, ry2, rz2, g2 = (2, 3, 4, 5, 6, 7, 0)
    # return np.array([
    #     [x1, y1, z1, rx1, ry1, rz1, g1],
    #     [x2, y2, z2, rx2, ry2, rz2, g2],
    # ])
    
    return np.concatenate([
            np.zeros(shape=(1, 7)),
            np.genfromtxt('mujoco/trajectory.csv', delimiter=',', skip_header=1).reshape(-1, 7)
        ], axis=0)

    
    
    
def pd_ctrl(m, d, traj_i, t):

    return (
        pos_pd_ctrl(m, d, traj_i[:3], t)    
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
    
    n = 3
    traj_target = load_trajectory(n)
    traj_target = discretize_trajectory(traj_target, n)
    N = traj_target.shape[0]
    
    traj_true = np.zeros_like(traj_target)
    
    viewer = mujoco.viewer.launch_passive(m, d)
    
    start = time.time()
    t = time.time() - start
    
    i = 0
    while t < np.inf:
        t = time.time() - start
    
        if i < N:
            d.ctrl[:-1] = pd_ctrl(t, m, d, traj_target[i, :])
            i += 1
        else:
            d.ctrl[:-1] = pd_ctrl(t, m, d, traj_target[-1, :])
            
        # traj_true = get_state()
        
        time.sleep(0.01)
        
        mujoco.mj_step(m, d)
        viewer.sync()
        
        
    
        
    
    
    
if __name__ == "__main__":
    
    main()