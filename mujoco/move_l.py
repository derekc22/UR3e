import mujoco
import numpy as np
import matplotlib
from utils import (
    load_model, reset,
    get_xpos, get_xrot, get_grip_ctrl, get_grasp_force, R_to_axis_angle
    pd_ctrl, update_errs
    )
from gen_traj import gen_traj_l
from mujoco.aux import build_trajectory, build_interpolated_trajectory
from aux import cleanup
matplotlib.use('Agg')  # Set backend to non-interactive
import yaml
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf)

    

def get_state(m, d):
    
    _, xpos_2f85 = get_xpos(m, d, "right_pad1_site")
    _, xrot_2f85 = get_xrot(m, d, "right_pad1_site")
    grip_2f85 = get_grasp_force(d)[0:1] #get_grip_ctrl(d)

    return np.concatenate([
        xpos_2f85, xrot_2f85, grip_2f85
    ])


def ctrl(t, m, d, traj_i):
    pos_u = pd_ctrl(t, m, d, traj_i[:3], get_pos_err, pos_gains, tot_pos_joint_errs)    
    rot_u = pd_ctrl(t, m, d, traj_i[3:12], get_rot_err, rot_gains, tot_rot_joint_errs)  # rotation matrix

    grip_u = grip_ctrl(m, traj_i[-1])
    
    return np.hstack([
        pos_u + rot_u, 
        # rot_u, 
        grip_u
    ])


def grip_ctrl(m, traj_i):
    ctrl_range = m.actuator_ctrlrange[-1] # 'fingers_actuator' is the last actuator
    return traj_i*ctrl_range[1] 


def get_pos_err(t, m, d, xpos_target):
    
    sensor_site_2f85, xpos_2f85 = get_xpos(m, d, "right_pad1_site")
    
    # Compute 3D cartesian position error
    xpos_delta = xpos_target - xpos_2f85
    
    update_errs(t, pos_errs, xpos_delta)

    # Get arm joints and their velocity addresses
    ur3e_joint_indices = np.arange(num_joints)

    # Compute full Jacobian and extract columns for arm joints
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, None, sensor_site_2f85)
    jacp_arm = jacp[:, ur3e_joint_indices]  # Extract relevant columns (3x6)

    jacp_pinv = np.linalg.pinv(jacp_arm)

    # Compute joint angle updates
    theta_delta = jacp_pinv @ xpos_delta
    return theta_delta # theta_delta
    
    
    

def get_rot_err(t, m, d, xrot_target):
    
    sensor_site_2f85, xrot_2f85 = get_xrot(m, d, "right_pad1_site") 
    xrot_2f85 = xrot_2f85.reshape(3, 3) # Rgc
    xrot_target = xrot_target.reshape(3, 3) 
     
    #########################################################################################################################
    # Compute rotational error (o3)
    
    R_err = xrot_target @ xrot_2f85.T                 # desired-to-current rotation
    skew = 0.5 * (R_err - R_err.T)
    xrot_delta =  np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

    update_errs(t, rot_errs, R_err.flatten())

    # Get arm joints and their velocity addresses
    ur3e_joint_indices = np.arange(num_joints)

    # Compute full Jacobian and extract columns for arm joints
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, None, jacr, sensor_site_2f85)
    jacr_arm = jacr[:, ur3e_joint_indices]  # Extract relevant columns (3x6)

    jacr_pinv = np.linalg.pinv(jacr_arm)

    # Compute joint angle updates

    theta_delta = jacr_pinv @ xrot_delta
    return theta_delta
    





model_path = "assets/ur3e_2f85.xml"
trajectory_fpath = "mujoco/traj_l.csv"
config_path = "mujoco/config_l.yml"
log_fpath = "mujoco/logs/logs_l/"
ctrl_mode = "l"
num_joints = 6

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

gen_traj_l()
traj_target = build_interpolated_trajectory(n, hold, trajectory_fpath, ctrl_mode) if n else build_trajectory(hold, trajectory_fpath, ctrl_mode)
T = traj_target.shape[0]
traj_true = np.zeros_like(traj_target)

pos_errs = np.zeros(shape=(T, 3))
rot_errs = np.zeros(shape=(T, 9))
grip_errs = np.zeros(shape=(T, 1))
tot_pos_joint_errs = np.zeros(num_joints)
tot_rot_joint_errs = np.zeros(num_joints)



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
            # print(f"rot_target: {R_to_axis_angle(traj_target[t, 3:12].reshape(3, 3))}, rot_true: {R_to_axis_angle(traj_true[t, 3:12].reshape(3, 3))}, rot_err: {R_to_axis_angle(rot_errs[t, :].reshape(3, 3))}")
            # print(f"grip_target: {traj_target[t, -1]}")
            # print("------------------------------------------------------------------------------------------")
            
            # time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    
    finally:
        viewer.close()
        cleanup(traj_target, traj_true, T, 
                trajectory_fpath, log_fpath, yml, 
                ctrl_mode, pos_errs=pos_errs, rot_errs=rot_errs
            )

        
        
    
        
    
    
    
if __name__ == "__main__":
    
    main()