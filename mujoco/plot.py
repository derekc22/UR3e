import numpy as np
from utils import R_to_axis_angle, get_joint_torques
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting


def plot_trajectory_l(traj_target, traj_true, pos_errs, rot_errs, T, save_fpath):
    
    t = np.arange(T)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 8)) # width, height
    axes = axes.flatten()

    # pos plotting
    pos_2f85_target = traj_target[:, :3]
    pos_2f85_traj_true = traj_true[:, :3]
    
    for i in range(3):
        ax = axes[i]
        j = i
        ax.plot(t, pos_2f85_target[:, j],'C0', label="target")
        ax.plot(t, pos_2f85_traj_true[:, j],'C1', label="true")
        ax.plot(t, pos_errs[:, j],'C2', label="err")
        ax.set_title(["x", "y", "z"][j])
        ax.grid(True)
        ax.set_xlim(left=0)
        # if i == 2:
        ax.legend(loc='upper right')
    
    
    # convert codebase to jax backend and use jax.vmap instead
    rot_2f85_target = traj_target[:, 3:7]
    rot_2f85_traj_true = traj_true[:, 3:7]
    
    for i in range(3, 6):
        ax = axes[i]
        j = i-3
        ax.plot(t, rot_2f85_target[:, j],'C0', label="target")
        ax.plot(t, rot_2f85_traj_true[:, j],'C1', label="true")
        ax.plot(t, rot_errs[:, j],'C2', label="err")
        ax.set_title(["rx", "ry", "rz"][j])
        ax.grid(True)
        ax.set_xlim(left=0)
        # if i == 5:
        ax.legend(loc='upper right')
        

    ax = axes[6]
    ax.plot(t, traj_target[:, 6],'C0', label="target")
    ax.plot(t, traj_true[:, 6],'C1', label="true")
    ax.set_title("grip")
    ax.grid(True)
    ax.set_xlim(left=0)
    ax.legend(loc='upper right')


    mean_pos_errs = np.mean(pos_errs, axis=0)
    mean_rot_errs = np.mean(rot_errs, axis=0)
    title = (
        f"mean_x_err: {mean_pos_errs[0]:.3g}, "
        f"mean_y_err: {mean_pos_errs[1]:.3g}, "
        f"mean_z_err: {mean_pos_errs[2]:.3g}, "
        f"mean_rx_err: {mean_rot_errs[0]:.3g}, "
        f"mean_ry_err: {mean_rot_errs[1]:.3g}, "
        f"mean_rz_err: {mean_rot_errs[2]:.3g}"
    )
    plt.suptitle(title)
    
    axes[7].set_visible(False)
    axes[8].set_visible(False)
    plt.tight_layout()    
    plt.savefig(f"{save_fpath}/plot.jpg")
            
    
def plot_3d_trajectory(traj_target, traj_true, pos_errs, save_fpath):
    
    x_target, y_target, z_target = traj_target[:, :3].T
    x_true, y_true, z_true = traj_true[:, :3].T
    
    # Create a figure and 3D axis
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the line
    ax.plot(x_target, y_target, z_target, color='C0', label="target")
    ax.plot(x_true, y_true, z_true, color='C1', label="true")

    # Add labels (optional)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
        
    
    mean_pos_errs = np.mean(pos_errs, axis=0)
    title = (
        f"mean_x_err:  {mean_pos_errs[0]:.3g}, "
        f"mean_y_err:  {mean_pos_errs[1]:.3g}, "
        f"mean_z_err:  {mean_pos_errs[2]:.3g}"
    )
    plt.suptitle(title)
    
    plt.tight_layout()    
    plt.savefig(f"{save_fpath}/plot3d.jpg")
    





def plot_2d_trajectory(traj_target, traj_true, pos_errs, save_fpath):
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 4)) # width, height
    axes = axes.flatten()

    indices = [(0, 1), (1, 2), (0, 2)] # (x, y), (y, z), (x, z)
    
    for i, j in enumerate(indices):
        ax = axes[i]
        ax.plot(traj_target[:, j[0]], traj_target[:, j[1]],'C0', label="target")
        ax.plot(traj_true[:, j[0]], traj_true[:, j[1]],'C1', label="true")
        ax.set_title(["xy", "yz", "xz"][i])
        ax.grid(True)
        ax.set_xlim(left=0)
        # if i == 2:
        ax.legend(loc='upper right')

    
    mean_pos_errs = np.mean(pos_errs, axis=0)
    title = (
        f"mean_x_err:  {mean_pos_errs[0]:.3g}, "
        f"mean_y_err:  {mean_pos_errs[1]:.3g}, "
        f"mean_z_err:  {mean_pos_errs[2]:.3g}"
    )
    plt.suptitle(title)
    
    plt.tight_layout()    
    plt.savefig(f"{save_fpath}/plot2d.jpg")
    



def plot_trajectory_j(traj_target, traj_true, qpos_errs, T, save_fpath):
    
    t = np.arange(T)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 8)) # width, height
    axes = axes.flatten()

    # qpos plotting
    qpos_2f85_target = traj_target[:, :6]
    qpos_2f85_traj_true = traj_true[:, :6]
    
    for i in range(6):
        ax = axes[i]
        ax.plot(t, qpos_2f85_target[:, i],'C0', label="target")
        ax.plot(t, qpos_2f85_traj_true[:, i],'C1', label="true")
        ax.plot(t, qpos_errs[:, i],'C2', label="err")
        ax.set_title(["j1", "j2", "elbow", "j4", "j5", "j6"][i])
        ax.grid(True)
        ax.set_xlim(left=0)
        ax.legend(loc='upper right')
    

    ax = axes[6]
    ax.plot(t, traj_target[:, 6],'C0', label="target")
    ax.plot(t, traj_true[:, 6],'C1', label="true")
    ax.set_title("g")
    ax.grid(True)
    ax.set_xlim(left=0)
    ax.legend(loc='upper right')


    mean_qpos_errs = np.mean(qpos_errs, axis=0)
    title = (
        f"mean_j1_err: {mean_qpos_errs[0]:.3g}, "
        f"mean_j2_err: {mean_qpos_errs[1]:.3g}, "
        f"mean_j3_err: {mean_qpos_errs[2]:.3g}, "
        f"mean_j4_err: {mean_qpos_errs[3]:.3g}, "
        f"mean_j5_err: {mean_qpos_errs[4]:.3g}, "
        f"mean_j6_err: {mean_qpos_errs[5]:.3g}"
    )
    plt.suptitle(title)
    
    axes[7].set_visible(False)
    axes[8].set_visible(False)
    plt.tight_layout()    
    plt.savefig(f"{save_fpath}/plot.jpg")