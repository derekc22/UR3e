import numpy as np
from utils import R_to_euler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting


def plot_trajectory(traj_target, traj_true, pos_errs, rot_errs, T, dtn):
    
    t_ = np.arange(T)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 8)) # width, height
    axes = axes.flatten()

    # pos plotting
    pos_2f85_target = traj_target[:, :3]
    pos_2f85_traj_true = traj_true[:, :3]
    
    for i in range(3):
        ax = axes[i]
        j = i
        ax.plot(t_, pos_2f85_target[:, j],'C0', label="target")
        ax.plot(t_, pos_2f85_traj_true[:, j],'C1', label="true")
        ax.plot(t_, pos_errs[:, j],'C2', label="err")
        ax.set_title(["x", "y", "z"][j])
        ax.grid(True)
        ax.set_xlim(left=0)
        # if i == 2:
        ax.legend(loc='upper right')
    
    
    # convert codebase to jax backend and use jax.vmap instead
    rot_2f85_target = np.array([R_to_euler(R) for R in traj_target[:, 3:12].reshape(T, 3, 3)])
    rot_2f85_traj_true = np.array([R_to_euler(R) for R in traj_true[:, 3:12].reshape(T, 3, 3)])
    
    for i in range(3, 6):
        ax = axes[i]
        j = i-3
        ax.plot(t_, rot_2f85_target[:, j],'C0', label="target")
        ax.plot(t_, rot_2f85_traj_true[:, j],'C1', label="true")
        ax.plot(t_, rot_errs[:, j],'C2', label="err")
        ax.set_title(["roll", "pitch", "yaw"][j])
        ax.grid(True)
        ax.set_xlim(left=0)
        # if i == 5:
        ax.legend(loc='upper right')
        

    ax = axes[6]
    ax.plot(t_, traj_target[:, 12],'C0', label="target")
    ax.plot(t_, traj_true[:, 12],'C1', label="true")
    ax.set_title("grip")
    ax.grid(True)
    ax.set_xlim(left=0)
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
    plt.savefig(f"mujoco/logs/{dtn}/plot.jpg")
        


        
    
def plot_3d_trajectory(traj_target, traj_true, pos_errs, dtn):
    
    x_target, y_target, z_target = traj_target[:, :3].T
    x_true, y_true, z_true = traj_true[:, :3].T
    
    # Create a figure and 3D axis
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the line
    ax.plot(x_target, y_target, z_target, color='C0', label="target")
    ax.plot(x_true, y_true, z_true, color='C1', label="true")

    # Add labels (optional)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
        
    
    mean_pos_errs = np.mean(pos_errs, axis=0)
    title = (
        f"mean_xpos_err:  {mean_pos_errs[0]:.3g}, "
        f"mean_ypos_err:  {mean_pos_errs[1]:.3g}, "
        f"mean_zpos_err:  {mean_pos_errs[2]:.3g}, "
    )
    plt.suptitle(title)
    
    plt.tight_layout()    
    plt.savefig(f"mujoco/logs/{dtn}/plot3d.jpg")
    





def plot_2d_trajectory(traj_target, traj_true, pos_errs, dtn):
        
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
        f"mean_xpos_err:  {mean_pos_errs[0]:.3g}, "
        f"mean_ypos_err:  {mean_pos_errs[1]:.3g}, "
        f"mean_zpos_err:  {mean_pos_errs[2]:.3g}, "
    )
    plt.suptitle(title)
    
    plt.tight_layout()    
    plt.savefig(f"mujoco/logs/{dtn}/plot2d.jpg")