import pickle as pkl
import glob
import os
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
import matplotlib.pyplot as plt
import numpy as np


dirpath = "./logs/pd_keyboard/"

actuators = [
    "l_hip_yaw_joint",
    "l_hip_roll_joint",
    "l_hip_pitch_joint",
    "l_knee_joint",
    "l_ankle_joint",
    "r_hip_yaw_joint",
    "r_hip_roll_joint",
    "r_hip_pitch_joint",
    "r_knee_joint",
    "r_ankle_joint"
]




def generateJointPlots(data_arrs, t_start_idx, data_col_start_idxs, title, ylabels, filename):
    plt.figure(figsize=(48, 6))
    plt.suptitle(title)

    if len(data_arrs) != len(data_col_start_idxs) != len (ylabels):
        raise ValueError("data_arrs, data_col_start_idxs, ylabels must all be the same length")

    for i in range(10):  # 10 joints
        ax1 = plt.subplot(2, 5, i + 1)
        ax1.plot(timesteps[t_start_idx:], data_arrs[0][t_start_idx:, data_col_start_idxs[0] + i],'C0')
        ax1.set_ylabel(ylabels[0])
        ax1.set_title(actuators[i])
        ax1.tick_params(axis='y', labelcolor='C0')
        ax1.grid(True)

        if len(data_arrs) > 1:
            ax2 = ax1.twinx()
            ax2.plot(timesteps[t_start_idx:], data_arrs[1][t_start_idx:, data_col_start_idxs[1] + i], 'C1')
            ax2.set_ylabel(ylabels[1])
            ax2.tick_params(axis='y', labelcolor='C1')

    plt.tight_layout()
    plt.savefig(os.path.join(logfile_folder, f"plots/{filename}.png"))
    plt.close()




logfile_folders = glob.glob(os.path.join(dirpath, "*"))
for logfile_folder in logfile_folders:

    print(f"Generating plots for {logfile_folder}")

    try:
        with open(os.path.join(logfile_folder, "log.pkl"), 'rb') as logfile:
            
            logdata = pkl.load(logfile)
            os.makedirs(os.path.join(logfile_folder, "plots"), exist_ok=True)

            qpos = np.array(logdata.get("qpos"))
            qvel = np.array(logdata.get("qvel"))
            ctrl = np.array(logdata.get("ctrl"))

            simdt = 0.001
            timesteps = np.arange(stop=len(qpos)*simdt, step=simdt)
            # Discard initial 2% of data to avoid transients
            t_start_idx = int(0.02*timesteps.shape[0])            

            # Generate joint plots
            generateJointPlots([ctrl], t_start_idx, data_col_start_idxs=[0], title="joint_ctrl", ylabels=["u(t) [Nm]"], filename="joint_ctrl")
            generateJointPlots([qpos], t_start_idx, data_col_start_idxs=[7], title="joint_pos", ylabels=["pos [rad]"], filename="joint_pos")
            generateJointPlots([qvel], t_start_idx, data_col_start_idxs=[6], title="joint_vel", ylabels=["vel [rad/s]"], filename="joint_vel")
            generateJointPlots([ctrl, qpos], t_start_idx, data_col_start_idxs=[0, 7], title="joint_ctrl_pos", ylabels=["u(t) [Nm]", "pos [rad]"], filename="joint_ctrl_pos")


    except FileNotFoundError:
        continue
