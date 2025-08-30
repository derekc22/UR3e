import numpy as np
import yaml
import os
import mujoco
from imitation.data.types import Trajectory
import matplotlib
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf)
matplotlib.use('Agg')  # Set backend to non-interactive
from controller.controller_func import get_task_space_state, pid_task_ctrl
from utils.utils import *
from controller.build_traj import build_traj_l_pick_move_place, build_traj_l_pick_place_imitation, build_traj_l_pick_place, build_traj_l_pick_place_imitation_augmented
from utils.gym_utils import *
from utils.utils import load_model, get_joint_torques, get_jnt_ranges
import pickle as pkl
import time


from collections import deque
from imitation.data.types import Trajectory

def stack_expert_trajectories(trajectories, history_len):
    """
    Manually applies frame stacking to a list of expert trajectories
    using a deque.

    :param trajectories: A list of trajectories with unstacked observations.
    :param history_len: The number of frames to stack.
    :return: A new list of trajectories with stacked observations.
    """
    new_trajectories = []
    for traj in trajectories:
        stacked_obs = []
        # Initialize a deque with a fixed length to hold the history
        history = deque(maxlen=history_len)

        for i, obs in enumerate(traj.obs):
            if i == 0:
                # For the very first frame, pad the history by repeating it
                for _ in range(history_len):
                    history.append(obs)
            else:
                # For all subsequent frames, just add the new one
                # The deque automatically discards the oldest
                history.append(obs)
            
            # The current state of the deque is the stacked observation
            stacked_obs.append(np.array(history))

        # Create a new trajectory with the processed observations
        new_traj = Trajectory(
            obs=np.array(stacked_obs),
            acts=traj.acts,
            infos=traj.infos,
            terminal=traj.terminal,
        )
        new_trajectories.append(new_traj)
        
    return new_trajectories


def get_obs(m, d):
    # return np.hstack([
    #     get_2f85_xpos(m, d), # 3 dim
    #     get_mug_xpos(m, d),  # 3 dim
    #     get_ghost_xpos(m, d), # 3 dim
    #     get_robust_block_grasp_state(m, d), # 1 dim
    #     get_2f85_xvelp(m, d), # 3 dim
    #     get_ur3e_qpos(d), # 6 dim
    #     get_ur3e_qvel(d), # 6 dim
    # ])
    
    return np.hstack([
        get_2f85_xpos(m, d), # 3 dim
        get_mug_xpos(m, d),  # 3 dim
        get_ghost_xpos(m, d), # 3 dim
        get_2f85_to_mug_rel_xpos(m, d), # 3 dim
        get_mug_to_ghost_rel_xpos(m, d), # 3 dim
        get_2f85_xvelp(m, d), # 3 dim
        get_2f85_to_mug_rel_xvelp(m, d), # 3 dim
        get_finger_jnt_disp(d), # 1 dim
        get_finger_jnt_vel(d), # 1 dim
        get_robust_block_grasp_state(m, d), # 1 dim
    ])
    

def collect_expert_demonstrations(num_demos):
    """Collect expert demonstrations using the PID controller"""
    model_path = "assets/main.xml"
    config_path = "controller/config/config_l_mug.yml"
    
    with open(config_path, "r") as f:
        yml = yaml.safe_load(f)
    pos_gains = {k: np.diag(v) for k, v in yml["pos"].items()}
    rot_gains = {k: np.diag(v) for k, v in yml["rot"].items()}
    hold = yml["hold"]
    
    trajectories = []

    for demo in range(num_demos):
        print(f"Collecting {action_mode} demonstration {demo+1}/{num_demos}")
        m, d = load_model(model_path)
        reset_with_mug(m, d, reset_mode=reset_mode, keyframe="down", noise_mag=noise_mag)

        if visualize:
            viewer = mujoco.viewer.launch_passive(m, d)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    
        init_r = get_site_xrotvec(m, d, "tcp")
        place = np.hstack([get_ghost_xpos(m, d), init_r, 1])
        
        # Build trajectory in memory
        pick = np.hstack([get_mug_xpos(m, d), init_r, 0.0])
        traj_target = build_traj_l_pick_place_imitation_augmented(get_task_space_state(m, d), [pick, place], hold)
        
        # pick = np.hstack([get_mug_xpos(m, d), init_r, 0.0])
        # traj_target = build_traj_l_pick_place_imitation(get_task_space_state(m, d), [pick, place], hold)
        
        # Build trajectory in memory
        # pick = np.hstack([get_mug_xpos(m, d), init_r, 0.5])
        # traj_target = build_traj_l_pick_place(get_task_space_state(m, d), [pick, place], hold)
        
        T = traj_target.shape[0]
        
        # Expert action (PID output)
        pos_errs = np.zeros(shape=(T, 3))
        rot_errs = np.zeros(shape=(T, 3))
        
        # Initialize PID error accumulators
        tot_pos_errs = np.zeros(shape=(3, ))
        tot_rot_errs = np.zeros(shape=(3, ))
        
        # Get initial state
        obs = []
        obs.append(get_obs(m, d))
        
        acts = []
        infos = []

        for t in range(T):
            if visualize:
                viewer.sync()

            if t % hold == 0: 
                tot_pos_errs.fill(0)
                tot_rot_errs.fill(0)
            
            u = pid_task_ctrl(t, m, d, traj_target[t, :], pos_gains, rot_gains, pos_errs, rot_errs, tot_pos_errs, tot_rot_errs)
            
            if action_mode == "indirect":
                # Convert to env action space [x, y, z, grip]
                env_action = np.array([
                    traj_target[t, 0],  # x
                    traj_target[t, 1],  # y
                    traj_target[t, 2],  # z
                    u[-1]               # grip
                ])
            elif action_mode == "direct":
                env_action = u
                
            # Step simulation
            d.ctrl = u
            mujoco.mj_step(m, d)
            # time.sleep(0.001)
            
            if t % down_sample == 0:
                # Append action
                acts.append(env_action)
                
                # Get and append next state
                next_obs = get_obs(m, d)
                obs.append(next_obs)
                
                # Append info
                infos.append({})
        
        # Create trajectory object
        trajectory = Trajectory(
            obs=np.array(obs),
            acts=np.array(acts),
            infos=np.array(infos),
            terminal=True
        )
        trajectories.append(trajectory)
        
        if visualize:
            viewer.close()
            time.sleep(1)
        
    return trajectories
    

def save_demos(trajectories, save_path):
    """Append demonstrations to a pickle file by loading, extending, and overwriting."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load existing data if file exists
    if os.path.exists(save_path) and resume_collecting:
        with open(save_path, "rb") as f:
            existing_data = pkl.load(f)
    else:
        existing_data = []

    # Extend and overwrite
    existing_data.extend(trajectories)

    with open(save_path, "wb") as f:
        pkl.dump(existing_data, f)

    print(f"Saved {len(trajectories)} new demonstrations with {len(trajectories[-1])} data points each. Total is now {len(existing_data)}.")



def load_demos(load_fpath):
    """Load expert trajectories from a file using pickle."""
    with open(load_fpath, "rb") as f:
        trajectories = pkl.load(f)
    print(f"Loaded {len(trajectories)} demonstrations from {load_fpath}")
    return trajectories


if __name__ == "__main__":
    with open("gymnasium_src/config/settings.yml", "r") as f:  yml = yaml.safe_load(f)
    settings = yml["collect_demos.py"]    
    action_mode = settings["action_mode"]
    resume_collecting = settings["resume_collecting"]
    num_demos = settings["num_demos"]
    visualize = settings["visualize"]
    reset_mode = settings["reset_mode"]
    noise_mag = settings["noise_mag"]
    down_sample = settings["down_sample"]

    # Collect and save demonstrations
    demos_fpath = f"gymnasium_src/demos/expert_demos_{action_mode}.pkl"
    expert_trajs = collect_expert_demonstrations(num_demos)
    save_demos(expert_trajs, demos_fpath)