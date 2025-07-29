import numpy as np
import yaml
import os
import mujoco
from imitation_env.envs import ImitationEnv
from imitation.data.types import Trajectory
import matplotlib
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf)
matplotlib.use('Agg')  # Set backend to non-interactive
from controller.controller_utils import get_task_space_state, pid_task_ctrl
from utils import *
from controller.build_traj import build_traj_l_pick_move_place, build_traj_l_pick_place_RL, build_traj_l_pick_place
from controller.aux import load_trajectory, cleanup
from gymnasium_env.gymnasium_env_utils import *
from utils import load_model, get_joint_torques, get_jnt_ranges
import pickle as pkl
import time

visualize = False

def collect_expert_demonstrations(num_demos=10):
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
        print(f"Collecting demonstration {demo+1}/{num_demos}")
        m, d = load_model(model_path)
        reset_with_mug(m, d, mode="stochastic", keyframe="down")

        if visualize:
            viewer = mujoco.viewer.launch_passive(m, d)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    
        init_r = get_site_xrotvec(m, d, "tcp")
        pick = np.hstack([get_mug_xpos(m, d), init_r, 0.5])
        place = np.hstack([get_ghost_xpos(m, d), init_r, 1])
        
        # Build trajectory in memory
        traj_target = build_traj_l_pick_place(get_task_space_state(m, d), [pick, place], hold)
        T = traj_target.shape[0]
        
        # Initialize PID error accumulators
        tot_pos_errs = np.zeros(shape=(3, ))
        tot_rot_errs = np.zeros(shape=(3, ))
        
        # Get initial state
        obs = []
        obs.append(np.hstack([
            get_2f85_xpos(m, d),
            get_mug_xpos(m, d),
            get_ghost_xpos(m, d),
            get_block_grasp(m, d),
            get_site_xpos(m, d, "right_pad1_site")
        ]))
        
        acts = []
        infos = [{}] * T  # Dummy infos
        
        for t in range(T):
            if visualize:
                viewer.sync()

            # Expert action (PID output)
            pos_errs = np.zeros(shape=(T, 3))
            rot_errs = np.zeros(shape=(T, 3))
            
            if t % hold == 0: 
                tot_pos_errs.fill(0)
                tot_rot_errs.fill(0)
            

            action = pid_task_ctrl(
                t, m, d, traj_target[t], 
                pos_gains, rot_gains,
                pos_errs, rot_errs,
                tot_pos_errs, tot_rot_errs
            )
            
            # Convert to env action space [x, y, z, grip]
            env_action = np.array([
                traj_target[t, 0],  # x
                traj_target[t, 1],  # y
                traj_target[t, 2],  # z
                action[-1]          # grip
            ])
            acts.append(env_action)
            
            # Step simulation
            d.ctrl = action
            mujoco.mj_step(m, d)
            
            # Get next state
            next_obs = np.hstack([
                get_2f85_xpos(m, d),
                get_mug_xpos(m, d),
                get_ghost_xpos(m, d),
                get_block_grasp(m, d),
                get_site_xpos(m, d, "right_pad1_site")
            ])
            obs.append(next_obs)
        
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

# def save_demonstrations(trajectories, save_path = "imitation_env/data/expert_demos.pkl"):
#     """Save demonstrations to a file using pickle."""
#     # Ensure the directory exists
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     # Save the trajectories
#     with open(save_path, "wb") as f:
#         pkl.dump(trajectories, f)
    
#     print(f"Saved {len(trajectories)} demonstrations to {save_path}")
    

def save_demonstrations(trajectories, save_path="imitation_env/data/expert_demos.pkl"):
    """Append demonstrations to a pickle file by loading, extending, and overwriting."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load existing data if file exists
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            existing_data = pkl.load(f)
    else:
        existing_data = []

    # Extend and overwrite
    existing_data.extend(trajectories)

    with open(save_path, "wb") as f:
        pkl.dump(existing_data, f)

    print(f"Saved {len(trajectories)} new demonstrations. Total is now {len(existing_data)}.")



def load_demonstrations(load_path = "imitation_env/data/expert_demos.pkl"):
    """Load expert trajectories from a file using pickle."""
    with open(load_path, "rb") as f:
        trajectories = pkl.load(f)
    print(f"Loaded {len(trajectories)} demonstrations from {load_path}")
    return trajectories


if __name__ == "__main__":
    # Collect and save demonstrations
    expert_trajs = collect_expert_demonstrations(num_demos=100)
    save_demonstrations(expert_trajs)