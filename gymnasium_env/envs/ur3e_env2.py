import os
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from mujoco import MjModel
import mujoco
from utils.gym_utils import *
from controller.move_l_task import *
from controller.build_traj import *
from controller.controller_utils import get_task_space_state
from controller.aux import plot_plots
import time
from utils.utils import *
np.set_printoptions(
    linewidth=400,     # Wider output (default is 75)
    threshold=np.inf,  # Print entire array, no summarization with ...
    precision=3,       # Show more decimal places
    suppress=False     # Do not suppress small floating point numbers
)

render_fps = 500
dt = 0.001
save_rate = int(100/render_fps**0.6505)

class UR3eEnv2(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": render_fps
    }

    def __init__(self, render_mode=None):
        
        model_path = os.path.abspath("./assets/main.xml")
        self.episode = 0
        self.t = 0
        self.traj_true = None
        self.pos_errs = None
        self.rot_errs = None
        self.ctrls = None
        self.actuator_frc = None
        self.tot_pos_errs = None
        self.tot_rot_errs = None
        self.tot_reward = 0

        obs_dim = 24
        observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), dtype=np.float64)

        super().__init__(
            model_path=model_path,
            frame_skip=int((1/render_fps)/dt),
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        x_mug_init, y_mug_init, _ = get_init_mug_xpos(self.model, "down")
        low  = np.array([x_mug_init - 0.25, y_mug_init - 0.25, 0.0, 0])
        high = np.array([x_mug_init + 0.25, y_mug_init + 0.25, 0.5, 1])
        self.action_space = spaces.Box(
            low=low, 
            high=high, 
            dtype=np.float64
        )
                
        with open("controller/config/config_l_mug.yml", "r") as f: yml = yaml.safe_load(f)
        self.pos_gains = {k: np.diag(v) for k, v in yml["pos"].items()}
        self.rot_gains = {k: np.diag(v) for k, v in yml["rot"].items()}
        
        self.collision_cache = init_collision_cache(self.model)

    def step(self, action):
        traj = np.hstack(
            [action[:3], [-1.209, -1.209, 1.209], action[3]
        ])
        u = pid_task_ctrl(
            0, self.model, self.data, traj, 
            self.pos_gains, self.rot_gains, 
            self.pos_errs, self.rot_errs, 
            self.tot_pos_errs, self.tot_rot_errs
        )
        
        self.do_simulation(u, self.frame_skip)
        observation = self._get_obs()
        reward = self.compute_dense_reward(observation)
        self.tot_reward += reward
        
        self.t += 1
        
        terminated = self._check_termination(observation)
        truncated = self._check_truncation()
        if np.linalg.norm(observation[3:6] - observation[6:9]) < 0.05:
            terminated = True
            reward += 50.0  # Success bonus
        
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, {}

    def reset_model(self):
        init_qpos, init_qvel = get_init(self.model, reset_mode="stochastic", keyframe="down", noise_mag="high")
        self.set_state(init_qpos, init_qvel)
        self.t = 0
        self.pos_errs = np.zeros((1, 3))
        self.rot_errs = np.zeros((1, 3))
        self.tot_pos_errs = np.zeros(3)
        self.tot_rot_errs = np.zeros(3)
        return self._get_obs()
    
    def _get_obs(self):
        return np.hstack([
            get_2f85_xpos(self.model, self.data), # 3 dim
            get_mug_xpos(self.model, self.data),  # 3 dim
            get_ghost_xpos(self.model, self.data), # 3 dim
            get_2f85_to_mug_rel_xpos(self.model, self.data), # 3 dim
            get_mug_to_ghost_rel_xpos(self.model, self.data), # 3 dim
            get_2f85_xvelp(self.model, self.data), # 3 dim
            get_2f85_to_mug_rel_xvelp(self.model, self.data), # 3 dim
            get_finger_jnt_disp(self.data), # 1 dim
            get_finger_jnt_vel(self.data), # 1 dim
            get_robust_block_grasp_state(self.model, self.data), # 1 dim
        ])
    
    def compute_dense_reward(self, observation: np.ndarray) -> float:
        gripper_to_mug_dist_vec = observation[9:12]
        mug_to_target_dist_vec = observation[12:15]
        is_grasped = observation[23]

        reach_distance = np.linalg.norm(gripper_to_mug_dist_vec)
        reach_reward = -1.0 * reach_distance
        
        place_distance = np.linalg.norm(mug_to_target_dist_vec)
        place_reward = -2.0 * place_distance # Weighted more heavily than reaching

        grasp_bonus = 0.0
        if is_grasped:
            reach_reward *= 0.1
            grasp_bonus = 2.5
        
        reward = reach_reward + place_reward + grasp_bonus

        if is_grasped and place_distance < 0.05:  # Using a 5cm success threshold
            reward += 50.0
            return reach_reward + place_reward + grasp_bonus
        
        return reward

    def _check_termination(self, observation):
        gripper_xpos = observation[:3]   # (3,)
        mug_xpos     = observation[3:6]
        ghost_xpos  = observation[6:9]

        max_arm_reach = 1
        place_threshold   = 0.005

        d_pick = np.linalg.norm(gripper_xpos - mug_xpos)
        d_place = np.linalg.norm(mug_xpos - ghost_xpos)

        # if d_place < place_threshold:
            # print("Reason: Place distance is below threshold (place is successful)")
            # return True
        if max_arm_reach < d_pick:
            print("Reason: Pick distance exceeds maximum arm reach")
            return True
        if get_self_collision(self.model, self.data, self.collision_cache):
            print("Reason: Self-collision detected")
            return True
        if get_mug_toppled(self.model, self.data):
            print("Reason: Mug is toppled")
            return True

        return False
    

    def _check_truncation(self):
        if self.t >= 2500:
            print("Truncated")
            return True
        return False