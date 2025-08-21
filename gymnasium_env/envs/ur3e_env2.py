import os
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from utils.gym_utils import *
from controller.move_l_task import *
from controller.build_traj import *
from controller.controller_func import get_task_space_state
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
        # reward = self.compute_reward(observation)
        reward = self.compute_reward(observation, action)
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
    
    # def compute_dense_reward(self, observation: np.ndarray) -> float:
    #     gripper_to_mug_dist_vec = observation[9:12]
    #     mug_to_target_dist_vec = observation[12:15]
    #     grasp_state = observation[23]

    #     reach_distance = np.linalg.norm(gripper_to_mug_dist_vec)
    #     reach_reward = -1.0 * reach_distance
        
    #     place_distance = np.linalg.norm(mug_to_target_dist_vec)
    #     place_reward = -2.0 * place_distance # Weighted more heavily than reaching

    #     grasp_bonus = 0.0
    #     if grasp_state:
    #         reach_reward *= 0.1
    #         grasp_bonus = 2.5
        
    #     reward = reach_reward + place_reward + grasp_bonus

    #     if grasp_state and place_distance < 0.05:  # Using a 5cm success threshold
    #         reward += 50.0
    #         return reach_reward + place_reward + grasp_bonus
        
    #     return reward
    

    def compute_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        """
        A comprehensive, shaped reward function with values scaled for stable and
        robust learning.
        """
        # --- 1. Unpack Observation Vector ---
        mug_abs_z = observation[5]
        gripper_to_mug_vec = observation[9:12]
        mug_to_target_vec = observation[12:15]
        gripper_vel = observation[15:18]
        is_grasped = observation[23]
        grip_action = action[-1]

        # --- 2. Define Key State Variables ---
        xy_dist_to_mug = np.linalg.norm(gripper_to_mug_vec[:2])
        # Ideal grasp is slightly above the handle. This error is 0 at the ideal height.
        z_error = abs(gripper_to_mug_vec[2] - 0.02)
        place_dist = np.linalg.norm(mug_to_target_vec)

        # --- 3. Grasp Readiness Gate ---
        # This composite term is high only when the gripper is perfectly positioned.
        # The `k` values in the exponentials control how "sharp" the peak is.
        # `exp(-k*x)` is bounded between 0 and 1; it does not "blow up".
        grasp_readiness = np.exp(-10 * xy_dist_to_mug) * np.exp(-20 * z_error)

        # --- 4. Reward Component Breakdown ---

        # ALIGNMENT: Small, continuous reward for being in the "ready" position.
        # Scaled to 2.0 to act as a gentle guide.
        align_reward = 2.0 * grasp_readiness

        # GRASP ACTION: A small incentive to attempt a grasp, but only when ready.
        # Scaled to 2.0 to subtly encourage the right action at the right time.
        grasp_action_reward = 2.0 * grip_action * grasp_readiness

        # GRASP ACHIEVEMENT (MILESTONE): A significant bonus for a successful grasp.
        # Scaled to 10.0, making it a major milestone without overshadowing the final goal.
        grasp_achieve_reward = 10.0 * is_grasped * grasp_readiness

        # LIFTING (MILESTONE): Another key milestone for lifting the grasped object.
        # The tanh function smoothly saturates the reward between 0 and 8.0.
        # Assumes `self.initial_mug_z` is stored at env.reset().
        lift_reward = 8.0 * is_grasped * np.tanh(8.0 * max(0, mug_abs_z))

        # PLACEMENT: Guides the mug to the target after it's been lifted.
        # This reward is a combination of a long-range linear pull (-dist) and a
        # short-range exponential bonus for precision. It's only active when grasped.
        placement_reward = is_grasped * (4.0 * np.exp(-15 * place_dist) - 1.5 * place_dist)

        # SUCCESS (ULTIMATE GOAL): A large, definitive bonus for completing the task.
        # Scaled to 50.0, making it the clear objective for the agent to maximize.
        success_bonus = 0.0
        if is_grasped and place_dist < 0.05: # 5cm success threshold
            success_bonus = 50.0

        # --- 5. Penalties ---
        # These are hard constraints the agent must learn to respect.
        penalties = 0.0
        # Scaled to -25.0 to be a very strong deterrent.
        # if get_self_collision(...) or get_table_collision(...):
        #     penalties += -25.0
        # if get_mug_toppled(...):
        #     penalties += -25.0
        # Small penalty to discourage being too low or moving too fast.
        penalties += -1.0 * max(0, -gripper_to_mug_vec[2]) # Penalize being below the handle
        penalties += -0.01 * np.linalg.norm(gripper_vel)

        # --- 6. Final Reward Calculation ---
        total_reward = (
            align_reward +
            grasp_action_reward +
            grasp_achieve_reward +
            lift_reward +
            placement_reward +
            success_bonus +
            penalties
        )

        return total_reward

    def _check_termination(self, observation):
        gripper_xpos = observation[:3]   # (3,)
        mug_xpos     = observation[3:6]
        ghost_xpos  = observation[6:9]

        max_arm_reach = 1
        # place_threshold   = 0.005

        d_pick = np.linalg.norm(gripper_xpos - mug_xpos)
        # d_place = np.linalg.norm(mug_xpos - ghost_xpos)

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