import os
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from mujoco import MjModel
import mujoco
from gymnasium_env.gymnasium_env_utils import *
from utils import *
np.set_printoptions(
    linewidth=400,     # Wider output (default is 75)
    threshold=np.inf,  # Print entire array, no summarization with ...
    precision=3,       # Show more decimal places
    suppress=False     # Do not suppress small floating point numbers
)

render_fps = 1

class UR3eEnv(MujocoEnv):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": render_fps
    }    

    def __init__(self, render_mode=None):

        
        # model_path = os.path.join(os.path.dirname(__file__), "archive/model/ur3e.xml")
        model_path = os.path.abspath("./assets/main.xml")

        # Temporary MujocoEnv init to access model parameters
        self._initialize_model(model_path)

        obs_dim = 43 # robot qpos (14) + robot qvel (14) + 2f85 pos (3) + mug qpos (7) + ghost pos (3) + grasp force (2)
        action_dim = self.model.nu
        
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float64
        )

        super().__init__(
            model_path=model_path,
            frame_skip=int((1/self.model.opt.timestep)/render_fps),
            observation_space=observation_space,
            # action_space=action_space,
            render_mode=render_mode
        )


        ctrl_range = self.model.actuator_ctrlrange  # shape: (action_dim, 2)
        min_ctrl = ctrl_range[:, 0]
        max_ctrl = ctrl_range[:, 1]
        # print(min_ctrl)
        # print(max_ctrl)
        self.action_space = spaces.Box(
            low=min_ctrl,
            high=max_ctrl,
            dtype=np.float64
        )

    def _initialize_model(self, model_path):
        self.model = MjModel.from_xml_path(model_path)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        reward = self.compute_reward(observation, action)
        terminated = self._check_termination(observation)
        truncated = self._check_truncation()
        info = {}

        # Indices based on observation structure
        # gripper_pos = observation[28:31]   # (3,)
        # mug_pos = observation[31:34]       # (3,)
        # target_pos = observation[38:41]    # (3,)
        # Distance metrics
        # d_grasp = np.linalg.norm(gripper_pos - mug_pos)
        # d_place = np.linalg.norm(mug_pos - target_pos)
        # print("d_grasp", d_grasp)
        # print("d_place", d_place)
        # print("----------------------------")


        if self.render_mode == "human":
            self.render()  # Ensure GUI updates

        return observation, reward, terminated, truncated, info



    def reset_model(self):
        init_qpos = self.model.keyframe("home").qpos
        init_qvel = self.model.keyframe("home").qvel
        self.set_state(init_qpos, init_qvel)
        return self._get_obs()
    

    def _get_obs(self):
        return np.concatenate([
            get_robot_qpos(self.data).flat, # 14 dim  [:14]
            get_robot_qvel(self.data).flat, # 14 dim  [14:28]
            get_2f85_xpos(self.model, self.data), # 3 dim [28:31]
            get_mug_qpos(self.data), # 7 dim [31:38]
            get_ghost_xpos(self.model, self.data), # 3 dim [38:41] 
            get_grasp_force(self.data) # 2 dim [41:]
        ])

    # def compute_reward(self, obs, action):
        # return -np.linalg.norm(obs)
    



    # def compute_reward(self, observation, action):
    #     """
    #     Reward = - (distance gripper→mug)  - (distance mug→target)
    #             + contact bonus when grasped
    #             + placement bonus when mug is within threshold of target
    #     """

    #     # 1. Positions
    #     gripper_pos = observation[28:31]      # indices 28,29,30
    #     mug_pos = observation[31:34]          # indices 31,32,33
    #     target_pos = observation[38:41]       # indices 38,39,40
    #     force = observation[41:43]            # indices 41,42

    #     # 3. Distance metrics
    #     d_grasp = np.linalg.norm(gripper_pos - mug_pos)
    #     d_place = np.linalg.norm(mug_pos - target_pos)

    #     # 4. Shaped rewards (negative distances)
    #     r_grasp = -d_grasp
    #     r_place = -d_place

    #     # 5. Grasp‐contact bonus
    #     contact_threshold = 0.01       # you may tune this to detect a firm grasp
    #     contact_bonus = 0.5
    #     if np.any(force > contact_threshold):
    #         r_grasp += contact_bonus

    #     # 6. Placement bonus
    #     place_threshold = 0.05         # 5 cm tolerance
    #     place_bonus = 1.0
    #     if d_place < place_threshold and np.any(force > contact_threshold):
    #         r_place += place_bonus

    #     # 7. Combined
    #     #   You can weight these if you find one phase dominates learning
    #     reward = r_grasp + r_place

    #     return reward
    

    def compute_reward(self, observation, action):
        """
        Reward = 
            + progress bonus (distance to mug and target)
            + contact bonus when grasped
            + placement bonus when placed
            - time penalty (to discourage dithering)
            - action penalty (to encourage minimal control)
        """

        # Indices based on observation structure
        gripper_pos = observation[28:31]   # (3,)
        mug_pos = observation[31:34]       # (3,)
        mug_qpos = observation[31:38]      # (7,) in case needed later
        target_pos = observation[38:41]    # (3,)
        grasp_force = observation[41:43]   # (2,)

        # Distance metrics
        d_grasp = np.linalg.norm(gripper_pos - mug_pos)
        d_place = np.linalg.norm(mug_pos - target_pos)

        # Parameters
        contact_threshold = 0.01
        place_threshold = 0.05

        # Base dense rewards
        # r_grasp_dense = -0.1 * d_grasp        # Encourage proximity to mug
        r_grasp_dense = 1/d_grasp        # Encourage proximity to mug
        r_place_dense = -0.5 * d_place        # Stronger encouragement for placing

        # Bonuses
        r_contact = 0.0
        if np.any(grasp_force > contact_threshold):
            r_contact += 1.0                  # Bonus for grasping

        r_place = 0.0
        if d_place < place_threshold and np.any(grasp_force > contact_threshold):
            r_place += 5.0                    # High bonus for correct placement

        # Penalties
        time_penalty = -0.01                 # Mild penalty per step to encourage speed
        action_penalty = -0.001 * np.linalg.norm(action)  # Discourage excessive torque

        # Final reward
        reward = (
            r_grasp_dense
            + r_place_dense
            + r_contact
            + r_place
            + time_penalty
            + action_penalty
        )

        return reward


        


    def _check_termination(self, observation):
        """
        Args:
            observation: np.ndarray of length 43
        Returns:
            bool terminated
        """

        mug_pos     = observation[31:34]
        target_pos  = observation[38:41]
        force       = observation[41:43]

        place_threshold   = 0.05
        contact_threshold = 0.01

        d_place = np.linalg.norm(mug_pos - target_pos)
        grasped = np.any(force > contact_threshold)

        # success if grasped and within threshold
        # if grasped and (d_place < place_threshold):
        if d_place < place_threshold:
            print("yo")
            return True

        return False
    

    def _check_truncation(self):
        return False
