import os
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from mujoco import MjModel
import mujoco
from utils import *

class UR3eEnv(MujocoEnv):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 200
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

        # Call full MujocoEnv constructor
        super().__init__(
            model_path=model_path,
            frame_skip=5,
            observation_space=observation_space,
            # action_space=action_space,
            render_mode=render_mode
        )


        self.action_space = spaces.Box(
            low=-33000,
            high=33000,
            shape=(action_dim,),
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

        if self.render_mode == "human":
            self.render()  # Ensure GUI updates

        return observation, reward, terminated, truncated, info

    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
    #     qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    # def _get_obs(self):
    #     return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def reset_model(self):
        qpos = self.init_qpos #+ self.np_random.uniform(low=-10, high=10, size=self.model.nq)
        qvel = self.init_qvel #+ self.np_random.uniform(low=-10, high=10, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            get_robot_qpos(self.data).flat, # 14 dim
            get_robot_qvel(self.data).flat, # 14 dim
            get_gripper_pos(self.model, self.data), # 3 dim
            get_mug_qpos(self.data), # 7 dim
            get_ghost_pos(self.model), # 3 dim
            get_grasp_force(self.data) # 2 dim
        ])

    # def compute_reward(self, obs, action):
        # return -np.linalg.norm(obs)
    



    def compute_reward(self, observation, action):
        """
        Reward = - (distance gripper→mug)  - (distance mug→target)
                + contact bonus when grasped
                + placement bonus when mug is within threshold of target
        """

        # 1. Positions
        gripper_pos = observation[28:31]      # indices 28,29,30
        mug_pos = observation[31:34]          # indices 31,32,33
        target_pos = observation[38:41]       # indices 38,39,40
        force = observation[41:43]            # indices 41,42

        # 3. Distance metrics
        d_grasp = np.linalg.norm(gripper_pos - mug_pos)
        d_place = np.linalg.norm(mug_pos - target_pos)

        # 4. Shaped rewards (negative distances)
        r_grasp = -d_grasp
        r_place = -d_place

        # 5. Grasp‐contact bonus
        contact_threshold = 0.01       # you may tune this to detect a firm grasp
        contact_bonus = 0.5
        if np.any(force > contact_threshold):
            r_grasp += contact_bonus

        # 6. Placement bonus
        place_threshold = 0.05         # 5 cm tolerance
        place_bonus = 1.0
        if d_place < place_threshold and np.any(force > contact_threshold):
            r_place += place_bonus

        # 7. Combined
        #   You can weight these if you find one phase dominates learning
        reward = r_grasp + r_place

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
        if grasped and (d_place < place_threshold):
            return True

        return False
    

    def _check_truncation(self):
        return False
