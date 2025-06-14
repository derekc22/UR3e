import os
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from mujoco import MjModel
import mujoco

class UR3eEnv(MujocoEnv):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 20
    }

    def __init__(self, render_mode=None):

        # model_path = os.path.join(os.path.dirname(__file__), "archive/model/ur3e.xml")
        model_path = os.path.abspath("./archive/model/ur3e.xml")

        # Temporary MujocoEnv init to access model parameters
        self._initialize_model(model_path)

        obs_dim = self.model.nq + self.model.nv
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
            low=-1.0,
            high=1.0,
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

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def compute_reward(self, obs, action):
        return -np.linalg.norm(obs)

    def _check_termination(self, obs):
        return False

    def _check_truncation(self):
        return False
