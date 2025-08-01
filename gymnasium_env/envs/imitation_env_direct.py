import os
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
from utils.gym_utils import *
from utils.utils import *
np.set_printoptions(
    linewidth=400,     # Wider output (default is 75)
    threshold=np.inf,  # Print entire array, no summarization with ...
    precision=3,       # Show more decimal places
    suppress=False     # Do not suppress small floating point numbers
)
import os
import numpy as np
from controller.move_l_task import *
from controller.build_traj import *
from controller.controller_utils import get_task_space_state
from controller.aux import plot_plots
from utils.utils import *

render_fps = 500
dt = 0.001

class ImitationEnvDirect(MujocoEnv):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": render_fps
    }    

    def __init__(self, render_mode=None):

        model_path = os.path.abspath("./assets/main.xml")
        self.episode = 0
        self.t = 0
        
        obs_dim = 13  # Same as before: [gripper_pos, block_pos, target_pos, grasp_state, pad_pos]
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float64
        )

        super().__init__(
            model_path=model_path,
            frame_skip=int((1/render_fps)/dt),
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        # Action space: u
        # low = np.array([-330, -330, -150, -54, -54, -54, 0])
        # high = np.array([330, 330, 150, 54, 54, 54, 255])
        ctrl_ranges = get_ctrl_ranges(self.model)
        low = ctrl_ranges[:, 0]
        high = ctrl_ranges[:, 1]
        self.action_space = spaces.Box(
            low=low, 
            high=high, 
            dtype=np.float64
        )
                
        config_path = "controller/config/config_l_mug.yml"
        with open(config_path, "r") as f: yml = yaml.safe_load(f)
        self.pos_gains = { k:np.diag(v) for k, v in yml["pos"].items() } 
        self.rot_gains = { k:np.diag(v) for k, v in yml["rot"].items() } 


        self.site_id = get_site_id(self.model, "tcp")
        self.collision_cache = init_collision_cache(self.model)

    def step(self, action):
        # Convert action to full task space command
        # traj = np.hstack([
        #     action[:3],  # x, y, z
        #     [-1.209, -1.209, 1.209],  # Fixed rotation
        #     action[3]    # grip
        # ])
        
        # u = pid_task_ctrl(
        #     0, self.model, self.data, traj, 
        #     self.pos_gains, self.rot_gains, 
        #     self.pos_errs, self.rot_errs, 
        #     self.tot_pos_errs, self.tot_rot_errs
        # )

        # Step simulation
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        
        # Simple reward for demonstration purposes
        # Supposedly, these rewards are not used by the imitation library during gail or airl
        # They are overridden by the learned reward function. Thus, a placeholder of -1 is used here
        # reward = -np.linalg.norm(observation[3:6] - observation[6:9])  # Distance between mug and target
        reward = -1 
        
        terminated = self._check_termination(observation)
        truncated = self._check_truncation()
        info = {}
        
        self.t += 1

        if self.render_mode == "human":
            self.render()  # Ensure GUI updates
        
        return observation, reward, terminated, truncated, info

    def reset_model(self):
        init_qpos, init_qvel = get_init(self.model, mode="stochastic", keyframe="down")
        self.set_state(init_qpos, init_qvel)
        self.t = 0
        self.episode += 1
        self.pos_errs = np.zeros(shape=(1, 3))
        self.rot_errs = np.zeros(shape=(1, 3))
        self.tot_pos_errs = np.zeros(3)
        self.tot_rot_errs = np.zeros(3)
        # print(f"Episode {self.episode}")
        return self._get_obs()
    
    def _get_obs(self):
        return np.hstack([
            get_2f85_xpos(self.model, self.data),  # 3 dim
            get_mug_xpos(self.model, self.data),    # 3 dim
            get_ghost_xpos(self.model, self.data),  # 3 dim 
            get_block_grasp_state(self.model, self.data),  # 1 dim
            # get_site_xpos(self.model, self.data, "right_pad1_site"),  # 3 dim
            get_site_velp(self.model, self.data, "tcp") # 3 dim
        ])

    def _check_termination(self, observation):
        # gripper_xpos = observation[:3]   # End-effector position
        # mug_xpos = observation[3:6]      # Mug position
        # ghost_xpos = observation[6:9]    # Target position
        
        # place_threshold = 0.005
        # d_place = np.linalg.norm(mug_xpos - ghost_xpos)
        
        # if d_place < place_threshold:
        #     return True
        # if get_self_collision(self.model, self.data, self.collision_cache):
        #     return True
        # if get_mug_toppled(self.model, self.data):
        #     return True
        
        # The imitation library prefers fixed-horizon episodes.
        # Termination signals can leak information about the reward.
        # All episodes should run until the time limit is reached.
        return False

    def _check_truncation(self):
        return self.t >= 1200 #500  # 1000 steps max