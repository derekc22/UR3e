import os
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from mujoco import MjModel
import mujoco
from utils.gymnasium_env_utils import *
from controller.move_l_task import *
from controller.build_traj import build_gripless_traj_mug
from controller.controller_func import get_task_space_state
import time
from utils.utils import *
np.set_printoptions(
    linewidth=400,     # Wider output (default is 75)
    threshold=np.inf,  # Print entire array, no summarization with ...
    precision=3,       # Show more decimal places
    suppress=False     # Do not suppress small floating point numbers
)

render_fps = 200

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
        self.episode = 0
        self.gripless_traj = None
        self.t = None
        

        obs_dim = 42
        # robot qpos  (14) + 
        # robot qvel  (14) + 
        # 2f85 pos    (3 ) + 
        # mug qpos    (7 ) + 
        # ghost pos   (3 ) + 
        # grasp force (1 )
        
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float64
        )

        super().__init__(
            model_path=model_path,
            # frame_skip=(1 if render_fps == 1000 else int((1/render_fps)/self.model.opt.timestep)),
            frame_skip=int((1/render_fps)/self.model.opt.timestep),
            observation_space=observation_space,
            # action_space=action_space,
            render_mode=render_mode
        )


        # New action space: [x, y, z, rot_x, rot_y, rot_z, grip]
        # low = np.array([-0.9, -0.9, 0.1, -np.pi, -np.pi, -np.pi, 0])
        # high = np.array([0.9, 0.9, 0.6, np.pi, np.pi, np.pi, 0.255])
        
        # New action space: [rot_x, rot_y, rot_z, grip]
        # low = np.array([-np.pi, -np.pi, -np.pi, 0])
        # high = np.array([np.pi, np.pi, np.pi,    1])

        # New action space: [grip]
        low = np.array([0])
        high = np.array([1])
        
        self.action_space = spaces.Box(
            low=low, 
            high=high, 
            dtype=np.float64
        )
        


        # PID gains (tune these as needed)
        self.pos_gains = {
            'kp': np.diag([520, 520, 520]),
            'kd': np.diag([20, 20, 20]),
            'ki': np.diag([50, 50, 50])
            # 'ki': np.diag([0, 0, 0])
        }
        self.rot_gains = {
            'kp': np.diag([35, 15, 15]),
            'kd': np.diag([2, 2, 2]),
            'ki': np.diag([1, 1, 1])
            # 'ki': np.diag([0, 0, 0])
        }
        # self.pos_gains = {
        #     'kp': np.diag([100, 100, 100]),
        #     'kd': np.diag([10, 10, 10]),
        #     'ki': np.diag([1, 1, 1])
        # }
        # self.rot_gains = {
        #     'kp': np.diag([100, 100, 100]),
        #     'kd': np.diag([10, 10, 10]),
        #     'ki': np.diag([1, 1, 1])
        # }
        self.pos_errs = np.zeros(shape=(1, 3))
        self.rot_errs = np.zeros(shape=(1, 3))
        self.tot_pos_errs = np.zeros(3)
        self.tot_rot_errs = np.zeros(3)
        self.site_id = get_site_id(self.model, "tcp")
        
        self.collision_cache = init_collision_cache(self.model)
        


    def _initialize_model(self, model_path):
        self.model = MjModel.from_xml_path(model_path)

    def step(self, action):
        
        # New action space: [rot_x, rot_y, rot_z, grip]
        # traj = np.hstack([
        #     get_mug_xpos(self.model, self.data), action
        # ])

        # New action space: [grip]
        traj_i = np.hstack([
            self.gripless_traj[self.t], action
        ])
        u = pid_task_ctrl(0, self.model, self.data, traj_i, 
                 self.pos_gains, self.rot_gains, 
                 self.pos_errs, self.rot_errs, 
                 self.tot_pos_errs, self.tot_rot_errs
                )
        self.t += 1

        self.do_simulation(u, self.frame_skip)
        observation = self._get_obs()
        reward = self.compute_reward(observation, action)
        terminated = self._check_termination(observation)
        truncated = self._check_truncation()
        info = {}
        
        # Check for self-collision
        if get_robot_collision(self.model, self.data, self.collision_cache):
            reward = -10.0  # or another strong penalty
            terminated = True
            info['termination_reason'] = 'self_collision'
            return observation, reward, terminated, truncated, info


        if self.render_mode == "human":
            self.render()  # Ensure GUI updates
        
        return observation, reward, terminated, truncated, info



    def reset_model(self):
        init_qpos_noisy, init_qvel = get_stochastic_init(self.model)
        self.set_state(init_qpos_noisy, init_qvel)

        stop = np.hstack([
            get_mug_xpos(self.model, self.data), [0, 0, 0]
        ])
        self.gripless_traj = build_gripless_traj_mug(
            get_task_space_state(self.model, self.data)[:-1], stop
        )
        # print(get_mug_xpos(self.model, self.data))
        # print(self.gripless_traj[-1, :])
        # exit()
        self.t = 0
        self.tot_pos_errs = np.zeros(3)
        self.tot_rot_errs = np.zeros(3)
        
        print("Episode: ", self.episode)
        self.episode += 1
        
        return self._get_obs()
    

    def _get_obs(self):
        return np.hstack([
            get_robot_qpos(self.data).flat, # 14 dim  [:14]
            get_robot_qvel(self.data).flat, # 14 dim  [14:28]
            get_2f85_xpos(self.model, self.data), # 3 dim [28:31]
            get_mug_qpos(self.data), # 7 dim [31:38]
            get_ghost_xpos(self.model, self.data), # 3 dim [38:41] 
            get_boolean_grasp_contact(self.data) # 1 dim [41:]
        ])


    

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
        gripper_xpos = observation[28:31]   # (3,)
        mug_xpos = observation[31:34]       # (3,)
        mug_qpos = observation[31:38]      # (7,) in case needed later
        ghost_xpos = observation[38:41]    # (3,)
        grasp_contact = observation[41:]   # (2,)

        # Distance metrics
        d_pick = np.linalg.norm(gripper_xpos - mug_xpos)
        d_place = np.linalg.norm(mug_xpos - ghost_xpos)
        xpos_delta_norm = np.linalg.norm(self.pos_errs)

        # Parameters
        contact_threshold = 0.01
        place_threshold = 0.05

        # Base dense rewards
        r_grasp_dense = -0.5 * d_pick        # Encourage proximity to mug
        # r_grasp_dense = 1/d_pick        # Encourage proximity to mug
        r_place_dense = -0.5 * d_place        # Stronger encouragement for placing

        # Bonuses
        r_contact = 0.0
        # if np.any(grasp_contact > contact_threshold):
        # if grasp_contact > contact_threshold:
            # r_contact += 1.0                  # Bonus for grasping
        if grasp_contact:
            print("grasped!")
            r_contact += 100

        r_place = 0.0
        if d_place < place_threshold and np.any(grasp_contact > contact_threshold):
            r_place += 5.0                    # High bonus for correct placement

        # Penalties
        time_penalty = -0.01                 # Mild penalty per step to encourage speed
        action_penalty = -0.001 * np.linalg.norm(action)  # Discourage excessive torque

        # PID Tracking penalty
        tracking_penalty = -0.5 * xpos_delta_norm  # Penalize position error

        # Final reward
        reward = (
            r_grasp_dense
            + r_place_dense
            + r_contact
            + r_place
            # The following may no longer applicable for grip-only action state:
            + time_penalty
            # + action_penalty
            # + tracking_penalty 
        )

        return reward


        


    def _check_termination(self, observation):
        """
        Args:
            observation: np.ndarray of length 43
        Returns:
            bool terminated
        """

        gripper_xpos = observation[28:31]   # (3,)
        mug_xpos     = observation[31:34]
        ghost_xpos  = observation[38:41]
        force       = observation[41]

        pick_threshold = 0.005
        max_arm_reach = 1
        place_threshold   = 0.005
        contact_threshold = 0.01

        d_pick = np.linalg.norm(gripper_xpos - mug_xpos)
        d_place = np.linalg.norm(mug_xpos - ghost_xpos)
        grasped = np.any(force > contact_threshold)
        
        # print(d_pick, pick_threshold)

        # success if grasped and within threshold
        # if grasped and (d_place < place_threshold):
        if d_place < place_threshold or (d_pick < pick_threshold) or (max_arm_reach < d_pick):
            return True

        return False
    

    def _check_truncation(self):
        return self.t >= self.gripless_traj.shape[0]
