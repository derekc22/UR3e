import os
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from mujoco import MjModel
import mujoco
from gymnasium_env.gymnasium_env_utils import *
from controller.move_l_task import *
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
        self.episode = 0

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
            frame_skip=int((1/self.model.opt.timestep)/render_fps),
            observation_space=observation_space,
            # action_space=action_space,
            render_mode=render_mode
        )


        # New action space: [x, y, z, rot_x, rot_y, rot_z, grip]
        # low = np.array([-0.9, -0.9, 0.1, -np.pi, -np.pi, -np.pi, 0])
        # high = np.array([0.9, 0.9, 0.6, np.pi, np.pi, np.pi, 0.255])
        low = np.array([0.2, -0.3, 0.1, -np.pi, -np.pi, -np.pi, 0])
        high = np.array([0.6, 0.3, 0.2, np.pi, np.pi, np.pi, 0.255])
        self.action_space = spaces.Box(
            low=low, 
            high=high, 
            dtype=np.float64
        )
        


        # PID gains (tune these as needed)
        self.pos_gains = {
            'kp': np.diag([120, 120, 120]),
            'kd': np.diag([20, 20, 20]),
            'ki': np.diag([25, 25, 25])
        }
        self.rot_gains = {
            'kp': np.diag([35, 15, 15]),
            'kd': np.diag([2, 2, 2]),
            'ki': np.diag([1, 1, 1])
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
        self.site_id = get_site_id(self.model, "right_pad1_site")
        
        self.collision_cache = init_collision_cache(self.model)
        


    def _initialize_model(self, model_path):
        self.model = MjModel.from_xml_path(model_path)

    def step(self, action):
        
        u = pid_task_ctrl(0, self.model, self.data, action, 
                 self.pos_gains, self.rot_gains, 
                 self.pos_errs, self.rot_errs, 
                 self.tot_pos_errs, self.tot_rot_errs
                )

        self.do_simulation(u, self.frame_skip)
        observation = self._get_obs()
        reward = self.compute_reward(observation, action)
        terminated = self._check_termination(observation)
        truncated = self._check_truncation()
        info = {}

        # Indices based on observation structure
        # gripper_pos = observation[28:31]   # (3,)
        # mug_pos = observation[31:34]       # (3,)
        # ghost_pos = observation[38:41]    # (3,)
        # Distance metrics
        # d_pick = np.linalg.norm(gripper_pos - mug_pos)
        # d_place = np.linalg.norm(mug_pos - ghost_pos)
        # print("d_pick", d_pick)
        # print("d_place", d_place)
        # print("----------------------------")
        
        # Check for self-collision
        if get_robot_collision(self.model, self.data, self.collision_cache):
            reward = -10.0  # or another strong penalty
            terminated = True
            info['termination_reason'] = 'self_collision'
            print("Collision Detected")
            return observation, reward, terminated, truncated, info


        if self.render_mode == "human":
            self.render()  # Ensure GUI updates

        return observation, reward, terminated, truncated, info



    def reset_model(self):
        init_qpos = self.model.keyframe("home").qpos
        init_qvel = self.model.keyframe("home").qvel
        
        noise = np.hstack([
            np.zeros(self.model.nq-7),
            np.random.uniform(low=-0.2, high=0.2, size=2),
            np.zeros(5),
        ])
        # self.set_state(init_qpos, init_qvel)
        self.set_state(init_qpos + noise, init_qvel)

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
            get_grasp_contact(self.data) # 1 dim [41:]
        ])


    


    # def compute_reward(self, observation, action):
    #     """
    #     Reward = - (distance gripper→mug)  - (distance mug→target)
    #             + contact bonus when grasped
    #             + placement bonus when mug is within threshold of target
    #     """

    #     # 1. Positions
    #     gripper_pos = observation[28:31]      # indices 28,29,30
    #     mug_pos = observation[31:34]          # indices 31,32,33
    #     ghost_pos = observation[38:41]       # indices 38,39,40
    #     force = observation[41:43]            # indices 41,42

    #     # 3. Distance metrics
    #     d_pick = np.linalg.norm(gripper_pos - mug_pos)
    #     d_place = np.linalg.norm(mug_pos - ghost_pos)

    #     # 4. Shaped rewards (negative distances)
    #     r_grasp = -d_pick
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
        ghost_pos = observation[38:41]    # (3,)
        grasp_force = observation[41]   # (2,)

        # Distance metrics
        d_pick = np.linalg.norm(gripper_pos - mug_pos)
        d_place = np.linalg.norm(mug_pos - ghost_pos)
        xpos_delta_norm = np.linalg.norm(self.pos_errs)

        # Parameters
        contact_threshold = 0.01
        place_threshold = 0.05

        # Base dense rewards
        r_grasp_dense = -0.1 * d_pick        # Encourage proximity to mug
        # r_grasp_dense = 1/d_pick        # Encourage proximity to mug
        r_place_dense = -0.5 * d_place        # Stronger encouragement for placing

        # Bonuses
        r_contact = 0.0
        # if np.any(grasp_force > contact_threshold):
        if grasp_force > contact_threshold:
            r_contact += 1.0                  # Bonus for grasping

        r_place = 0.0
        if d_place < place_threshold and np.any(grasp_force > contact_threshold):
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
            + time_penalty
            + action_penalty
            + tracking_penalty
        )

        return reward


        


    def _check_termination(self, observation):
        """
        Args:
            observation: np.ndarray of length 43
        Returns:
            bool terminated
        """

        gripper_pos = observation[28:31]   # (3,)
        mug_pos     = observation[31:34]
        ghost_pos  = observation[38:41]
        force       = observation[41]

        pick_threshold = 0.005
        max_arm_reach = 1
        place_threshold   = 0.005
        contact_threshold = 0.01

        d_pick = np.linalg.norm(gripper_pos - mug_pos)
        d_place = np.linalg.norm(mug_pos - ghost_pos)
        grasped = np.any(force > contact_threshold)

        # success if grasped and within threshold
        # if grasped and (d_place < place_threshold):
        if d_place < place_threshold or not (pick_threshold < d_pick < max_arm_reach):
            return True

        return False
    

    def _check_truncation(self):
        return False
