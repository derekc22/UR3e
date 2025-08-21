import os
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from mujoco import MjModel
import mujoco
from utils.gymnasium_env_utils import *
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

render_fps = 1
dt = 0.001
save_rate = int(100/render_fps**0.6505)

class UR3eEnv(MujocoEnv):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": render_fps
    }    

    def __init__(self, render_mode=None):

        
        # model_path = os.path.join(os.path.dirname(__file__), "archive/model/ur3e.xml")
        model_path = os.path.abspath("./assets/main.xml")

        # Temporary MujocoEnv init to access model parameters
        # self._initialize_model(model_path)
        self.episode = 0
        self.t = 0
        self.traj_true = None
        self.pos_errs = None
        self.rot_errs = None
        self.ctrls = None
        self.actuator_frc = None
        self.tot_pos_errs = None
        self.tot_rot_errs = None
        

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
            # frame_skip=int((1/render_fps)/self.model.opt.timestep),
            frame_skip=int((1/render_fps)/dt),
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
        # low = np.array([0])
        # high = np.array([1])

        # New action space: [rot_z, grip]
        # five_deg = 10*(np.pi/180)
        # low = np.array([-five_deg, 1])
        # high = np.array([five_deg, ])

        # New action space: [x, y, z, grip]
        low = np.array([-0.01, -0.2, 0.05, 0])
        high = np.array([0.01, 0.2, 0.3, 1])
        
        self.action_space = spaces.Box(
            low=low, 
            high=high, 
            dtype=np.float64
        )
        


        # PID gains (tune these as needed)
        self.pos_gains = {
            'kp': np.diag([120, 120, 120]),
            'kd': np.diag([20, 20, 20]),
            # 'ki': np.diag([50, 50, 50])
            'ki': np.diag([0, 0, 0])
        }
        self.rot_gains = {
            'kp': np.diag([35, 15, 15]),
            'kd': np.diag([2, 2, 2]),
            # 'ki': np.diag([1, 1, 1])
            'ki': np.diag([0, 0, 0])
        }
        # self.pos_gains = {
        #     'kp': np.diag([0.01, 0.01, 0.01]),
        #     'kd': np.diag([20, 20, 20]),
        #     'ki': np.diag([0, 0, 0])
        #     # 'ki': np.diag([0, 0, 0])
        # }
        # self.rot_gains = {
        #     'kp': np.diag([0.01, 0.01, 0.01]),
        #     'kd': np.diag([2, 2, 2]),
        #     'ki': np.diag([0, 0, 0])
        #     # 'ki': np.diag([0, 0, 0])
        # }
        self.site_id = get_site_id(self.model, "tcp")
        self.collision_cache = init_collision_cache(self.model)
        


    # def _initialize_model(self, model_path):
    #     self.model = MjModel.from_xml_path(model_path)

    def step(self, action):
        
        # New action space: [rot_x, rot_y, rot_z, grip]
        # traj = np.hstack([
        #     get_mug_xpos(self.model, self.data), action
        # ])

        # New action space: [grip]
        # traj = np.hstack([
        #     self.traj_target[self.t], action
        # ])
        
        # New action space: [rot_z, grip]
        # traj = np.hstack([
        #     self.traj_target[self.t, :-1], action
        # ])

        # New action space: [x, y, z, grip]
        traj = np.hstack([
            action[:3], [-1.209, -1.209,  1.209], action[-1]
        ])
        # print(get_task_space_state(self.model, self.data))
        # print(traj)
        # exit()
        # traj = build_traj_l_point_custom(
        #     get_task_space_state(self.model, self.data),
        #     stop, hold=1
        # )
        
        print(traj)
        # print(action)
        u = pid_task_ctrl(
            0, self.model, self.data, traj, 
            self.pos_gains, self.rot_gains, 
            self.pos_errs, self.rot_errs, 
            self.tot_pos_errs, self.tot_rot_errs
        )

        self.do_simulation(u, self.frame_skip)
        observation = self._get_obs()
        reward = self.compute_reward(observation, action)
        terminated = self._check_termination(observation)
        truncated = self._check_truncation()
        # if (terminated or truncated) and self.episode % save_rate == 0:
            # plot_plots(self.traj_target, self.traj_true, self.ctrls, self.actuator_frc, "gymnasium", log_fpath="logs/gymnasium", pos_errs=self.pos_errs, rot_errs=self.rot_errs)

        info = {}

        # self.ctrls[self.t] = u
        # self.traj_true[self.t] = get_task_space_state(self.model, self.data)
        # self.actuator_frc[self.t] = get_joint_torques(self.data)
        
        self.t += 1

        if self.render_mode == "human":
            self.render()  # Ensure GUI updates
        
        return observation, reward, terminated, truncated, info



    def reset_model(self):
        init_qpos, init_qvel = get_init(self.model, mode="stochastic", keyframe="down")
        self.set_state(init_qpos, init_qvel)

        # stop = np.hstack([get_mug_xpos(self.model, self.data), [0, 0, 0]])
        # self.traj_target = build_gripless_traj_gym(
        #     get_task_space_state(self.model, self.data)[:-1], 
        #     stop, hold=1)

        # pick = np.hstack([ get_mug_xpos(self.model, self.data), [0, 0, 0], [-1] ])
        # place = np.hstack([ get_ghost_xpos(self.model, self.data), [0, 0, 0], [-1] ])
        # self.traj_target = build_traj_l_pick_place(
        #     get_task_space_state(self.model, self.data), 
        #     [pick, place], hold=1)[:, :-1]

        self.t = 0
        # T = self.traj_target.shape[0]
        # self.traj_true = np.zeros(shape=(T, 7))
        # self.pos_errs = np.zeros(shape=(T, 3))
        # self.rot_errs = np.zeros(shape=(T, 3))
        # self.ctrls = np.zeros(shape=(T, self.model.nu))
        # self.actuator_frc = np.zeros(shape=(T, self.model.nu))
        # self.tot_pos_errs = np.zeros(3)
        # self.tot_rot_errs = np.zeros(3)

        # Not used - only needed so the call to pid_task_ctrl() doesn't throw an error
        self.pos_errs = np.zeros(shape=(1, 3))
        self.rot_errs = np.zeros(shape=(1, 3))
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
            get_block_grasp(self.model, self.data) # 1 dim [41:]
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
        grasp_contact = observation[41]   # (1,)

        # Distance metrics
        d_pick = np.linalg.norm(gripper_xpos - mug_xpos)
        d_place = np.linalg.norm(mug_xpos - ghost_xpos)
        # xpos_delta_norm = np.linalg.norm(self.pos_errs)

        # Parameters
        contact_threshold = 0.01
        place_threshold = 0.05

        # Base dense rewards
        r_grasp_dense = -0.5 * d_pick        # Encourage proximity to mug
        # r_grasp_dense = 1/d_pick        # Encourage proximity to mug
        r_place_dense = -0.5 * d_place        # Stronger encouragement for placing

        # Bonuses
        # r_contact = 0.0
        # if np.any(grasp_contact > contact_threshold):
        # if grasp_contact > contact_threshold:
            # r_contact += 1.0                  # Bonus for grasping
        r_contact = 12.5 * grasp_contact

        r_place = 0.0
        if d_place < place_threshold and np.any(grasp_contact > contact_threshold):
            r_place += 5.0                    # High bonus for correct placement

        # Penalties
        time_penalty = -0.01                 # Mild penalty per step to encourage speed
        action_penalty = -0.001 * np.linalg.norm(action)  # Discourage excessive torque

        # PID Tracking penalty
        # tracking_penalty = -0.5 * xpos_delta_norm  # Penalize position error
        
        table_collision_penalty = -5 * get_table_collision(self.model, self.data, self.collision_cache)

        # Check for self-collision
        self_collision_penalty = -100 * get_self_collision(self.model, self.data, self.collision_cache)

        # Final reward
        reward = (
            r_grasp_dense
            + r_place_dense
            + r_contact
            + r_place
            + table_collision_penalty
            + self_collision_penalty
            # The following may no longer applicable for grip-only action state:
            # + time_penalty
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
        # grasped = np.any(force > contact_threshold)
        
        # print(d_pick, pick_threshold)

        # success if grasped and within threshold
        # if grasped and (d_place < place_threshold):
        # if (d_place < place_threshold or 
        #     d_pick < pick_threshold or 
        #     max_arm_reach < d_pick or
        #     get_self_collision(self.model, self.data, self.collision_cache) or 
        #     get_mug_toppled(self.model, self.data)):            
        #     return True
        if d_place < place_threshold:
            print("Reason: Place distance is below threshold")
            return True
        if d_pick < pick_threshold:
            print("Reason: Pick distance is below threshold")
            return True
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
        # return self.t >= self.traj_target.shape[0] - 1
        if self.t >= 1000:
            print("truncated")
            return True
