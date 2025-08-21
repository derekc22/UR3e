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

class UR3eEnv(MujocoEnv):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": render_fps
    }    

    def __init__(self, render_mode=None):
        
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
        
        self.tot_reward = 0
        

        obs_dim = 13
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
        # mug = [0.29799994, 0.13349916, 0.045]
        # ghost = [0.29799994 0.25 0.045]
        low = np.array([0.28799994, 0.13349916, 0.005, 0])
        high = np.array([0.35799994, 0.35349916, 0.165, 1])
        
        self.action_space = spaces.Box(
            low=low, 
            high=high, 
            dtype=np.float64
        )
        


        # PID gains (tune these as needed)
        self.pos_gains = {
            'kp': np.diag([320, 320, 320]),
            'kd': np.diag([20, 20, 25]),
            # 'ki': np.diag([50, 50, 50])
            'ki': np.diag([0, 0, 0])
        }
        self.rot_gains = {
            'kp': np.diag([325, 325, 325]),
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

        # traj = build_traj_l_point_custom(
        #     get_task_space_state(self.model, self.data),
        #     stop, hold=5
        # )
        
        # print(traj)
        # print(get_2f85_xpos(self.model, self.data))
        # print(action)
        # for traj_i in traj:
        u = pid_task_ctrl(
            0, self.model, self.data, traj, 
            self.pos_gains, self.rot_gains, 
            self.pos_errs, self.rot_errs, 
            self.tot_pos_errs, self.tot_rot_errs
        )
        # self.data.ctrl = u
        # mujoco.mj_step(self.model, self.data)

        self.do_simulation(u, self.frame_skip)
        observation = self._get_obs()
        reward = self.compute_reward(observation, action)
        self.tot_reward+=reward
        # if self.t % 50 == 0:
        # print(self.tot_reward)
        terminated = self._check_termination(observation)
        truncated = self._check_truncation()
        # if (terminated or truncated) and self.episode % save_rate == 0:
            # plot_plots(self.traj_target, self.traj_true, self.ctrls, self.actuator_frc, "gymnasium", log_fpath="logs/gymnasium", pos_errs=self.pos_errs, rot_errs=self.rot_errs)

        info = {}

        # self.ctrls[self.t] = u
        # self.traj_true[self.t] = get_task_space_state(self.model, self.data)
        # self.actuator_frc[self.t] = get_joint_torques(self.data)
        
        self.t += 1
        # print(self.t)

        if self.render_mode == "human":
            self.render()  # Ensure GUI updates
        
        return observation, reward, terminated, truncated, info



    def reset_model(self):
        init_qpos, init_qvel = get_init(self.model, reset_mode="stochastic", keyframe="down", noise_mag="high")
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
        self.tot_reward = 0
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
            # get_robot_qpos(self.data).flat, # 14 dim  [:14]
            # get_robot_qvel(self.data).flat, # 14 dim  [14:28]
            get_2f85_xpos(self.model, self.data), # 3 dim [:3]
            get_mug_xpos(self.model, self.data), # 3 dim [3:6]
            get_ghost_xpos(self.model, self.data), # 3 dim [6:9] 
            get_block_grasp_state(self.model, self.data), # 1 dim [9:]
            get_site_xpos(self.model, self.data, "right_pad1_site")
        ])
    



    def compute_reward(self, observation, action):
        # Unpack observation components
        gripper_pos = observation[:3]      # End-effector position
        block_center = observation[3:6]    # Block center position
        target_pos = observation[6:9]      # Target position
        grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
        pad_pos = observation[10:13]       # Gripper pad position (right pad)
        
        # Block dimensions - IMPORTANT CORRECTION
        block_half_height = get_body_size(self.model, "fish")[-1]
        block_top_z = block_center[2] + block_half_height
        block_bottom_z = block_center[2] - block_half_height
        
        # Key spatial relationships - CORRECTED
        pad_to_block_top = pad_pos[2] - block_top_z
        gripper_to_block_center = gripper_pos[2] - block_center[2]
        horizontal_error = np.linalg.norm(gripper_pos[:2] - block_center[:2])
        
        # ---- GRASP VALIDATION ----
        valid_grasp = (
            grasp_state == 2 and                         
            abs(pad_to_block_top) < 0.04 and             
            horizontal_error < 0.03                      
        )
        
        # # ---- GRASP READINESS ----
        # grasp_readiness = (
        #     np.exp(-80 * horizontal_error**2) * 
        #     np.exp(-265 * pad_to_block_top**2)
        # )
        
        # ---- REWARD COMPONENTS ----
        # 1. SMART DESCENT REWARD (CORRECTED)
        ideal_height_above = 0.5  # 1.5cm above block center
        height_error = gripper_to_block_center - ideal_height_above
        # descent_reward = 15 * np.exp(-100 * height_error**2)
        # descent_reward = 30 * (height_error+1) * np.exp(-height_error)
        z_tol = 0.1
        descent_reward =  1 * ( (1/z_tol) * (height_error+z_tol) * np.exp(-(1/z_tol)*height_error) )
        # print(descent_reward)
        
        # ---- GRASP READINESS ----
        grasp_readiness = (
            np.exp(- horizontal_error**2) * 
            np.exp(- pad_to_block_top**2) *
            np.exp(- height_error**2) *
            100*np.exp(-action[-1]**2)
        )
        
        # 2. Alignment reward
        alignment_reward = 4 * np.exp(-60 * horizontal_error**2)
        
        # 3. Grasp rewards (unchanged)
        grip_strength = action[-1]
        grasp_reward = (
            5.5 * (grasp_state >= 1) +                  
            8.5 * (grasp_state == 2) +                  
            23.5 * grip_strength * grasp_readiness +    
            28.5 * (grasp_state == 2) * grasp_readiness + 
            11.5 * (grasp_state == 2) * grasp_readiness * np.tanh(8 * grip_strength)
        )
        
        # 4. Lifting reward (CORRECTED)
        lift_reward = 12 * (grasp_state == 2) * np.tanh(4 * block_bottom_z)
        
        # 5. Placement rewards
        d_place = np.linalg.norm(block_center - target_pos)
        placement_reward = -2 * d_place + 20 * np.exp(-70 * d_place**2)
        if d_place < 0.05 and valid_grasp:
            placement_reward += 40
        
        # 6. DANGEROUS HEIGHT PENALTY (CORRECTED)
        # Penalize when gripper is below block's center
        # dangerous_height_penalty = -30 * np.exp(-500 * max(0, block_center[2] - gripper_pos[2])**2)
        # height_err_danger = block_center[2] - gripper_pos[2]
        # dangerous_height_penalty = 10 * ( (1/z_tol) * (-height_err_danger+z_tol) * np.exp((1/z_tol)*height_err_danger) )
        dangerous_height_penalty = min(0, -100000000000 * (block_center[2] - gripper_pos[2] + 0.5)**3  )
        # print(grasp_readiness)
        # print(dangerous_height_penalty, -10000 * (block_center[2] - gripper_pos[2] + 0.035)**3) 


        
        # 7. Other penalties
        penalties = (
            -40 * get_self_collision(self.model, self.data, self.collision_cache) +
            -25 * get_table_collision(self.model, self.data, self.collision_cache) +
            -8 * get_mug_toppled(self.model, self.data) +
            -4 * max(0, pad_to_block_top) +
            dangerous_height_penalty
        )
        # print(penalties)
        
        # 8. Action encouragement
        action_reward = 700.5 * grip_strength * grasp_readiness
        
        # 9. Contact achievement bonus
        # contact_achievement_bonus = 1700.5 * (grasp_state >= 1) * grasp_readiness * np.tanh(10 * grip_strength)
        contact_achievement_bonus = 1700.5 * (grasp_state == 2) * grasp_readiness * np.tanh(10 * grip_strength) #* (grasp_state)
        
        # ---- FINAL REWARD ----
        reward = (
            descent_reward +
            alignment_reward +
            grasp_reward +
            lift_reward +
            placement_reward +
            action_reward +
            contact_achievement_bonus +
            penalties
        )
        
        # Debug output
        if self.t % 20 == 0:
            print(f"t: {self.t:4d} | R: {reward:6.2f} | "
                f"Desc: {descent_reward:5.2f} | "
                f"Align: {alignment_reward:5.2f} | "
                f"Grasp: {grasp_reward:5.2f} | "
                # f"Lift: {lift_reward:5.2f} | "
                f"Place: {placement_reward:5.2f} | "
                f"ActR: {action_reward:5.2f} | "
                f"ContactBonus: {contact_achievement_bonus:5.2f} | "
                f"Pen: {penalties:5.2f} | "
                f"DangerHtPen: {dangerous_height_penalty:5.2f} | "
                # f"GripperZ: {gripper_pos[2]:.4f} | "
                # f"BlockCenterZ: {block_center[2]:.4f} | "
                # f"BlockTopZ: {block_top_z:.4f} | "
                f"Readiness: {grasp_readiness:.2f} | "
                f"GState: {grasp_state} | "
                f"Grip: {grip_strength:.2f} | "
                # print(gripper_pos[2], block_center, action[2])
                
                f"GrZ: {gripper_pos[2]:.2f} | "
                f"BZ: {block_center[2]:.2f} | "
                f"PZ: {pad_pos[2]:.2f} | "
                f"BT: {block_top_z:.2f} | "
                f"actionZ: {action[2]:.2f}")

        return reward



    def _check_termination(self, observation):
        """
        Args:
            observation: np.ndarray of length 43
        Returns:
            bool terminated
        """

        gripper_xpos = observation[:3]   # (3,)
        mug_xpos     = observation[3:6]
        ghost_xpos  = observation[6:9]
        grasp_contact       = observation[9:10]
        _, _, pad_z = observation[10:]

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
            print("Reason: Place distance is below threshold (place is successful)")
            return True
        # why the heck would u terminate on these? this is good!
        # if d_pick < pick_threshold:
            # print("Reason: Pick distance is below threshold")
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
        # return self.t >= self.traj_target.shape[0] - 1
        # if self.t >= 500:
        # if self.t >= 5000:
        # if self.t >= 10000:
        if self.t >= 500:
        # if self.t >= 200:
            print("truncated")
            return True
        return False
