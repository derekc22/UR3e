
    # def compute_reward(self, observation, action):
    #     # Unpack observation components
    #     gripper_pos = observation[:3]      # End-effector position
    #     mug_pos = observation[3:6]         # Block position
    #     target_pos = observation[6:9]      # Target position
    #     grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
    #     pad_pos = observation[10:13]       # Gripper pad position (right pad)
        
    #     # Geometric properties
    #     block_height = get_body_size(self.model, "fish")[-1]
    #     block_top_z = mug_pos[2] + block_height
    #     block_center_z = mug_pos[2]
        
    #     # Key spatial relationships
    #     pad_to_block_top = pad_pos[2] - block_top_z
    #     gripper_to_block_center = gripper_pos[2] - block_center_z
    #     horizontal_error = np.linalg.norm(gripper_pos[:2] - mug_pos[:2])
        
    #     # ---- GRASP VALIDATION ----
    #     valid_grasp = (
    #         grasp_state == 2 and                         
    #         abs(pad_to_block_top) < 0.04 and             
    #         horizontal_error < 0.03                      
    #     )
        
    #     # ---- GRASP READINESS ----
    #     grasp_readiness = (
    #         np.exp(-80 * horizontal_error**2) * 
    #         np.exp(-80 * pad_to_block_top**2)
    #     )
        
    #     # ---- REWARD COMPONENTS ----
    #     # 1. SMART DESCENT REWARD (FIXED)
    #     # Reward peaks when gripper is 1-2cm above block center
    #     ideal_height_above = 0.015  # 1.5cm above block center
    #     height_error = abs(gripper_to_block_center - ideal_height_above)
    #     descent_reward = 15 * np.exp(-100 * height_error**2)
        
    #     # 2. Alignment reward
    #     alignment_reward = 4 * np.exp(-60 * horizontal_error**2)
        
    #     # 3. Grasp rewards (unchanged)
    #     grip_strength = action[-1]
    #     grasp_reward = (
    #         5.5 * (grasp_state >= 1) +                  
    #         8.5 * (grasp_state == 2) +                  
    #         23.5 * grip_strength * grasp_readiness +    
    #         28.5 * (grasp_state == 2) * grasp_readiness + 
    #         11.5 * (grasp_state == 2) * grasp_readiness * np.tanh(8 * grip_strength)
    #     )
        
    #     # 4. Lifting reward
    #     lift_reward = 12 * (grasp_state == 2) * np.tanh(4 * mug_pos[2])
        
    #     # 5. Placement rewards
    #     d_place = np.linalg.norm(mug_pos - target_pos)
    #     placement_reward = -2 * d_place + 20 * np.exp(-70 * d_place**2)
    #     if d_place < 0.05 and valid_grasp:
    #         placement_reward += 40
        
    #     # 6. NEW: DANGEROUS HEIGHT PENALTY
    #     # Severely penalize very low heights that cause collisions
    #     dangerous_height_penalty = -30 * np.exp(-500 * max(0, gripper_pos[2] - 0.03)**2)
        
    #     # 7. Other penalties
    #     penalties = (
    #         -40 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -25 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -8 * get_mug_toppled(self.model, self.data) +
    #         -4 * max(0, pad_to_block_top) +
    #         dangerous_height_penalty  # NEW PENALTY
    #     )
        
    #     # 8. Action encouragement
    #     action_reward = 7.5 * grip_strength * grasp_readiness
        
    #     # 9. Contact achievement bonus
    #     contact_achievement_bonus = 17.5 * (grasp_state >= 1) * grasp_readiness * np.tanh(10 * grip_strength)
        
    #     # ---- FINAL REWARD ----
    #     reward = (
    #         descent_reward +
    #         alignment_reward +
    #         grasp_reward +
    #         lift_reward +
    #         placement_reward +
    #         action_reward +
    #         contact_achievement_bonus +
    #         penalties
    #     )
        
    #     # Debug output
    #     if self.t % 20 == 0:
    #         print(f"t: {self.t:4d} | R: {reward:6.2f} | "
    #             f"Desc: {descent_reward:5.2f} | "
    #             f"Align: {alignment_reward:5.2f} | "
    #             f"Grasp: {grasp_reward:5.2f} | "
    #             f"Lift: {lift_reward:5.2f} | "
    #             f"Place: {placement_reward:5.2f} | "
    #             f"ActR: {action_reward:5.2f} | "
    #             f"ContactBonus: {contact_achievement_bonus:5.2f} | "
    #             f"Pen: {penalties:5.2f} | "
    #             f"DangerHtPen: {dangerous_height_penalty:5.2f} | "  # NEW
    #             f"GripperZ: {gripper_pos[2]:.4f} | "  # NEW
    #             f"BlockCenterZ: {block_center_z:.4f} | "  # NEW
    #             f"Readiness: {grasp_readiness:.2f} | "
    #             f"GState: {grasp_state} | "
    #             f"Grip: {grip_strength:.2f}")
        
    #     return reward

    # def compute_reward(self, observation, action):
    #     # Unpack observation components
    #     gripper_pos = observation[:3]      # End-effector position
    #     mug_pos = observation[3:6]         # Mug position
    #     target_pos = observation[6:9]      # Target position
    #     grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
    #     pad_pos = observation[10:13]       # Gripper pad position (right pad)
        
    #     # Geometric properties
    #     descent_FOS = 0.02
    #     block_height = get_body_size(self.model, "fish")[-1]
    #     block_top_z = mug_pos[2] + block_height + descent_FOS

    #     # Key spatial relationships
    #     pad_to_block_top = pad_pos[2] - block_top_z  # >0 when above block
    #     horizontal_error = np.linalg.norm(gripper_pos[:2] - mug_pos[:2])
        
    #     # ---- GRASP VALIDATION ----
    #     valid_grasp = (
    #         grasp_state == 2 and                         
    #         abs(pad_to_block_top) < 0.04 and             
    #         horizontal_error < 0.03                      
    #     )
        
    #     # ---- GRASP READINESS ----
    #     grasp_readiness = (
    #         np.exp(-80 * horizontal_error**2) * 
    #         np.exp(-150 * pad_to_block_top**2)
    #     )
        
    #     # ---- REWARD COMPONENTS ----
    #     # 1. Descent reward (unchanged)
    #     descent_reward = 10 * np.exp(-80 * pad_to_block_top**2)
        
    #     # 2. Alignment reward (unchanged)
    #     alignment_reward = 4 * np.exp(-60 * horizontal_error**2)
        
    #     # 3. PRECISION-TUNED GRASP REWARDS (MICRO-ADJUSTMENTS)
    #     grip_strength = action[-1]
    #     grasp_reward = (
    #         5.5 * (grasp_state >= 1) +                  
    #         8.5 * (grasp_state == 2) +                  
    #         25.5 * grip_strength * grasp_readiness +    # Increased from 22 -> 23.5
    #         30.5 * (grasp_state == 2) * grasp_readiness + # Increased from 27 -> 28.5
    #         13.5 * (grasp_state == 2) * grasp_readiness * np.tanh(8 * grip_strength)
    #     )
        
    #     # 4. Lifting reward (unchanged)
    #     lift_reward = 12 * (grasp_state == 2) * np.tanh(4 * mug_pos[2])
        
    #     # 5. Placement rewards (unchanged)
    #     d_place = np.linalg.norm(mug_pos - target_pos)
    #     placement_reward = -2 * d_place + 20 * np.exp(-70 * d_place**2)
    #     if d_place < 0.05 and valid_grasp:
    #         placement_reward += 40
        
    #     # 6. Finger clipping penalty (unchanged)
    #     clipping_penalty = -10 * (pad_to_block_top > 0) * grip_strength * np.exp(-50 * horizontal_error**2)
        
    #     # 6. NEW: DANGEROUS HEIGHT PENALTY
    #     # Severely penalize very low heights that cause collisions
    #     dangerous_height_penalty = -30 * np.exp(-500 * max(0, gripper_pos[2] - 0.05)**2)
        
    #     # 7. Other penalties (unchanged)
    #     penalties = (
    #         -40 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -25 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -8 * get_mug_toppled(self.model, self.data) +
    #         -4 * max(0, pad_to_block_top) +
    #         clipping_penalty +
    #         dangerous_height_penalty
    #     )
        
    #     # 8. Action encouragement (MICRO-BOOST)
    #     action_reward = 7.5 * grip_strength * grasp_readiness  # Increased from 7
        
    #     # 9. Contact achievement bonus (MICRO-BOOST)
    #     contact_achievement_bonus = 17.5 * (grasp_state >= 1) * grasp_readiness * np.tanh(10 * grip_strength)  # Increased from 17
        
    #     # ---- FINAL REWARD ----
    #     reward = (
    #         descent_reward +
    #         alignment_reward +
    #         grasp_reward +
    #         lift_reward +
    #         placement_reward +
    #         action_reward +
    #         contact_achievement_bonus +
    #         penalties
    #     )
        
    #     # Debug output
    #     if self.t % 20 == 0:
    #         # print(f"t: {self.t:4d} | R: {reward:6.2f} | "
    #         #     f"Desc: {descent_reward:5.2f} | "
    #         #     f"Align: {alignment_reward:5.2f} | "
    #         #     f"Grasp: {grasp_reward:5.2f} | "
    #         #     f"Lift: {lift_reward:5.2f} | "
    #         #     f"Place: {placement_reward:5.2f} | "
    #         #     f"ActR: {action_reward:5.2f} | "
    #         #     f"ContactBonus: {contact_achievement_bonus:5.2f} | "
    #         #     f"Pen: {penalties:5.2f} | "
    #         #     f"ClipPen: {clipping_penalty:5.2f} | "
    #         #     f"PadH: {pad_to_block_top:.3f} | "
    #         #     f"Readiness: {grasp_readiness:.2f} | "
    #         #     f"GState: {grasp_state} | "
    #         #     f"Grip: {grip_strength:.2f}")
    #         print(gripper_pos, mug_pos, block_height, block_top_z, action)
        
    #     return reward


    # def compute_reward(self, observation, action):
    #     # Unpack observation components
    #     gripper_pos = observation[:3]      # End-effector position
    #     mug_pos = observation[3:6]         # Mug position
    #     target_pos = observation[6:9]      # Target position
    #     grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
    #     pad_pos = observation[10:13]       # Gripper pad position (right pad)
        
    #     # Geometric properties
    #     block_height = get_body_size(self.model, "fish")[-1]
    #     block_top_z = mug_pos[2] + block_height/2
        
    #     # Key spatial relationships
    #     pad_to_block_top = pad_pos[2] - block_top_z  # >0 when above block
    #     horizontal_error = np.linalg.norm(gripper_pos[:2] - mug_pos[:2])
        
    #     # ---- GRASP VALIDATION ----
    #     valid_grasp = (
    #         grasp_state == 2 and                         
    #         abs(pad_to_block_top) < 0.04 and             
    #         horizontal_error < 0.03                      
    #     )
        
    #     # ---- GRASP READINESS ----
    #     grasp_readiness = (
    #         np.exp(-80 * horizontal_error**2) * 
    #         np.exp(-80 * pad_to_block_top**2)
    #     )
        
    #     # ---- REWARD COMPONENTS ----
    #     # 1. Descent reward (unchanged)
    #     descent_reward = 15 * np.exp(-80 * pad_to_block_top**2)
        
    #     # 2. Alignment reward (unchanged)
    #     alignment_reward = 4 * np.exp(-60 * horizontal_error**2)
        
    #     # 3. PRECISION-TUNED GRASP REWARDS (MICRO-ADJUSTMENTS)
    #     grip_strength = action[-1]
    #     grasp_reward = (
    #         5.5 * (grasp_state >= 1) +                  
    #         8.5 * (grasp_state == 2) +                  
    #         23.5 * grip_strength * grasp_readiness +    # Increased from 22
    #         28.5 * (grasp_state == 2) * grasp_readiness + # Increased from 27
    #         11.5 * (grasp_state == 2) * grasp_readiness * np.tanh(8 * grip_strength)
    #     )
        
    #     # 4. Lifting reward (unchanged)
    #     lift_reward = 12 * (grasp_state == 2) * np.tanh(4 * mug_pos[2])
        
    #     # 5. Placement rewards (unchanged)
    #     d_place = np.linalg.norm(mug_pos - target_pos)
    #     placement_reward = -2 * d_place + 20 * np.exp(-70 * d_place**2)
    #     if d_place < 0.05 and valid_grasp:
    #         placement_reward += 40
        
    #     # 6. Finger clipping penalty (unchanged)
    #     clipping_penalty = -10 * (pad_to_block_top > 0) * grip_strength * np.exp(-50 * horizontal_error**2)
        
    #     # 7. Other penalties (unchanged)
    #     penalties = (
    #         -40 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -25 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -8 * get_mug_toppled(self.model, self.data) +
    #         -4 * max(0, pad_to_block_top) +
    #         clipping_penalty
    #     )
        
    #     # 8. Action encouragement (MICRO-BOOST)
    #     action_reward = 7.5 * grip_strength * grasp_readiness  # Increased from 7
        
    #     # 9. Contact achievement bonus (MICRO-BOOST)
    #     contact_achievement_bonus = 17.5 * (grasp_state >= 1) * grasp_readiness * np.tanh(10 * grip_strength)  # Increased from 17
        
    #     # ---- FINAL REWARD ----
    #     reward = (
    #         descent_reward +
    #         alignment_reward +
    #         grasp_reward +
    #         lift_reward +
    #         placement_reward +
    #         action_reward +
    #         contact_achievement_bonus +
    #         penalties
    #     )
        
    #     # Debug output
    #     if self.t % 20 == 0:
    #         print(f"t: {self.t:4d} | R: {reward:6.2f} | "
    #             f"Desc: {descent_reward:5.2f} | "
    #             f"Align: {alignment_reward:5.2f} | "
    #             f"Grasp: {grasp_reward:5.2f} | "
    #             f"Lift: {lift_reward:5.2f} | "
    #             f"Place: {placement_reward:5.2f} | "
    #             f"ActR: {action_reward:5.2f} | "
    #             f"ContactBonus: {contact_achievement_bonus:5.2f} | "
    #             f"Pen: {penalties:5.2f} | "
    #             f"ClipPen: {clipping_penalty:5.2f} | "
    #             f"PadH: {pad_to_block_top:.3f} | "
    #             f"Readiness: {grasp_readiness:.2f} | "
    #             f"GState: {grasp_state} | "
    #             f"Grip: {grip_strength:.2f}")
        
    #     return reward


    # def compute_reward(self, observation, action):
    #     # Unpack observation components
    #     gripper_pos = observation[:3]      # End-effector position
    #     mug_pos = observation[3:6]         # Mug position
    #     target_pos = observation[6:9]      # Target position
    #     grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
    #     pad_pos = observation[10:13]       # Gripper pad position (right pad)
        
    #     # Geometric properties
    #     block_height = get_body_size(self.model, "fish")[-1]
    #     block_top_z = mug_pos[2] + block_height/2
        
    #     # Key spatial relationships
    #     pad_to_block_top = pad_pos[2] - block_top_z  # >0 when above block
    #     horizontal_error = np.linalg.norm(gripper_pos[:2] - mug_pos[:2])
        
    #     # ---- GRASP VALIDATION ----
    #     valid_grasp = (
    #         grasp_state == 2 and                         
    #         abs(pad_to_block_top) < 0.04 and             
    #         horizontal_error < 0.03                      
    #     )
        
    #     # ---- GRASP READINESS ----
    #     grasp_readiness = (
    #         np.exp(-80 * horizontal_error**2) * 
    #         np.exp(-80 * pad_to_block_top**2)
    #     )
        
    #     # ---- REWARD COMPONENTS ----
    #     # 1. Descent reward (unchanged)
    #     descent_reward = 15 * np.exp(-80 * pad_to_block_top**2)
        
    #     # 2. Alignment reward (unchanged)
    #     alignment_reward = 4 * np.exp(-60 * horizontal_error**2)
        
    #     # 3. PRECISION-TUNED GRASP REWARDS (AMPLIFIED)
    #     grip_strength = action[-1]
    #     grasp_reward = (
    #         # Slightly increased contact bonuses
    #         5.5 * (grasp_state >= 1) +                  # Any contact bonus
    #         8.5 * (grasp_state == 2) +                  # Full contact bonus
            
    #         # Amplified grip force reward
    #         22 * grip_strength * grasp_readiness +      # Increased from 20
            
    #         # Boosted grasp bonus
    #         27 * (grasp_state == 2) * grasp_readiness + # Increased from 25
            
    #         # Enhanced hold bonus
    #         11 * (grasp_state == 2) * grasp_readiness * np.tanh(8 * grip_strength) # Increased from 10
    #     )
        
    #     # 4. Lifting reward (unchanged)
    #     lift_reward = 12 * (grasp_state == 2) * np.tanh(4 * mug_pos[2])
        
    #     # 5. Placement rewards (unchanged)
    #     d_place = np.linalg.norm(mug_pos - target_pos)
    #     placement_reward = -2 * d_place + 20 * np.exp(-70 * d_place**2)
    #     if d_place < 0.05 and valid_grasp:
    #         placement_reward += 40
        
    #     # 6. Finger clipping penalty (unchanged)
    #     clipping_penalty = -10 * (pad_to_block_top > 0) * grip_strength * np.exp(-50 * horizontal_error**2)
        
    #     # 7. Other penalties (unchanged)
    #     penalties = (
    #         -40 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -25 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -8 * get_mug_toppled(self.model, self.data) +
    #         -4 * max(0, pad_to_block_top) +
    #         clipping_penalty
    #     )
        
    #     # 8. Action encouragement (AMPLIFIED)
    #     action_reward = 7 * grip_strength * grasp_readiness  # Increased from 6
        
    #     # 9. Contact achievement bonus (AMPLIFIED)
    #     contact_achievement_bonus = 17 * (grasp_state >= 1) * grasp_readiness * np.tanh(10 * grip_strength)  # Increased from 15
        
    #     # ---- FINAL REWARD ----
    #     reward = (
    #         descent_reward +
    #         alignment_reward +
    #         grasp_reward +
    #         lift_reward +
    #         placement_reward +
    #         action_reward +
    #         contact_achievement_bonus +
    #         penalties
    #     )
        
    #     # Debug output
    #     if self.t % 20 == 0:
    #         print(f"t: {self.t:4d} | R: {reward:6.2f} | "
    #             f"Desc: {descent_reward:5.2f} | "
    #             f"Align: {alignment_reward:5.2f} | "
    #             f"Grasp: {grasp_reward:5.2f} | "
    #             f"Lift: {lift_reward:5.2f} | "
    #             f"Place: {placement_reward:5.2f} | "
    #             f"ActR: {action_reward:5.2f} | "
    #             f"ContactBonus: {contact_achievement_bonus:5.2f} | "
    #             f"Pen: {penalties:5.2f} | "
    #             f"ClipPen: {clipping_penalty:5.2f} | "
    #             f"PadH: {pad_to_block_top:.3f} | "
    #             f"Readiness: {grasp_readiness:.2f} | "
    #             f"GState: {grasp_state} | "
    #             f"Grip: {grip_strength:.2f}")
        
    #     return reward


    # def reset_model(self):
    #     init_qpos, init_qvel = get_init(self.model, mode="deterministic", keyframe="down")
    #     self.set_state(init_qpos, init_qvel)

    #     # stop = np.hstack([get_mug_xpos(self.model, self.data), [0, 0, 0]])
    #     # self.traj_target = build_gripless_traj_gym(
    #     #     get_task_space_state(self.model, self.data)[:-1], 
    #     #     stop, hold=1)

    #     # pick = np.hstack([ get_mug_xpos(self.model, self.data), [0, 0, 0], [-1] ])
    #     # place = np.hstack([ get_ghost_xpos(self.model, self.data), [0, 0, 0], [-1] ])
    #     # self.traj_target = build_traj_l_pick_place(
    #     #     get_task_space_state(self.model, self.data), 
    #     #     [pick, place], hold=1)[:, :-1]

    #     self.t = 0
    #     self.tot_reward = 0
    #     # T = self.traj_target.shape[0]
    #     # self.traj_true = np.zeros(shape=(T, 7))
    #     # self.pos_errs = np.zeros(shape=(T, 3))
    #     # self.rot_errs = np.zeros(shape=(T, 3))
    #     # self.ctrls = np.zeros(shape=(T, self.model.nu))
    #     # self.actuator_frc = np.zeros(shape=(T, self.model.nu))
    #     # self.tot_pos_errs = np.zeros(3)
    #     # self.tot_rot_errs = np.zeros(3)

    #     # Not used - only needed so the call to pid_task_ctrl() doesn't throw an error
    #     self.pos_errs = np.zeros(shape=(1, 3))
    #     self.rot_errs = np.zeros(shape=(1, 3))
    #     self.tot_pos_errs = np.zeros(3)
    #     self.tot_rot_errs = np.zeros(3)
        
    #     print("Episode: ", self.episode)
    #     self.episode += 1
        
    #     return self._get_obs()
    

    # def _get_obs(self):
    #     return np.hstack([
    #         # get_robot_qpos(self.data).flat, # 14 dim  [:14]
    #         # get_robot_qvel(self.data).flat, # 14 dim  [14:28]
    #         get_2f85_xpos(self.model, self.data), # 3 dim [:3]
    #         get_mug_xpos(self.model, self.data), # 3 dim [3:6]
    #         get_ghost_xpos(self.model, self.data), # 3 dim [6:9] 
    #         get_block_grasp(self.model, self.data), # 1 dim [9:]
    #         get_site_xpos(self.model, self.data, "right_pad1_site")
    #     ])
    
    # def compute_reward(self, observation, action):
    #     # Unpack observation components
    #     gripper_pos = observation[:3]      # End-effector position
    #     mug_pos = observation[3:6]         # Mug position
    #     target_pos = observation[6:9]      # Target position
    #     grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
    #     pad_pos = observation[10:13]       # Gripper pad position (right pad)
        
    #     # Geometric properties
    #     block_height = get_body_size(self.model, "fish")[-1]
    #     block_top_z = mug_pos[2] + block_height/2
        
    #     # Key spatial relationships
    #     pad_to_block_top = pad_pos[2] - block_top_z  # >0 when above block
    #     horizontal_error = np.linalg.norm(gripper_pos[:2] - mug_pos[:2])
        
    #     # ---- GRASP VALIDATION ----
    #     valid_grasp = (
    #         grasp_state == 2 and                         
    #         abs(pad_to_block_top) < 0.04 and             
    #         horizontal_error < 0.03                      
    #     )
        
    #     # ---- GRASP READINESS ----
    #     grasp_readiness = (
    #         np.exp(-80 * horizontal_error**2) * 
    #         np.exp(-80 * pad_to_block_top**2)
    #     )
        
    #     # ---- REWARD COMPONENTS ----
    #     # 1. Descent reward
    #     descent_reward = 15 * np.exp(-80 * pad_to_block_top**2)
        
    #     # 2. Alignment reward
    #     alignment_reward = 4 * np.exp(-60 * horizontal_error**2)
        
    #     # 3. Precision-tuned grasp rewards
    #     grip_strength = action[-1]
    #     grasp_reward = (
    #         5 * (grasp_state >= 1) +                    # Any contact bonus
    #         8 * (grasp_state == 2) +                    # Full contact bonus
    #         20 * grip_strength * grasp_readiness +      # Grip force reward
    #         25 * (grasp_state == 2) * grasp_readiness + # Grasp bonus
    #         10 * (grasp_state == 2) * grasp_readiness * np.tanh(8 * grip_strength) # Hold bonus
    #     )
        
    #     # 4. Lifting reward
    #     lift_reward = 12 * (grasp_state == 2) * np.tanh(4 * mug_pos[2])
        
    #     # 5. Placement rewards
    #     d_place = np.linalg.norm(mug_pos - target_pos)
    #     placement_reward = -2 * d_place + 20 * np.exp(-70 * d_place**2)
    #     if d_place < 0.05 and valid_grasp:
    #         placement_reward += 40
        
    #     # 6. Finger clipping penalty
    #     clipping_penalty = -10 * (pad_to_block_top > 0) * grip_strength * np.exp(-50 * horizontal_error**2)
        
    #     # 7. Other penalties
    #     penalties = (
    #         -40 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -25 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -8 * get_mug_toppled(self.model, self.data) +
    #         -4 * max(0, pad_to_block_top) +  # General top contact penalty
    #         clipping_penalty
    #     )
        
    #     # 8. Action encouragement
    #     action_reward = 6 * grip_strength * grasp_readiness
        
    #     # 9. NEW: Contact achievement bonus
    #     contact_achievement_bonus = 15 * (grasp_state >= 1) * grasp_readiness * np.tanh(10 * grip_strength)
        
    #     # ---- FINAL REWARD ----
    #     reward = (
    #         descent_reward +
    #         alignment_reward +
    #         grasp_reward +
    #         lift_reward +
    #         placement_reward +
    #         action_reward +
    #         contact_achievement_bonus +  # NEW bonus
    #         penalties
    #     )
        
    #     # Debug output
    #     if self.t % 20 == 0:
    #         print(f"t: {self.t:4d} | R: {reward:6.2f} | "
    #             f"Desc: {descent_reward:5.2f} | "
    #             f"Align: {alignment_reward:5.2f} | "
    #             f"Grasp: {grasp_reward:5.2f} | "
    #             f"Lift: {lift_reward:5.2f} | "
    #             f"Place: {placement_reward:5.2f} | "
    #             f"ActR: {action_reward:5.2f} | "
    #             f"ContactBonus: {contact_achievement_bonus:5.2f} | "  # NEW
    #             f"Pen: {penalties:5.2f} | "
    #             f"ClipPen: {clipping_penalty:5.2f} | "
    #             f"PadH: {pad_to_block_top:.3f} | "
    #             f"Readiness: {grasp_readiness:.2f} | "
    #             f"GState: {grasp_state} | "
    #             f"Grip: {grip_strength:.2f}")
        
    #     return reward

    # def compute_reward(self, observation, action):
    #     # Unpack observation components
    #     gripper_pos = observation[:3]      # End-effector position
    #     mug_pos = observation[3:6]         # Mug position
    #     target_pos = observation[6:9]      # Target position
    #     grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
    #     pad_pos = observation[10:13]       # Gripper pad position (right pad)
        
    #     # Geometric properties
    #     block_height = get_body_size(self.model, "fish")[-1]
    #     block_top_z = mug_pos[2] + block_height/2
        
    #     # Key spatial relationships
    #     pad_to_block_top = pad_pos[2] - block_top_z  # >0 when above block
    #     horizontal_error = np.linalg.norm(gripper_pos[:2] - mug_pos[:2])
        
    #     # ---- GRASP VALIDATION ----
    #     valid_grasp = (
    #         grasp_state == 2 and                         
    #         abs(pad_to_block_top) < 0.04 and             
    #         horizontal_error < 0.03                      
    #     )
        
    #     # ---- GRASP READINESS ----
    #     grasp_readiness = (
    #         np.exp(-80 * horizontal_error**2) * 
    #         np.exp(-80 * pad_to_block_top**2)
    #     )
        
    #     # ---- REWARD COMPONENTS ----
    #     # 1. Descent reward
    #     descent_reward = 15 * np.exp(-80 * pad_to_block_top**2)
        
    #     # 2. Alignment reward
    #     alignment_reward = 4 * np.exp(-60 * horizontal_error**2)
        
    #     # 3. FINAL GRASP REWARD BALANCE
    #     grip_strength = action[-1]
    #     grasp_reward = (
    #         4 * (grasp_state >= 1) +                    # Any contact bonus
    #         6 * (grasp_state == 2) +                    # Full contact bonus
    #         15 * grip_strength * grasp_readiness +       # Grip force reward
    #         20 * (grasp_state == 2) * grasp_readiness + # Grasp bonus
    #         8 * (grasp_state == 2) * grasp_readiness * np.tanh(8 * grip_strength) # Hold bonus
    #     )
        
    #     # 4. Lifting reward
    #     lift_reward = 12 * (grasp_state == 2) * np.tanh(4 * mug_pos[2])
        
    #     # 5. Placement rewards
    #     d_place = np.linalg.norm(mug_pos - target_pos)
    #     placement_reward = -2 * d_place + 20 * np.exp(-70 * d_place**2)
    #     if d_place < 0.05 and valid_grasp:
    #         placement_reward += 40
        
    #     # 6. NEW: FINGER CLIPPING PENALTY
    #     # Penalize when pad is above block and gripper is closing
    #     clipping_penalty = -10 * (pad_to_block_top > 0) * grip_strength * np.exp(-50 * horizontal_error**2)
        
    #     # 7. Other penalties
    #     penalties = (
    #         -40 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -25 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -8 * get_mug_toppled(self.model, self.data) +
    #         -4 * max(0, pad_to_block_top) +  # General top contact penalty
    #         clipping_penalty  # Specific finger clipping penalty
    #     )
        
    #     # 8. Action encouragement
    #     action_reward = 5 * grip_strength * grasp_readiness
        
    #     # ---- FINAL REWARD ----
    #     reward = (
    #         descent_reward +
    #         alignment_reward +
    #         grasp_reward +
    #         lift_reward +
    #         placement_reward +
    #         action_reward +
    #         penalties
    #     )
        
    #     # Debug output
    #     if self.t % 20 == 0:
    #         print(f"t: {self.t:4d} | R: {reward:6.2f} | "
    #             f"Desc: {descent_reward:5.2f} | "
    #             f"Align: {alignment_reward:5.2f} | "
    #             f"Grasp: {grasp_reward:5.2f} | "
    #             f"Lift: {lift_reward:5.2f} | "
    #             f"Place: {placement_reward:5.2f} | "
    #             f"ActR: {action_reward:5.2f} | "
    #             f"Pen: {penalties:5.2f} | "
    #             f"ClipPen: {clipping_penalty:5.2f} | "
    #             f"PadH: {pad_to_block_top:.3f} | "
    #             f"Readiness: {grasp_readiness:.2f} | "
    #             f"GState: {grasp_state} | "
    #             f"Grip: {grip_strength:.2f}")
        
    #     return reward


    # def compute_reward(self, observation, action):
    #     # Unpack observation components
    #     gripper_pos = observation[:3]      # End-effector position
    #     mug_pos = observation[3:6]         # Mug position
    #     target_pos = observation[6:9]      # Target position
    #     grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
    #     pad_pos = observation[10:13]       # Gripper pad position (right pad)
        
    #     # Geometric properties
    #     block_height = get_body_size(self.model, "fish")[-1]
    #     block_top_z = mug_pos[2] + block_height/2
        
    #     # Key spatial relationships
    #     pad_to_block_top = pad_pos[2] - block_top_z  # >0 when above block
    #     horizontal_error = np.linalg.norm(gripper_pos[:2] - mug_pos[:2])
        
    #     # ---- GRASP VALIDATION ----
    #     # True when pads are at block's vertical center (±4cm) and aligned
    #     valid_grasp = (
    #         grasp_state == 2 and                         # Both pads contacting
    #         abs(pad_to_block_top) < 0.04 and             # Pads near mid-height
    #         horizontal_error < 0.03                      # Horizontally centered
    #     )
        
    #     # ---- GRASP READINESS ----
    #     # Measures how ideal the pre-grasp position is (0 to 1)
    #     grasp_readiness = (
    #         np.exp(-80 * horizontal_error**2) * 
    #         np.exp(-80 * pad_to_block_top**2)
    #     )
        
    #     # ---- REWARD COMPONENTS ----
    #     # 1. Descent reward - peaks at optimal grasp height
    #     descent_reward = 15 * np.exp(-80 * pad_to_block_top**2)
        
    #     # 2. Alignment reward
    #     alignment_reward = 4 * np.exp(-60 * horizontal_error**2)
        
    #     # 3. BALANCED grasp rewards
    #     grip_strength = action[-1]
    #     grasp_reward = (
    #         # Base contact bonuses
    #         4 * (grasp_state >= 1) +                    # Any contact bonus
    #         6 * (grasp_state == 2) +                    # Full contact bonus
            
    #         # Significant grip force reward when ready to grasp
    #         18 * grip_strength * grasp_readiness +
            
    #         # Substantial bonus for actually grasping at the right moment
    #         22 * (grasp_state == 2) * grasp_readiness +
            
    #         # Bonus for maintaining grasp while positioned
    #         10 * (grasp_state == 2) * grasp_readiness * np.tanh(8 * grip_strength)
    #     )
        
    #     # 4. Lifting reward - uses absolute Z since table=0
    #     lift_reward = 12 * (grasp_state == 2) * np.tanh(4 * mug_pos[2])
        
    #     # 5. Placement rewards
    #     d_place = np.linalg.norm(mug_pos - target_pos)
    #     placement_reward = -2 * d_place + 20 * np.exp(-70 * d_place**2)
    #     if d_place < 0.05 and valid_grasp:
    #         placement_reward += 40  # Significant success bonus
        
    #     # 6. Penalties
    #     penalties = (
    #         -40 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -25 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -8 * get_mug_toppled(self.model, self.data) +
    #         -4 * max(0, pad_to_block_top)  # Penalize pads above block
    #     )
        
    #     # 7. Action encouragement
    #     action_reward = 5 * grip_strength * grasp_readiness
        
    #     # ---- FINAL REWARD ----
    #     reward = (
    #         descent_reward +
    #         alignment_reward +
    #         grasp_reward +
    #         lift_reward +
    #         placement_reward +
    #         action_reward +
    #         penalties
    #     )
        
    #     # Debug output
    #     if self.t % 20 == 0:
    #         print(f"t: {self.t:4d} | R: {reward:6.2f} | "
    #             f"Desc: {descent_reward:5.2f} | "
    #             f"Align: {alignment_reward:5.2f} | "
    #             f"Grasp: {grasp_reward:5.2f} | "
    #             f"Lift: {lift_reward:5.2f} | "
    #             f"Place: {placement_reward:5.2f} | "
    #             f"ActR: {action_reward:5.2f} | "
    #             f"Pen: {penalties:5.2f} | "
    #             f"PadH: {pad_to_block_top:.3f} | "
    #             f"Readiness: {grasp_readiness:.2f} | "
    #             f"GState: {grasp_state} | "
    #             f"Grip: {grip_strength:.2f}")
        
    #     return reward
    

    # def compute_reward(self, observation, action):
    #     # Unpack observation components
    #     gripper_pos = observation[:3]      # End-effector position
    #     mug_pos = observation[3:6]         # Mug position
    #     target_pos = observation[6:9]      # Target position
    #     grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
    #     pad_pos = observation[10:13]       # Gripper pad position (right pad)
        
    #     # Geometric properties
    #     block_height = get_body_size(self.model, "fish")[-1]
    #     block_top_z = mug_pos[2] + block_height/2
        
    #     # Key spatial relationships
    #     pad_to_block_top = pad_pos[2] - block_top_z  # >0 when above block
    #     horizontal_error = np.linalg.norm(gripper_pos[:2] - mug_pos[:2])
        
    #     # ---- GRASP VALIDATION ----
    #     # True when pads are at block's vertical center (±4cm) and aligned
    #     valid_grasp = (
    #         grasp_state == 2 and                         # Both pads contacting
    #         abs(pad_to_block_top) < 0.04 and             # Pads near mid-height
    #         horizontal_error < 0.03                      # Horizontally centered
    #     )
        
    #     # ---- GRASP READINESS ----
    #     # Measures how ideal the pre-grasp position is (0 to 1)
    #     grasp_readiness = (
    #         np.exp(-80 * horizontal_error**2) * 
    #         np.exp(-80 * pad_to_block_top**2)
    #     )
        
    #     # ---- REWARD COMPONENTS ----
    #     # 1. Descent reward - peaks at optimal grasp height
    #     descent_reward = 20 * np.exp(-80 * pad_to_block_top**2)
        
    #     # 2. Alignment reward
    #     alignment_reward = 5 * np.exp(-60 * horizontal_error**2)
        
    #     # 3. SUPERCHARGED grasp rewards
    #     grip_strength = action[-1]
    #     grasp_reward = (
    #         # Base contact bonuses
    #         5 * (grasp_state >= 1) +                    # Any contact bonus
    #         8 * (grasp_state == 2) +                    # Full contact bonus
            
    #         # Massive grip force reward when ready to grasp
    #         25 * grip_strength * grasp_readiness +
            
    #         # Huge bonus for actually grasping at the right moment
    #         30 * (grasp_state == 2) * grasp_readiness +
            
    #         # Progressive bonus for maintaining grasp while positioned
    #         15 * (grasp_state == 2) * grasp_readiness * np.tanh(10 * grip_strength)
    #     )
        
    #     # 4. Lifting reward - uses absolute Z since table=0
    #     lift_reward = 15 * (grasp_state == 2) * np.tanh(5 * mug_pos[2])
        
    #     # 5. Placement rewards
    #     d_place = np.linalg.norm(mug_pos - target_pos)
    #     placement_reward = -3 * d_place + 25 * np.exp(-80 * d_place**2)
    #     if d_place < 0.05 and valid_grasp:
    #         placement_reward += 50  # Massive success bonus
        
    #     # 6. Penalties
    #     penalties = (
    #         -50 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -30 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -10 * get_mug_toppled(self.model, self.data) +
    #         -5 * max(0, pad_to_block_top)  # Penalize pads above block
    #     )
        
    #     # 7. Action encouragement (MAXIMIZED)
    #     action_reward = 8 * grip_strength * grasp_readiness
        
    #     # ---- FINAL REWARD ----
    #     reward = (
    #         descent_reward +
    #         alignment_reward +
    #         grasp_reward +
    #         lift_reward +
    #         placement_reward +
    #         action_reward +
    #         penalties
    #     )
        
    #     # Debug output
    #     if self.t % 20 == 0:
    #         print(f"t: {self.t:4d} | R: {reward:6.2f} | "
    #             f"Desc: {descent_reward:5.2f} | "
    #             f"Align: {alignment_reward:5.2f} | "
    #             f"Grasp: {grasp_reward:5.2f} | "
    #             f"Lift: {lift_reward:5.2f} | "
    #             f"Place: {placement_reward:5.2f} | "
    #             f"ActR: {action_reward:5.2f} | "
    #             f"Pen: {penalties:5.2f} | "
    #             f"PadH: {pad_to_block_top:.3f} | "
    #             f"Readiness: {grasp_readiness:.2f} | "
    #             f"GState: {grasp_state} | "
    #             f"Grip: {grip_strength:.2f}")
        
    #     return reward

    # def compute_reward(self, observation, action):
    #     # Unpack observation components
    #     gripper_pos = observation[:3]      # End-effector position
    #     mug_pos = observation[3:6]         # Mug position
    #     target_pos = observation[6:9]      # Target position
    #     grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
    #     pad_pos = observation[10:13]       # Gripper pad position (right pad)
        
    #     # Geometric properties
    #     block_height = get_body_size(self.model, "fish")[-1]
    #     block_top_z = mug_pos[2] + block_height/2
        
    #     # Key spatial relationships
    #     pad_to_block_top = pad_pos[2] - block_top_z  # >0 when above block
    #     horizontal_error = np.linalg.norm(gripper_pos[:2] - mug_pos[:2])
        
    #     # ---- GRASP VALIDATION ----
    #     # True when pads are at block's vertical center (±2cm) and aligned
    #     valid_grasp = (
    #         grasp_state == 2 and                         # Both pads contacting
    #         abs(pad_to_block_top) < 0.04 and             # Pads near mid-height
    #         horizontal_error < 0.03                      # Horizontally centered
    #     )
        
    #     # ---- CRITICAL NEW: GRASP READINESS ----
    #     # Measures how ideal the pre-grasp position is (0 to 1)
    #     grasp_readiness = (
    #         np.exp(-80 * horizontal_error**2) * 
    #         np.exp(-80 * pad_to_block_top**2)
    #     )
        
    #     # ---- REWARD COMPONENTS ----
    #     # 1. Descent reward - peaks at optimal grasp height
    #     descent_reward = 20 * np.exp(-80 * pad_to_block_top**2)
        
    #     # 2. Alignment reward
    #     alignment_reward = 5 * np.exp(-60 * horizontal_error**2)
        
    #     # 3. Enhanced grasp rewards (STRONGER INCENTIVES)
    #     grip_strength = action[-1]
    #     grasp_reward = (
    #         # Base contact bonuses
    #         3 * (grasp_state >= 1) +                    # Any contact bonus
    #         5 * (grasp_state == 2) +                    # Full contact bonus
            
    #         # Strong grip force reward when ready to grasp
    #         10 * grip_strength * grasp_readiness +
            
    #         # Extra bonus for actually grasping at the right moment
    #         15 * (grasp_state == 2) * grasp_readiness
    #     )
        
    #     # 4. Lifting reward - uses absolute Z since table=0
    #     lift_reward = 12 * (grasp_state == 2) * np.tanh(5 * mug_pos[2])
        
    #     # 5. Placement rewards
    #     d_place = np.linalg.norm(mug_pos - target_pos)
    #     placement_reward = -3 * d_place + 20 * np.exp(-80 * d_place**2)
    #     if d_place < 0.05 and valid_grasp:
    #         placement_reward += 40  # Increased success bonus
        
    #     # 6. Penalties
    #     penalties = (
    #         -50 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -30 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -10 * get_mug_toppled(self.model, self.data) +
    #         -5 * max(0, pad_to_block_top)  # Penalize pads above block
    #     )
        
    #     # 7. Action encouragement (STRONGER)
    #     action_reward = 4 * grip_strength * grasp_readiness
        
    #     # ---- FINAL REWARD ----
    #     reward = (
    #         descent_reward +
    #         alignment_reward +
    #         grasp_reward +
    #         lift_reward +
    #         placement_reward +
    #         action_reward +
    #         penalties
    #     )
        
    #     # Debug output
    #     if self.t % 20 == 0:
    #         print(f"t: {self.t:4d} | R: {reward:6.2f} | "
    #             f"Desc: {descent_reward:5.2f} | "
    #             f"Align: {alignment_reward:5.2f} | "
    #             f"Grasp: {grasp_reward:5.2f} | "
    #             f"Lift: {lift_reward:5.2f} | "
    #             f"Place: {placement_reward:5.2f} | "
    #             f"ActR: {action_reward:5.2f} | "
    #             f"Pen: {penalties:5.2f} | "
    #             f"PadH: {pad_to_block_top:.3f} | "
    #             f"Readiness: {grasp_readiness:.2f} | "
    #             f"GState: {grasp_state} | "
    #             f"Grip: {grip_strength:.2f}")
        
    #     return reward


    # def compute_reward(self, observation, action):
    #     # Unpack observation components
    #     gripper_pos = observation[:3]      # End-effector position
    #     mug_pos = observation[3:6]         # Mug position
    #     target_pos = observation[6:9]      # Target position
    #     grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
    #     pad_pos = observation[10:13]       # Gripper pad position (right pad)
        
    #     # Geometric properties
    #     mug_half_height = get_body_size(self.model, "fish")[-1]/2
    #     block_top_z = mug_pos[2] + mug_half_height
    #     # table_height = 0.045  # Adjust if different
        
    #     # Critical spatial relationships
    #     pad_height_above_block = pad_pos[2] - block_top_z
    #     horizontal_error = np.linalg.norm(gripper_pos[:2] - mug_pos[:2])
        
    #     # ---- VALID GRASP CHECK ----
    #     # Looser conditions for valid grasp
    #     valid_grasp = (
    #         grasp_state == 2 and                         # Both pads contacting
    #         abs(pad_height_above_block) < 0.04 and       # More lenient height condition
    #         horizontal_error < 0.03                      # More lenient horizontal condition
    #     )
        
    #     # ---- REWARD COMPONENTS ----
    #     # 1. Descent reward - peaks when pads at ideal side-grasp height
    #     descent_reward = 20 * np.exp(-80 * (pad_height_above_block)**2)
        
    #     # 2. Horizontal alignment reward
    #     alignment_reward = 5 * np.exp(-60 * horizontal_error**2)
        
    #     # 3. Grasp rewards (encourage closing gripper)
    #     grip_strength = action[-1]
    #     grasp_reward = (
    #         # Reward for any contact
    #         3 * (grasp_state >= 1) * np.exp(-60 * horizontal_error**2) +
    #         # Extra reward for full contact
    #         5 * (grasp_state == 2) * np.exp(-60 * horizontal_error**2) +
    #         # Reward for grip force when aligned
    #         4 * grip_strength * np.exp(-60 * horizontal_error**2)
    #     )
        
    #     # 4. Lifting reward (uses height above table)
    #     lift_reward = 10 * (grasp_state == 2) * np.tanh(5 * max(0, mug_pos[2]))
        
    #     # 5. Placement rewards
    #     d_place = np.linalg.norm(mug_pos - target_pos)
    #     placement_reward = -3 * d_place + 20 * np.exp(-80 * d_place**2)
    #     if d_place < 0.05 and grasp_state == 2 and mug_pos[2] > 0.02:
    #         placement_reward += 30
        
    #     # 6. Penalties (less severe)
    #     penalties = (
    #         -50 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -30 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -10 * get_mug_toppled(self.model, self.data) +
    #         # Only penalize top contact when pad is significantly above block
    #         -5 * (grasp_state == 2) * max(0, pad_height_above_block)
    #     )
        
    #     # 7. Action reward (encourage closing gripper)
    #     action_reward = 2 * grip_strength * np.exp(-60 * horizontal_error**2)
        
    #     # ---- COMPOSITE REWARD ----
    #     reward = (
    #         descent_reward +
    #         alignment_reward +
    #         grasp_reward +
    #         lift_reward +
    #         placement_reward +
    #         action_reward +
    #         penalties
    #     )
        
    #     # Debug output
    #     if self.t % 20 == 0:  # More frequent output
    #         print(f"t: {self.t:4d} | R: {reward:6.2f} | "
    #             f"Desc: {descent_reward:5.2f} | "
    #             f"Align: {alignment_reward:5.2f} | "
    #             f"Grasp: {grasp_reward:5.2f} | "
    #             f"Lift: {lift_reward:5.2f} | "
    #             f"Place: {placement_reward:5.2f} | "
    #             f"ActR: {action_reward:5.2f} | "
    #             f"Pen: {penalties:5.2f} | "
    #             f"PadH: {pad_height_above_block:.3f} | "
    #             f"Herr: {horizontal_error:.3f} | "
    #             f"GState: {grasp_state} | "
    #             f"Grip: {grip_strength:.2f}")
        
    #     return reward


    # def compute_reward(self, observation, action):
    #     # Unpack observation components
    #     # DEEPSEEK 4
    #     gripper_pos = observation[:3]      # End-effector position
    #     mug_pos = observation[3:6]         # Mug position
    #     target_pos = observation[6:9]      # Target position
    #     grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
    #     pad_pos = observation[10:13]       # Gripper pad position (right pad)
        
    #     # Geometric properties
    #     mug_half_height = get_body_size(self.model, "fish")[-1]/2
    #     block_top_z = mug_pos[2] + mug_half_height
        
    #     # Critical spatial relationships
    #     pad_height_above_block = pad_pos[2] - block_top_z
    #     gripper_height_above_block = gripper_pos[2] - block_top_z
    #     horizontal_error = np.linalg.norm(gripper_pos[:2] - mug_pos[:2])
        
    #     # ---- VALID GRASP CHECK ----
    #     # True only when pads are at side-grasp height and horizontally aligned
    #     valid_grasp = (
    #         grasp_state == 2 and                         # Both pads contacting
    #         # abs(pad_height_above_block) < 0.005 and      # Pads at block's vertical center
    #         abs(pad_height_above_block) < 0.02 and      # Pads at block's vertical center
    #         # abs(pad_height_above_block) < 0.001 and      # Pads at block's vertical center
    #         horizontal_error < 0.02                      # Horizontally centered
    #     )
    #     # print(pad_height_above_block)
        
    #     # ---- REWARD COMPONENTS ----
    #     # 1. Descent reward - peaks when pads at ideal side-grasp height
    #     descent_reward = 20 * np.exp(-100 * (pad_height_above_block)**2)
        
    #     # 2. Horizontal alignment reward
    #     alignment_reward = 5 * np.exp(-80 * horizontal_error**2)
        
    #     # 3. Grasp rewards (ONLY for valid side-grasps)
    #     grip_strength = action[-1]
    #     grasp_reward = (
    #         8 * valid_grasp * grip_strength +          # Reward grip force only for valid grasps
    #         15 * valid_grasp * np.tanh(5 * (mug_pos[2]))  # Lifting reward
    #     )
                
    #     # 4. Anti-top-contact penalty
    #     top_contact_penalty = -10 * (grasp_state == 2) * np.exp(-50 * max(0, pad_height_above_block)**2)
        
    #     # 5. Placement rewards
    #     d_place = np.linalg.norm(mug_pos - target_pos)
    #     placement_reward = -3 * d_place + 25 * np.exp(-100 * d_place**2)
    #     if d_place < 0.05 and valid_grasp:
    #         placement_reward += 30
        
    #     # 6. Penalties
    #     penalties = (
    #         -100 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -50 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -20 * get_mug_toppled(self.model, self.data) +
    #         top_contact_penalty
    #     )
        
    #     # ---- COMPOSITE REWARD ----
    #     reward = (
    #         descent_reward +
    #         alignment_reward +
    #         grasp_reward +
    #         placement_reward +
    #         penalties
    #     )
        
    #     # Debug output
    #     # if self.t % 50 == 0:
    #     #     print(f"t: {self.t:4d} | R: {reward:6.2f} | "
    #     #         f"Desc: {descent_reward:5.2f} | "
    #     #         f"Align: {alignment_reward:5.2f} | "
    #     #         f"Grasp: {grasp_reward:5.2f} | "
    #     #         f"Place: {placement_reward:5.2f} | "
    #     #         f"Pen: {penalties:5.2f} | "
    #     #         f"PadH: {pad_height_above_block:.3f} | "
    #     #         f"ValidG: {int(valid_grasp)}")
        
    #     return reward


    # def compute_reward(self, observation, action):
    #     """
    #     Normalized, differentiable reward function with O(1) running RMS.
    #     Combines potential-based shaping with task-specific incentives.
    #     """
    #     DEEPSEEK 3
    #     # Unpack observation components
    #     gripper_pos = observation[:3]      # End-effector position
    #     mug_pos = observation[3:6]         # Mug position
    #     target_pos = observation[6:9]      # Target position
    #     grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
    #     pad_pos = observation[10:13]       # Gripper pad position
        
    #     # Geometric properties
    #     mug_height = get_body_size(self.model, "fish")[-1]/2
    #     table_height = 0.045
    #     sweetspot_z = mug_pos[2] + mug_height + 0.01
    #     pad_z = pad_pos[2]
        
    #     # Distance metrics
    #     d_xy = np.linalg.norm(gripper_pos[:2] - mug_pos[:2])
    #     d_place = np.linalg.norm(mug_pos - target_pos)
    #     z_error = max(0, sweetspot_z - pad_z)
    #     lift_height = max(0, mug_pos[2] - table_height)
        
    #     # ---- Normalized Potential Functions (O(1) scale) ----
    #     # Alignment potential (max 1.0)
    #     phi_align = np.exp(-80 * d_xy**2)
        
    #     # Descent potential (max 1.0)
    #     phi_descent = np.exp(-100 * z_error**2)
        
    #     # Contact potential
    #     phi_contact = 0.5 * (grasp_state >= 1) + 0.5 * (grasp_state == 2)
        
    #     # Grip potential
    #     grip_strength = action[-1]
    #     phi_grip = 1 / (1 + np.exp(-20 * (grip_strength - 0.6)))
        
    #     # Lift potential
    #     phi_lift = np.tanh(5 * lift_height)
        
    #     # Placement potential
    #     place_thresh = 0.05
    #     phi_place = -d_place + 1 / (1 + np.exp(50 * (d_place - place_thresh)))
        
    #     # ---- Composite Reward (Normalized components) ----
    #     # Weighted potentials (all O(1))
    #     reward = (
    #         0.8 * phi_align * phi_descent +      # Alignment during descent
    #         0.6 * phi_contact * phi_grip +       # Contact + grip
    #         0.7 * (grasp_state == 2) * phi_lift + # Lifting only when grasped
    #         0.9 * phi_place                     # Placement
    #     )
        
    #     # ---- Penalties (Normalized) ----
    #     penalties = (
    #         -1.0 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -0.5 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -0.2 * get_mug_toppled(self.model, self.data) +
    #         -0.4 * (grasp_state < 2) * np.exp(-15 * lift_height**2)  # Moving without grasp
    #     )
        
    #     # Time penalty (normalized)
    #     time_pen = -0.01
        
    #     # Action penalty (normalized)
    #     action_pen = -0.001 * np.sum(action**2)
        
    #     # Final reward
    #     total_reward = reward + penalties + time_pen + action_pen
        
    #     # Debug output
    #     if self.t % 100 == 0:
    #         print(f"t: {self.t:4d} | R: {total_reward:6.2f} | "
    #             f"Align: {phi_align:.2f} | "
    #             f"Descent: {phi_descent:.2f} | "
    #             f"Contact: {phi_contact:.2f} | "
    #             f"Grip: {phi_grip:.2f} | "
    #             f"Lift: {phi_lift:.2f} | "
    #             f"Place: {phi_place:.2f} | "
    #             f"Pen: {penalties:.2f} | "
    #             f"GState: {grasp_state}")
        
    #     return total_reward


    # def compute_reward(self, observation, action):
    #     DEEPSEEK 2
    #     # Unpack observation components
    #     gripper_pos = observation[:3]      # End-effector position
    #     mug_pos = observation[3:6]         # Mug position
    #     target_pos = observation[6:9]      # Target position
    #     grasp_state = observation[9]       # Grasp state (0-2)
    #     pad_pos = observation[10:13]       # Gripper pad position

    #     # Calculate key distances and heights
    #     mug_height = get_body_size(self.model, "fish")[-1]/2
    #     sweetspot_z = mug_pos[2] + mug_height + 0.01
    #     pad_z = pad_pos[2]
        
    #     # Key vectors and distances
    #     gripper_to_mug = gripper_pos - mug_pos
    #     mug_to_target = mug_pos - target_pos
    #     d_xy = np.linalg.norm(gripper_to_mug[:2])  # Horizontal distance to mug
    #     d_place = np.linalg.norm(mug_to_target)    # Distance to target
        
    #     # ---- Critical new components ----
    #     # 1. Descent reward - strongly encourages moving down toward mug
    #     z_error = max(0, sweetspot_z - pad_z)  # Distance to grasp sweet spot
    #     descent_bonus = 15 * (1 - np.tanh(20 * z_error))
        
    #     # 2. Height difference reward - prioritizes lowering before moving
    #     height_diff = gripper_pos[2] - mug_pos[2]
    #     height_reward = 5 * np.exp(-10 * max(0, height_diff)**2)
        
    #     # 3. Grasp readiness - rewards proper positioning before gripping
    #     xy_aligned = np.exp(-80 * d_xy**2)
    #     z_ready = np.exp(-80 * max(0, height_diff - 0.05)**2)
    #     grasp_readiness = 3 * xy_aligned * z_ready
        
    #     # 4. Grasp mechanics - only rewards gripping when properly positioned
    #     grip_strength = action[-1]
    #     grip_bonus = 8 * grip_strength * xy_aligned * np.exp(-50 * z_error**2)
        
    #     # 5. Anti-cheat penalty - punishes moving toward target without mug
    #     mug_height_above_table = mug_pos[2] - 0.045  # Table height = 0.045
    #     moving_without_mug = -10 * (grasp_state < 1) * np.exp(-10 * mug_height_above_table**2)
        
    #     # ---- Placement rewards ----
    #     placement_reward = -3 * d_place + 20 * np.exp(-80 * d_place**2)
        
    #     # Add placement bonus only when mug is lifted and near target
    #     if d_place < 0.05 and grasp_state > 0.5 and mug_pos[2] > sweetspot_z:
    #         placement_reward += 30
        
    #     # ---- Penalties ----
    #     penalties = (
    #         -100 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -50 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -15 * get_mug_toppled(self.model, self.data) +
    #         -5 * grip_strength * (gripper_pos[2] > sweetspot_z + 0.04)  # Penalize high gripping
    #     )
        
    #     # ---- Composite reward ----
    #     reward = (
    #         descent_bonus +
    #         height_reward +
    #         grasp_readiness +
    #         grip_bonus +
    #         placement_reward +
    #         moving_without_mug +
    #         penalties
    #     )
        
    #     # Debug output
    #     if self.t % 50 == 0:
    #         print(f"t: {self.t:4d} | R: {reward:6.2f} | "
    #             f"Desc: {descent_bonus:5.2f} | "
    #             f"Ht: {height_reward:5.2f} | "
    #             f"Readiness: {grasp_readiness:5.2f} | "
    #             f"Grip: {grip_bonus:5.2f} | "
    #             f"Place: {placement_reward:5.2f} | "
    #             f"Pen: {penalties:5.2f} | "
    #             f"Zerr: {z_error:.3f}")
        
    #     return reward
    

    # def compute_reward(self, observation, action):
    #     """
    #     DEEPSEEK 1
    #     Optimized hybrid reward function combining:
    #     - Height-conditioned alignment from our approach
    #     - Potential-based shaping from GPT's version
    #     - Enhanced grasp mechanics from both
    #     - Safety and efficiency components
        
    #     Components:
    #     ▸ Alignment       - Height-conditioned XY centering (exponential)
    #     ▸ Pick Height     - Smooth pad lowering (exponential)
    #     ▸ Grip Squeeze    - Proximity-scaled grip bonus (logistic)
    #     ▸ Lift            - Height-gain reward (tanh)
    #     ▸ Placement       - Distance penalty + success bonus (linear + logistic)
    #     ▸ Safety          - Collision/toppling penalties
    #     ▸ Efficiency      - Time and action penalties
    #     """

    #     # Unpack observation components
    #     gripper_xpos = observation[:3]    # End-effector position
    #     mug_xpos = observation[3:6]       # Mug position
    #     ghost_xpos = observation[6:9]     # Target position
    #     grasp_contact = observation[9]    # Grasp state (0-2)
    #     pad_x, pad_y, pad_z = observation[10:13]  # Gripper pad position

    #     # Calculate key distances and differences
    #     dx, dy, dz = gripper_xpos - mug_xpos
    #     d_xy = np.sqrt(dx**2 + dy**2)
    #     d_place = np.linalg.norm(mug_xpos - ghost_xpos)
    #     mug_height = get_body_size(self.model, "fish")[-1]/2
    #     sweetspot_z = mug_xpos[2] + mug_height + 0.01
        
    #     # Constants (tuned parameters)
    #     k_align = 80.0       # XY alignment bandwidth
    #     k_height = 60.0      # Z approach bandwidth
    #     k_place = 50.0       # Placement bonus bandwidth
    #     k_lift = 4.0         # Lift reward slope
    #     tau_grip = 0.60      # Grip activation threshold
    #     tau_place = 0.05     # Placement success threshold
        
    #     # --- Reward Components ---
        
    #     # 1. Height-conditioned XY Alignment
    #     z_above = max(0, gripper_xpos[2] - mug_xpos[2])
    #     r_align = 3 * (np.exp(-k_align * dx**2) + 
    #                 np.exp(-k_align * dy**2)) * \
    #                 (1 - np.exp(-8 * z_above))
        
    #     # 2. Pad Approach (Z-direction)
    #     z_error = max(0, sweetspot_z - pad_z)
    #     r_approach = 8 * np.exp(-k_height * z_error**2)
        
    #     # 3. Grasp Mechanics (Proximity-scaled)
    #     grip_strength = np.clip(action[-1], 0.0, 1.0)
    #     r_grasp = (
    #         # Grip activation bonus (logistic)
    #         5 / (1 + np.exp(-20 * (grip_strength - tau_grip))) *
    #         # Scaled by proximity to sweet spot
    #         np.exp(-40 * z_error**2) +
    #         # Continuous contact-based lifting
    #         7 * (grasp_contact / 2) * 
    #         np.tanh(k_lift * (gripper_xpos[2] - mug_xpos[2]))
    #     )
        
    #     # 4. Placement (Dense + Success Bonus)
    #     r_place = (
    #         -4 * d_place +  # Distance penalty
    #         # Success bonus (logistic)
    #         25 / (1 + np.exp(k_place * (d_place - tau_place)))
    #     )
        
    #     # 5. Penalties
    #     penalties = (
    #         -100 * get_self_collision(self.model, self.data, self.collision_cache) +
    #         -50 * get_table_collision(self.model, self.data, self.collision_cache) +
    #         -10 * get_mug_toppled(self.model, self.data)
    #     )
        
    #     # 6. Efficiency Incentives
    #     efficiency = (
    #         -0.01 +  # Time penalty
    #         -0.001 * np.sum(action**2)  # Action magnitude penalty
    #     )
        
    #     # --- Composite Reward ---
    #     reward = (
    #         r_align +
    #         r_approach +
    #         r_grasp +
    #         r_place +
    #         penalties +
    #         efficiency
    #     )
        
    #     # Debug output (optional)
    #     if self.t % 100 == 0:
    #         print(f"t: {self.t:4d} | R: {reward:6.2f} = "
    #             f"Align: {r_align:5.2f} | "
    #             f"Approach: {r_approach:5.2f} | "
    #             f"Grasp: {r_grasp:5.2f} | "
    #             f"Place: {r_place:5.2f} | "
    #             f"Penalties: {penalties:5.2f}")
        
    #     return reward



        

    # def compute_reward(self, observation, action):
    #     """
    #     GPT
    #     Reward for UR3e pick‑and‑place, combining smooth potential‑based shaping
    #     with a height‑gated centring term and a discrete success bonus.

    #     Components (all continuous unless noted)
    #         • XY centring (gated by gripper height)
    #         • Pad approach in Z
    #         • Grip squeeze (logistic)
    #         • Lift once grasped
    #         • Dense placement shaping + logistic proximity bonus
    #         • Discrete success bonus at 5 cm
    #         • Safety penalties (self, table, toppling)
    #         • Efficiency penalties (time, ‖action‖²)

    #     All dense terms are scaled so their running RMS is O(1).
    #     """

    #     # ------------------ unpack observation -----------------------------------
    #     gripper_xpos = observation[:3]          # end‑effector (x,y,z)
    #     mug_xpos     = observation[3:6]         # mug COM (x,y,z)
    #     ghost_xpos   = observation[6:9]         # target (x,y,z)
    #     grasp_signal = observation[9]           # 0=none, 1=touch, 2=held
    #     _, _, pad_z  = observation[10:13]       # gripper pad height

    #     # ------------------ constants --------------------------------------------
    #     k_xy     = 120.0
    #     k_z      = 200.0
    #     k_place  = 60.0
    #     k_lift   = 10.0
    #     tau_grip = 0.60
    #     tau_place = 0.05
    #     time_penalty = -0.01
    #     action_penalty = -0.001

    #     mug_half_height = get_body_size(self.model, "fish")[-1] / 2
    #     grip_strength = np.clip(action[-1], 0.0, 1.0)

    #     # ------------------ geometric errors -------------------------------------
    #     dx, dy, dz = gripper_xpos - mug_xpos
    #     d_xy = np.hypot(dx, dy)
    #     z_above = max(0.0, dz)                       # how far gripper is above mug
    #     pad_target_z = mug_xpos[2] + mug_half_height + 0.01
    #     z_err = pad_target_z - pad_z                # >0 while pad still above rim
    #     d_place = np.linalg.norm(mug_xpos - ghost_xpos)

    #     # ------------------ dense shaping potentials -----------------------------
    #     # Height‑gated XY centring
    #     gate = 1.0 - np.exp(-8.0 * z_above)         # rises smoothly from 0→1
    #     phi_center = gate * np.exp(-k_xy * d_xy**2) # max ≈ 1

    #     # Z approach to rim
    #     phi_approach = np.exp(-k_z * z_err**2)

    #     # Grip squeeze as logistic
    #     phi_grip = 1.0 / (1.0 + np.exp(-20.0 * (grip_strength - tau_grip)))

    #     # Lift once held
    #     phi_lift = (
    #         phi_grip * grasp_signal / 2.0 *
    #         np.tanh(k_lift * dz)
    #     )

    #     # Placement shaping
    #     phi_place_dense = -d_place                   # negative distance
    #     phi_place_bonus = 1.0 / (1.0 + np.exp(k_place * (d_place - tau_place)))

    #     # ------------------ discrete success bonus -------------------------------
    #     success_bonus = 0.0
    #     if d_place < tau_place and grasp_signal > 0.5:
    #         success_bonus = 1.0                      # small, sparse reward

    #     # ------------------ safety & efficiency ----------------------------------
    #     penalty_self   = -1.0 * get_self_collision(self.model, self.data, self.collision_cache)
    #     penalty_table  = -0.5 * get_table_collision(self.model, self.data, self.collision_cache)
    #     penalty_topple = -0.1 * get_mug_toppled(self.model, self.data)

    #     dense = (
    #         phi_center
    #         + phi_approach
    #         + phi_grip
    #         + phi_lift
    #         + phi_place_dense
    #         + phi_place_bonus
    #     )

    #     sparse = (
    #         success_bonus
    #         + penalty_self
    #         + penalty_table
    #         + penalty_topple
    #         + time_penalty
    #         + action_penalty * float(np.sum(action ** 2))
    #     )

    #     return float(dense + sparse)



def compute_reward(self, observation, action):
    # Unpack observation components
    gripper_pos = observation[:3]      # End-effector position
    mug_pos = observation[3:6]         # Mug position
    target_pos = observation[6:9]      # Target position
    grasp_state = observation[9]       # 0=no contact, 1=1-pad contact, 2=2-pad contact
    pad_pos = observation[10:13]       # Gripper pad position (right pad)
    
    # Geometric properties
    mug_height = get_body_size(self.model, "fish")[-1]/2
    table_height = 0.045
    block_top_z = mug_pos[2] + mug_height
    
    # Critical spatial relationships
    pad_height_above_block = pad_pos[2] - block_top_z
    gripper_height_above_block = gripper_pos[2] - block_top_z
    horizontal_error = np.linalg.norm(gripper_pos[:2] - mug_pos[:2])
    
    # ---- VALID GRASP CHECK ----
    # True only when pads are at side-grasp height and horizontally aligned
    valid_grasp = (
        grasp_state == 2 and                         # Both pads contacting
        abs(pad_height_above_block) < 0.005 and      # Pads at block's vertical center
        horizontal_error < 0.02                      # Horizontally centered
    )
    
    # ---- REWARD COMPONENTS ----
    # 1. Descent reward - peaks when pads at ideal side-grasp height
    descent_reward = 20 * np.exp(-100 * (pad_height_above_block)**2)
    
    # 2. Horizontal alignment reward
    alignment_reward = 5 * np.exp(-80 * horizontal_error**2)
    
    # 3. Grasp rewards (ONLY for valid side-grasps)
    grip_strength = action[-1]
    grasp_reward = (
        8 * valid_grasp * grip_strength +          # Reward grip force only for valid grasps
        15 * valid_grasp * np.tanh(5 * (mug_pos[2] - table_height))  # Lifting reward
    )
    
    # 4. Anti-top-contact penalty
    top_contact_penalty = -10 * (grasp_state == 2) * np.exp(-50 * max(0, pad_height_above_block)**2)
    
    # 5. Placement rewards
    d_place = np.linalg.norm(mug_pos - target_pos)
    placement_reward = -3 * d_place + 25 * np.exp(-100 * d_place**2)
    if d_place < 0.05 and valid_grasp:
        placement_reward += 30
    
    # 6. Penalties
    penalties = (
        -100 * get_self_collision(self.model, self.data, self.collision_cache) +
        -50 * get_table_collision(self.model, self.data, self.collision_cache) +
        -20 * get_mug_toppled(self.model, self.data) +
        top_contact_penalty
    )
    
    # ---- COMPOSITE REWARD ----
    reward = (
        descent_reward +
        alignment_reward +
        grasp_reward +
        placement_reward +
        penalties
    )
    
    # Debug output
    if self.t % 50 == 0:
        print(f"t: {self.t:4d} | R: {reward:6.2f} | "
              f"Desc: {descent_reward:5.2f} | "
              f"Align: {alignment_reward:5.2f} | "
              f"Grasp: {grasp_reward:5.2f} | "
              f"Place: {placement_reward:5.2f} | "
              f"Pen: {penalties:5.2f} | "
              f"PadH: {pad_height_above_block:.3f} | "
              f"ValidG: {int(valid_grasp)}")
    
    return reward