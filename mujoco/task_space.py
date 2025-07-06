# def ctrl(t, m, d, traj_i):
#     site_name = "right_pad1_site"
#     site_id, current_pos = get_xpos(m, d, site_name)
#     _, current_rot_mat = get_xrot(m, d, site_name)
#     current_rot = current_rot_mat.reshape(3, 3)
    
#     # Compute full geometric Jacobian (6x6 for arm joints)
#     jac = np.zeros((6, m.nv))
#     mujoco.mj_jacSite(m, d, jac[:3], jac[3:], site_id)
#     jac_arm = jac[:, :6]
    
#     # Position error
#     e_pos = traj_i[:3] - current_pos
    
#     # Orientation error (axis-angle)
#     target_rot = traj_i[3:12].reshape(3, 3)
#     R_err = target_rot @ current_rot.T
#     skew = 0.5 * (R_err - R_err.T)
#     e_rot = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    
#     # Task-space PD force
#     F_pos = pos_gains["kp"] * e_pos - pos_gains["kd"] * (jac_arm[:3] @ d.qvel[:6])
#     F_rot = rot_gains["kp"] * e_rot - rot_gains["kd"] * (jac_arm[3:] @ d.qvel[:6])
#     F = np.concatenate([F_pos, F_rot])
    
#     # Compute joint torques (gravity compensation + task force)
#     tau_arm = jac_arm.T @ F + d.qfrc_bias[:6]
#     grip_u = grip_ctrl(m, traj_i[-1])
    
#     return np.hstack([tau_arm, grip_u])