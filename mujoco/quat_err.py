def sysid_loss_h_only(
                        params,
                        model,
                        data,
                        n_steps,
                        # data set
                        real_state_traj,
                        ctrl_seqs,
                        # augmented network
                        control_net,
                        ):

    n_batch = real_state_traj.shape[0]
    x0s = real_state_traj[:,0,:]
    real_state_traj = real_state_traj[:,1:,:] # remove the first state
    # print('real_state_traj:',real_state_traj.shape)
    state_traj = simulate_open_loop_h_only(
                                            # params to optimize
                                            params,
                                            # model and data
                                            model,
                                            data,
                                            # simulation params
                                            x0s,
                                            ctrl_seqs,
                                            n_steps,
                                            # models to augment
                                            control_net,
                                        )

    base_pos_traj = state_traj[:,:,jnp.array([0, 1, 2])] # N_batch x N_steps x base_pos
    ref_base_pos_traj = real_state_traj[:,:,jnp.array([0, 1, 2])] # N_batch x N_steps x base_pos

    base_quat_traj = state_traj[:,:,jnp.array([3, 4, 5, 6])] # N_batch x N_steps x base_quat
    ref_base_quat_traj = real_state_traj[:,:,jnp.array([3, 4, 5, 6])] # N_batch x N_steps x base_quat

    jpos_traj = state_traj[:,:, 7:model.nq+1] # N_batch x N_steps x base_jpos
    ref_jpos_traj = real_state_traj[:,:, 7:model.nq+1] # N_batch x N_steps x base_jpos

    base_tvel_traj = state_traj[:,:,jnp.array([model.nq+0, model.nq+1, model.nq+2 ])] # N_batch x N_steps x base_tvel
    ref_base_tvel_traj = real_state_traj[:,:,jnp.array([model.nq+0, model.nq+1, model.nq+2 ])] # N_batch x N_steps x base_tvel

    base_avel_traj = state_traj[:,:,jnp.array([model.nq+3, model.nq+4, model.nq+5 ])] # N_batch x N_steps x base_tvel
    ref_base_avel_traj = real_state_traj[:,:,jnp.array([model.nq+3, model.nq+4, model.nq+5 ])] # N_batch x N_steps x base_tvel
    
    jvel_traj = state_traj[:,:,model.nq+6:] # N_batch x N_steps x base_jvel
    ref_jvel_traj = real_state_traj[:,:,model.nq+6:] # N_batch x N_steps x base_jvel

    pos_error = jnp.square(jnp.linalg.norm(ref_base_pos_traj - base_pos_traj, axis=-1)) # norm along the state dim
    quat_inner = jnp.sum(ref_base_quat_traj * base_quat_traj, axis=-1)
    quat_error = 1.0 - jnp.square(quat_inner)
    jpos_error = jnp.square(jnp.linalg.norm(ref_jpos_traj - jpos_traj, axis=-1)) # norm along the state dim
    
    tvel_error = jnp.square(jnp.linalg.norm(ref_base_tvel_traj - base_tvel_traj, axis=-1)) # norm along the state dim
    avel_error = jnp.square(jnp.linalg.norm(ref_base_avel_traj - base_avel_traj, axis=-1)) # norm along the state dim
    jvel_error = jnp.square(jnp.linalg.norm(ref_jvel_traj - jvel_traj, axis=-1)) # norm along the state dim

    # compute the loss
    norm_squared = pos_error + quat_error + jpos_error + tvel_error + avel_error + jvel_error

    # sum = jnp.mean(norm_squared, axis=-1) # mean along the time steps (normalized sum
    sum = jnp.sum(norm_squared, axis=-1) # sum along the time steps
    # normalize 
    sum = sum / n_steps
    # sum = jnp.mean(sum, axis=-1) # mean along the batch
    sum = jnp.sum(sum, axis=-1) # sum along the batch
    # normalize 
    sum = sum / n_batch

    return sum