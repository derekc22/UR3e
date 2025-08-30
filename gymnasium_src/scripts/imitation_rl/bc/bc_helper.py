"""data inputs"""
hold = 100
num_points = 15
down_sample = 1

"""bc inputs"""
batch_size = 15

################################################################
desired_runtime_in_min = 0.5
batches_per_sec_at_desired_batch_size_history_len_and_net_arch = 65
################################################################

# choose one to fix, the other will be spit out
n_epochs = 1
num_demos = 25


"""derived"""
num_transitions_per_traj = ((hold * num_points * 7)/down_sample)

"""calculations"""
total_num_transitions = num_transitions_per_traj * num_demos * n_epochs
runtime_in_hrs = (total_num_transitions/batch_size) / (batches_per_sec_at_desired_batch_size_history_len_and_net_arch * 3600)
print("runtime in hrs: ", runtime_in_hrs)

num_demos_to_collect = (desired_runtime_in_min / 60) * (batches_per_sec_at_desired_batch_size_history_len_and_net_arch * 3600) * batch_size / (num_transitions_per_traj * n_epochs)
print("num demos to collect: ", num_demos_to_collect)

n_epochs_to_run = (desired_runtime_in_min / 60) * (batches_per_sec_at_desired_batch_size_history_len_and_net_arch * 3600) * batch_size / (num_transitions_per_traj * num_demos)
print("num epochs to run: ", n_epochs_to_run)




