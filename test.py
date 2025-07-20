import numpy as np
from controller.aux import *
from controller.build_traj import *
np.set_printoptions(
    linewidth=400,     # Wider output (default is 75)
    threshold=np.inf,  # Print entire array, no summarization with ...
    precision=8,       # Show more decimal places
    suppress=False     # Do not suppress small floating point numbers
)

# noise = np.hstack([
#     np.zeros(10-7),
#     np.random.uniform(low=-0.01, high=0.01, size=7)
# ])
# print(noise)

# trajectory_fpath = "controller/data/traj_l.csv"
# x = load_trajectory_file(trajectory_fpath)

# print(build_interpolated_trajectory(2, 3, trajectory_fpath))
# load_trajectory_file(trajectory_fpath)


t1 = [1, 2, 3, 4,  5,  6,  7]
t2 = [7, 8, 9, 10, 11, 12, 13]

print(interpolate(t1, t2))