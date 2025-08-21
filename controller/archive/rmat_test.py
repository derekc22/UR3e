import numpy as np
from controller.controller_func import R_to_euler, euler_to_R
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf, suppress=True)

# traj = np.genfromtxt('mujoco/trajectory.csv', delimiter=',', skip_header=1).reshape(-1, 7)
# euler_angles = traj[:, 3:6]

euler_angles = np.zeros(shape=(5, 3))


T = euler_angles.shape[0]

rot_matrix = np.apply_along_axis(
    euler_to_R, 
    axis=1, arr=euler_angles
).reshape(T, 3, 3)



euler_angles2 = np.array([R_to_euler(R) for R in rot_matrix])

print(euler_angles2.shape)

for i, j in zip(euler_angles, euler_angles2):
    print(i, j, i-j)


