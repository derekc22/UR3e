import mujoco
import numpy as np
import matplotlib
from utils import (
    load_model, reset,
    get_grasp_force, get_arm_qpos,
    pd_ctrl, grip_ctrl, update_errs
    )
from gen_traj import gen_traj_j
from mujoco.aux import build_trajectory, build_interpolated_trajectory
from aux import cleanup
matplotlib.use('Agg')  # Set backend to non-interactive
import yaml
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf)


    
def get_state(m, d):
    
    qpos_ur3e = get_arm_qpos(d)
    grip_2f85 = get_grasp_force(d)[0:1] #get_grip_ctrl(d)

    return np.concatenate([
        qpos_ur3e, grip_2f85
    ])


def ctrl(t, m, d, traj_t):
    qpos_u = pd_ctrl(t, m, d, traj_t[:-1], get_qpos_err, qpos_gains, joint_ranges, tot_qpos_joint_errs)    
    grip_u = grip_ctrl(m, traj_t[-1])
    
    return np.hstack([
        qpos_u,
        grip_u
    ])





def get_qpos_err(t, m, d, qpos_target):
    qpos_delta = qpos_target - get_arm_qpos(d)
    update_errs(t, qpos_errs, qpos_delta)
    return qpos_delta
    






model_path = "assets/ur3e_2f85.xml"
trajectory_fpath = "mujoco/data/traj_j.csv"
config_fpath = "mujoco/config/config_j.yml"
log_fpath = "mujoco/logs/logs_j/"
ctrl_mode = "j"
num_joints = 6

with open(config_fpath, "r") as f: yml = yaml.safe_load(f)
qpos_gains = { k:np.diag(v) for k, v in yml["qpos"].items() } 
hold = yml["hold"]
n = yml["n"]

# ur3e  = 6  nq, 6  nv, 6 nu
# 2f85  = 8  nq, 8  nv, 1 nu
# total = 14 nq, 14 nv, 7 nu
m, d = load_model(model_path)

gen_traj_j()
traj_target = build_interpolated_trajectory(n, hold, trajectory_fpath) if n else build_trajectory(hold, trajectory_fpath)
T = traj_target.shape[0]
traj_true = np.zeros_like(traj_target)

joint_ranges = m.jnt_range[:num_joints]
qpos_errs = np.zeros(shape=(T, num_joints))
tot_qpos_joint_errs = np.zeros(num_joints)



def main():
    
    viewer = mujoco.viewer.launch_passive(m, d)
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            
    reset(m, d)
    save_flag = True

    try:
        for t in range (T):
            viewer.sync()
    
            d.ctrl = ctrl(t, m, d, traj_target[t, :])
            mujoco.mj_step(m, d)
            
            traj_true[t] = get_state(m, d)
            
            # print(f"qpos_target: {traj_target[t, :7]}, pos_true: {traj_true[t, :7]}, pos_err: {qpos_errs[t, :]}")
            # print(f"grip_target: {traj_target[t, -1]}")
            # print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            
            # time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        save_flag = False 
        raise(e)
    finally: 
        viewer.close()
        if save_flag: cleanup(traj_target, traj_true, T, trajectory_fpath, log_fpath, yml, ctrl_mode, qpos_errs=qpos_errs)

        
        
    
        
    
    
    
if __name__ == "__main__":
    
    main()