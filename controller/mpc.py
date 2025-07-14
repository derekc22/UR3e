import mujoco
import numpy as np
import matplotlib

# Numerical and plotting defaults
np.set_printoptions(precision=3, linewidth=3000, threshold=np.inf)
matplotlib.use("Agg")  # Non‑interactive backend

from controller.controller_utils import load_model, reset, get_task_space_state, get_joint_torques
from gen_traj import gen_traj_l
from controller.aux import (
    build_trajectory,
    build_interpolated_trajectory,
    cleanup,
)
from mpc_controller import MPCController

import yaml


def main() -> None:
    """Runs the L‑shaped pick‑and‑place task controlled by the MPC."""

    model_path = "assets/ur3e_2f85.xml"
    trajectory_fpath = "controller/data/traj_l.csv"
    config_path = "controller/config/config_l_task.yml"
    log_fpath = "controller/logs/logs_l_task/"
    ctrl_mode = "l_task"

    # ------------------------------------------------------------------------------------------------------------------
    # Configuration and initialisation
    # ------------------------------------------------------------------------------------------------------------------

    with open(config_path, "r") as f:
        yml = yaml.safe_load(f)

    hold: int = yml["hold"]
    n: int = yml["n"]

    m, d = load_model(model_path)
    mpc = MPCController(m)

    # Build reference trajectory (uses your original generator and utilities)
    gen_traj_l()
    traj_target = (
        build_interpolated_trajectory(n, hold, trajectory_fpath)
        if n
        else build_trajectory(hold, trajectory_fpath)
    )
    T = traj_target.shape[0]

    # Pre‑allocate logs
    traj_true = np.zeros_like(traj_target)
    pos_errs = np.zeros((T, 3))  # Retained for downstream log compatibility
    rot_errs = np.zeros((T, 3))
    ctrls = np.zeros((T, m.nu))
    actuator_frc = np.zeros((T, m.nu))

    # Viewer
    viewer = mujoco.viewer.launch_passive(m, d)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

    reset(m, d)
    save_flag = True

    # ------------------------------------------------------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------------------------------------------------------

    try:
        for t in range(T):
            viewer.sync()

            # --- Model‑Predictive control --------------------------------------------------
            u = mpc(d, traj_target[t, :])
            d.ctrl = u
            ctrls[t] = u

            mujoco.mj_step(m, d)

            traj_true[t] = get_task_space_state(m, d)
            actuator_frc[t] = get_joint_torques(d)

    except KeyboardInterrupt:
        pass
    except Exception:
        save_flag = False
        raise
    finally:
        viewer.close()
        if save_flag:
            cleanup(
                traj_target,
                traj_true,
                ctrls,
                actuator_frc,
                trajectory_fpath,
                log_fpath,
                yml,
                ctrl_mode,
                pos_errs=pos_errs,
                rot_errs=rot_errs,
            )


if __name__ == "__main__":
    main()
