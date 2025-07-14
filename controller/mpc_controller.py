import numpy as np
import casadi as ca
import mujoco
from scipy.spatial.transform import Rotation as R

from controller.controller_utils import (
    get_xpos,
    get_xrot,
    grip_ctrl,
)


class MPCController:
    """Task‑space Model Predictive Controller for the UR3e‑2F85 platform.

    The controller tracks a six‑degree‑of‑freedom end‑effector pose reference
    while respecting joint velocity limits. A quadratic program is solved at
    every control step using CasADi. The optimisation variables are joint
    velocities over the prediction horizon. Gravity compensation is applied
    after mapping the first optimised velocity to a joint‑torque command.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded MuJoCo model.
    horizon : int, default 20
        Prediction horizon length (number of discrete steps).
    q_weight : np.ndarray, shape (6,)
        Diagonal weights for position and orientation error [m, rad].
    r_weight : np.ndarray, shape (6,)
        Diagonal weights for joint velocity effort [rad/s].
    vel_limits : float or np.ndarray, default np.deg2rad(90)
        Symmetric joint‑velocity limits applied as |q̇| ≤ vel_limits.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        horizon: int = 20,
        q_weight: np.ndarray | None = None,
        r_weight: np.ndarray | None = None,
        vel_limits: float | np.ndarray = np.deg2rad(90),
    ) -> None:
        self.m = model
        self.N = horizon
        self.dt = float(model.opt.timestep)

        self.Q = np.diag(q_weight if q_weight is not None else [200, 200, 200, 50, 50, 50])
        self.R = np.diag(r_weight if r_weight is not None else [1, 1, 1, 1, 1, 1])

        if np.isscalar(vel_limits):
            vel_limits = np.full(6, vel_limits)
        self.vel_limits = np.asarray(vel_limits, dtype=float)

    # ------------------------------------------------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------------------------------------------------

    def __call__(
        self,
        d: mujoco.MjData,
        traj_t: np.ndarray,
        site: str = "right_pad1_site",
    ) -> np.ndarray:
        """Compute a single control action.

        Parameters
        ----------
        d : mujoco.MjData
            Current MuJoCo data.
        traj_t : np.ndarray, shape (7,)
            Pose target [x, y, z, rx, ry, rz, g] for the current time step.
        site : str, default "right_pad1_site"
            End‑effector site used for kinematics.

        Returns
        -------
        np.ndarray, shape (7,)
            Joint torques for the six arm actuators plus gripper command.
        """
        # Current EE pose ------------------------------------------------------------------
        site_id, xpos = get_xpos(self.m, d, site)
        _, xrot_mat = get_xrot(self.m, d, site)
        xrot_vec = R.from_matrix(xrot_mat.reshape(3, 3)).as_rotvec()

        x_current = np.hstack([xpos, xrot_vec])  # (6,)
        x_ref = traj_t[:6]

        # Jacobian (6 × 6) for the arm ------------------------------------------------------
        jac_full = np.zeros((6, self.m.nv))
        mujoco.mj_jacSite(self.m, d, jac_full[:3], jac_full[3:], site_id)
        J = jac_full[:, :6]

        # Build and solve QP ---------------------------------------------------------------
        u_opt = self._solve_mpc(x_current, x_ref, J)

        # Map desired joint velocity to torque --------------------------------------------
        tau = self._velocity_to_torque(d, u_opt)

        # Gripper feed‑forward -------------------------------------------------------------
        grip = grip_ctrl(self.m, traj_t[-1])

        return np.hstack([tau, grip])

    # ------------------------------------------------------------------------------------------------------------------
    # MPC formulation and solver
    # ------------------------------------------------------------------------------------------------------------------

    def _solve_mpc(self, x0: np.ndarray, x_ref: np.ndarray, J: np.ndarray) -> np.ndarray:
        """Linear MPC with fixed Jacobian over the horizon."""
        opti = ca.Opti()

        X = opti.variable(6, self.N + 1)
        U = opti.variable(6, self.N)

        opti.subject_to(X[:, 0] == x0)

        for k in range(self.N):
            # Discrete‑time linearised kinematics
            opti.subject_to(X[:, k + 1] == X[:, k] + J @ U[:, k] * self.dt)
            # Velocity limits
            opti.subject_to(ca.fabs(U[:, k]) <= self.vel_limits)

        # Objective -----------------------------------------------------------------------
        cost = 0
        for k in range(self.N):
            e = X[:, k] - x_ref
            cost += ca.mtimes([e.T, self.Q, e]) + ca.mtimes([U[:, k].T, self.R, U[:, k]])
        # Terminal cost
        e_terminal = X[:, self.N] - x_ref
        cost += ca.mtimes([e_terminal.T, self.Q, e_terminal])

        opti.minimize(cost)
        opti.solver("qrqp", {"print_time": False})

        sol = opti.solve()
        return sol.value(U)[:, 0]  # First control action

    # ------------------------------------------------------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------------------------------------------------------

    def _velocity_to_torque(self, d: mujoco.MjData, qdot_des: np.ndarray) -> np.ndarray:
        """Inverse dynamics mapping using an acceleration estimate."""
        nv = self.m.nv
        M_full = np.zeros((nv, nv))
        mujoco.mj_fullM(self.m, M_full, d.qM)
        M_arm = M_full[:6, :6]

        qdot_curr = d.qvel[:6]
        qddot = (qdot_des - qdot_curr) / self.dt

        tau = M_arm @ qddot + d.qfrc_bias[:6]
        return tau
