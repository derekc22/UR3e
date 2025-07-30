from gymnasium.envs.registration import register

# 1. Standard Gymnasium RL Environment
register(
    id="gymnasium_env/ur3e-v0",
    entry_point="gymnasium_env.envs.ur3e_env:UR3eEnv",
)

# 2. Imitation Learning Environment with Indirect Control (PID)
register(
    id="imitation_env/indirect-v0",
    entry_point="imitation_env_indirect.envs.imitation_env:ImitationEnvIndirect",
)

# 3. Imitation Learning Environment with Direct Torque Control
register(
    id="imitation_env/direct-v0",
    entry_point="imitation_env_direct.envs.imitation_env_direct:ImitationEnvDirect",
)