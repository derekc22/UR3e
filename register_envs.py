from gymnasium.envs.registration import register

# 1. Standard Gymnasium RL Environment
register(
    id="gymnasium_env/ur3e-v0",
    entry_point="gymnasium_env.envs.ur3e_env:UR3eEnv",
)

# 2. Imitation Learning Environment with Indirect Control (PID)
register(
    id="gymnasium_env/imitation_indirect-v0",
    entry_point="gymnasium_env.envs.imitation_env_indirect:ImitationEnvIndirect",
)

# 3. Imitation Learning Environment with Direct Torque Control
register(
    id="gymnasium_env/imitation_direct-v0",
    entry_point="gymnasium_env.envs.imitation_env_direct:ImitationEnvDirect",
)

# 4. Standard Gymnasium RL Environment for Hindsight Experience Replay (HER) with Actor-Critic 
register(
    id="gymnasium_env/ur3e-v2",
    entry_point="gymnasium_env.envs.ur3e_env2:UR3eEnv2",
)