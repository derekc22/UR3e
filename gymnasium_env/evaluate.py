import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.envs.registration import register


register(
    id="gymnasium_env/ur3e-v0",
    entry_point="gymnasium_env.envs.ur3e_env:UR3eEnv"
)

model = PPO.load("policies/gymnasium_env_policies/8-sturdy-pick/ur3e_pickplace_model.zip")

env = gym.make(
    "gymnasium_env/ur3e-v0",
    render_mode="human"
)

obs, _ = env.reset()
for _ in range(1000000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    
    if terminated or truncated:
        print("Episode completed!")
        obs, _ = env.reset()

env.close()