import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("ur3e_pickplace_model")

env = gym.make(
    "gymnasium_env/ur3e-v0",
    render_mode="human"
)

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    
    if terminated or truncated:
        print("Episode completed!")
        obs, _ = env.reset()

env.close()