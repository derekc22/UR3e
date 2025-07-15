# import gymnasium as gym
# from gymnasium.envs.registration import register

# register(
#     id="gymnasium_env/ur3e",
#     entry_point="gymnasium_env.envs:UR3eEnv"
# )
# env = gym.make("gymnasium_env/ur3e", render_mode="human")

# obs, info = env.reset()
# for _ in range(1000):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()
# env.close()


import gymnasium as gym
from gymnasium.envs.registration import register
import time

register(
    id="gymnasium_env/ur3e-v0",
    entry_point="gymnasium_env.envs.ur3e_env:UR3eEnv"
)

# env = gym.make("gymnasium_env/ur3e-v0", render_mode="human")
env = gym.make("gymnasium_env/ur3e-v0", render_mode="rgb_array")
print(env.metadata)


obs, info = env.reset()
try:
    for _ in range(100000000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # print("action shape: ", action.shape)
        # print("obs shape: ", obs.shape)
        print(reward)
        if terminated or truncated:
            obs, info = env.reset()
except KeyboardInterrupt:
    pass
finally:
    env.close()