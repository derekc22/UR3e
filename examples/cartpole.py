import gymnasium as gym
import numpy as np
from PIL import Image

# Load a Gymnasium environment with image rendering
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Reset to start the episode
obs, _ = env.reset()

# Take one step
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)

# Get the rendered image (not shown to screen)
image_array = env.render()

# Save the image to disk
img = Image.fromarray(image_array)
img.save("./examples/cartpole_frame.png")

env.close()
