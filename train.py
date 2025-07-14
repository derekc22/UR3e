import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Register environment
register(
    id="gymnasium_env/ur3e-v0",
    entry_point="gymnasium_env.envs.ur3e_env:UR3eEnv"
)

# Create vectorized environment
# env = make_vec_env(
#     "gymnasium_env/ur3e-v0",
#     n_envs=4,
#     env_kwargs={"render_mode": None},
#     vec_env_cls=DummyVecEnv
# )
env = make_vec_env(
    "gymnasium_env/ur3e-v0",
    n_envs=1,
    # env_kwargs={"render_mode": None},
    env_kwargs={"render_mode": "human"},
    vec_env_cls=DummyVecEnv
)

# Evaluation callback
eval_callback = EvalCallback(
    env,
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ur3e_tensorboard/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)

# Train the model
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback
)

# Save the model
model.save("ur3e_pickplace_model")