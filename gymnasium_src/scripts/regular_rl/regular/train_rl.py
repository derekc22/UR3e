import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import register_envs 
import yaml

def train_rl():
    # Create environment
    venv = make_vec_env(
        env_id=f"gymnasium_env/ur3e-v0",
        n_envs=n_envs,
        env_kwargs={"render_mode": "rgb_array"},
        # env_kwargs={"render_mode": "human"},
        vec_env_cls=SubprocVecEnv
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        venv,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        tensorboard_log="./ur3e_tensorboard/",
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef
    )

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )

    # Save the model
    model.save("policies/rl_policies")

if __name__ == "__main__":
    with open("gymnasium_src/config/config_rl.yml", "r") as f:  yml = yaml.safe_load(f)    
    n_envs = yml["n_envs"]
    hyperparameters = yml["hyperparameters"]
    eval_freq = hyperparameters["eval_freq"]
    learning_rate = float(hyperparameters["learning_rate"])
    n_steps = hyperparameters["n_steps"]
    batch_size = hyperparameters["batch_size"]
    n_epochs = hyperparameters["n_epochs"]
    gamma = hyperparameters["gamma"]
    gae_lambda = hyperparameters["gae_lambda"]
    clip_range = hyperparameters["clip_range"]
    ent_coef = hyperparameters["ent_coef"]
    total_timesteps = hyperparameters["total_timesteps"]
    
    train_rl()