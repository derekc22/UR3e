import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np
import yaml
import os
import register_envs
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv


def train_ddpg():
    
    venv_kwargs = {}   
    policy_kwargs = dict(net_arch=net_arch)
    
    if feature_encoder == "transformer":
        venv_kwargs.update(dict(        
            wrapper_class=FrameStackObservation,
            wrapper_kwargs=dict(stack_size=history_len)
        ))
        
        from gymnasium_src.feature_extractors.transformer import TransformerFeatureExtractor
        policy_kwargs.update(dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256), # Output dimension of the transformer
        ))
    
    # Create the custom environment
    venv = make_vec_env(
        env_id=f"gymnasium_env/ur3e-v2",
        n_envs=n_envs,
        env_kwargs={"render_mode": "rgb_array"},
        # env_kwargs={"render_mode": "human"},
        vec_env_cls=SubprocVecEnv,
        **venv_kwargs
    )

    # Create the vectorized environment
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=clip_obs)

    # Action noise
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.array(action_noise_mean), 
        sigma=np.array(action_noise_sigma)
    )

    # Define the model
    model = DDPG(
        "MlpPolicy", 
        venv,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=1,
        device=device
    )

    save_dir = "policies/rl_policies"
    policy_fpath = f"{save_dir}/ddpg_policy.zip"
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(policy_fpath) and resume_training:
        print(f"Loading existing policy from {policy_fpath}")
        model.set_parameters(policy_fpath)

    # Train the model
    model.learn(total_timesteps=total_timesteps, log_interval=10)

    # Save the final model
    model.save(policy_fpath)
    print(f"Saved DDPG policy to {policy_fpath}")

if __name__ == "__main__":
    with open("gymnasium_src/config/config_ddpg.yml", "r") as f: yml = yaml.safe_load(f)    
    
    # Extract general parameters
    device = yml["device"]
    resume_training = yml["resume_training"]
    n_envs = yml["n_envs"]
    
    hyperparameters = yml["hyperparameters"]
    clip_obs = hyperparameters["clip_obs"]
    net_arch = hyperparameters["net_arch"]
    learning_rate = float(hyperparameters["learning_rate"])
    buffer_size = hyperparameters["buffer_size"]
    learning_starts = hyperparameters["learning_starts"]
    batch_size = hyperparameters["batch_size"]
    tau = hyperparameters["tau"]
    gamma = hyperparameters["gamma"]
    train_freq = tuple(hyperparameters["train_freq"])
    gradient_steps = hyperparameters["gradient_steps"]
    total_timesteps = hyperparameters["total_timesteps"]
    feature_encoder = hyperparameters.get("feature_encoder")
    history_len = hyperparameters.get("history_len")
    
    action_noise_params = hyperparameters["action_noise"]
    action_noise_mean = action_noise_params["mean"]
    action_noise_sigma = action_noise_params["sigma"]
    
    train_ddpg()