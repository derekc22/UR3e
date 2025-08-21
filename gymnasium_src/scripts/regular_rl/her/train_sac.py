import os
import numpy as np
import yaml
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import register_envs


def train_sac():
    """
    Trains a SAC agent.
    """

    # Define file paths
    save_dir = "policies/rl_policies"
    os.makedirs(save_dir, exist_ok=True)
    policy_fpath = f"{save_dir}/sac_policy_{action_mode}.zip"
    vecnormalize_fpath = f"{save_dir}/sac_vecnormalize_{action_mode}.pkl"

    # Setup environment kwargs
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

    # Create the vectorized environment
    venv = make_vec_env(
        env_id=f"gymnasium_env/ur3e-v2",
        n_envs=n_envs,
        env_kwargs={"render_mode": "human" if visualize else "rgb_array"},
        vec_env_cls=SubprocVecEnv,
        **venv_kwargs
    )

    if resume_training and os.path.exists(policy_fpath):
        print("Loading existing policy and env stats to resume SAC training...")
        # 1. Load VecNormalize statistics
        venv = VecNormalize.load(vecnormalize_fpath, venv)

        # 2. Load SAC model
        model = SAC.load(policy_fpath, env=venv, device=device)

    else:
        print("Starting SAC training from scratch...")
        # Normalize observations
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=clip_obs)

        # Initialize SAC model
        model = SAC(
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
            ent_coef=ent_coef,
            policy_kwargs=dict(net_arch=net_arch),
            verbose=1,
            device=device
        )

    # Train the SAC agent
    model.learn(total_timesteps=total_timesteps, log_interval=10)

    # Save all components
    model.save(policy_fpath)
    venv.save(vecnormalize_fpath)

    print(f"Saved SAC policy to {policy_fpath}")
    print(f"Saved VecNormalize stats to {vecnormalize_fpath}")


if __name__ == "__main__":
    with open("gymnasium_src/config/config_sac.yml", "r") as f: yml = yaml.safe_load(f)    
    action_mode = yml["action_mode"]
    device = yml["device"]
    n_envs = yml["n_envs"]
    resume_training = yml["resume_training"]
    visualize = yml["visualize"]

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
    ent_coef = hyperparameters["ent_coef"]
    total_timesteps = hyperparameters["total_timesteps"]
    feature_encoder = hyperparameters.get("feature_encoder")
    history_len = hyperparameters.get("history_len")

    # Train SAC agent
    train_sac()