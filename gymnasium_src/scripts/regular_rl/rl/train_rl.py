import os
import numpy as np
import yaml
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
# from stable_baselines3.common.callbacks import EvalCallback
import register_envs

def train_rl():
    """
    Trains a RL agent.
    """
    
    # Define file paths
    save_dir = "policies/rl_policies"
    os.makedirs(save_dir, exist_ok=True)
    policy_fpath = f"{save_dir}/rl_policy_{action_mode}.zip"
    vecnormalize_fpath = f"{save_dir}/rl_vecnormalize_{action_mode}.pkl"
    
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
        print("Loading existing policy and env stats to resume PPO training...")
        # 1. Load VecNormalize statistics
        venv = VecNormalize.load(vecnormalize_fpath, venv)

        # 2. Load PPO model
        model = PPO.load(policy_fpath, env=venv, device=device)

    else:
        print("Starting PPO training from scratch...")
        # Normalize observations
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=clip_obs)

        # Initialize PPO model
        model = PPO(
            "MlpPolicy",
            venv,
            tensorboard_log="./ur3e_tensorboard/",
            learning_rate=learning_rate,
            # n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            # clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=dict(net_arch=net_arch),
            verbose=1,
            device=device
        )
        
    # Evaluation callback
    # eval_callback = EvalCallback(
    #     venv,
    #     eval_freq=eval_freq,
    #     deterministic=True,
    #     render=False
    # )

    # Train the PPO agent
    model.learn(total_timesteps=total_timesteps, log_interval=10)

    # Save all components
    model.save(policy_fpath)
    venv.save(vecnormalize_fpath)

    print(f"Saved PPO policy to {policy_fpath}")
    print(f"Saved VecNormalize stats to {vecnormalize_fpath}")


if __name__ == "__main__":
    with open("gymnasium_src/config/config_rl.yml", "r") as f:  yml = yaml.safe_load(f)    
    action_mode = yml["action_mode"]
    device = yml["device"]
    n_envs = yml["n_envs"]
    resume_training = yml["resume_training"]
    visualize = yml["visualize"]
    

    # eval_freq = hyperparameters["eval_freq"]
    # n_steps = hyperparameters["n_steps"]
    # n_epochs = hyperparameters["n_epochs"]
    # gae_lambda = hyperparameters["gae_lambda"]
    # clip_range = hyperparameters["clip_range"]
        
    hyperparameters = yml["hyperparameters"]
    clip_obs = hyperparameters["clip_obs"]
    net_arch = hyperparameters["net_arch"]
    learning_rate = float(hyperparameters["learning_rate"])
    buffer_size = hyperparameters["buffer_size"]
    learning_starts = hyperparameters["learning_starts"]
    batch_size = hyperparameters["batch_size"]
    n_epochs = hyperparameters["n_epochs"]
    gamma = hyperparameters["gamma"]
    gae_lambda = hyperparameters["gae_lambda"]
    ent_coef = hyperparameters["ent_coef"]
    total_timesteps = hyperparameters["total_timesteps"]
    feature_encoder = hyperparameters.get("feature_encoder")
    history_len = hyperparameters.get("history_len")
    
    train_rl()