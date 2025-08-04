import gymnasium as gym
from stable_baselines3 import PPO
import torch
import numpy as np
import register_envs # Import your new registration file
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import yaml
from gymnasium.wrappers import FrameStackObservation


def run_trained_bc_policy(policy_path):
    """
    Loads and runs a trained policy.

    This function bypasses `reconstruct_policy` to use `torch.load` with
    `weights_only=False`, which is necessary for PyTorch 2.6+ when loading
    policy files that contain more than just model weights (e.g., environment specs).
    This is safe because we trust the source of the policy file.
    """
    env = gym.make(
        id=f"gymnasium_env/imitation_{agent_mode}-v0",
        render_mode="human"
    )
    
    # 1. Create a policy with the same architecture as the trained one.
    # We do this by creating a dummy PPO agent and extracting its policy.
    policy_kwargs = dict(net_arch=net_arch)
    
    if feature_encoder == "transformer":
        from gymnasium_src.feature_extractors.transformer import TransformerFeatureExtractor
        env = FrameStackObservation(env, stack_size=history_len)
        
        policy_kwargs.update(dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256), # Output dimension of the transformer
        ))
        
    policy = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs).policy

    # 2. Load the saved data dictionary.
    # `weights_only=False` is required due to PyTorch 2.6+ security updates.
    # This is safe because we trust the source of the policy file.
    saved_data = torch.load(policy_path, weights_only=False)

    # 3. Load the weights from the dictionary into the new policy.
    policy.load_state_dict(saved_data["state_dict"])
    policy.eval()  # Set the policy to evaluation mode
    
    obs, _ = env.reset()
    while True:
        # Use the loaded policy to predict actions
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    with open("gymnasium_src/config/config_bc.yml", "r") as f:  yml = yaml.safe_load(f)    
    agent_mode = yml["agent_mode"]
    hyperparameters = yml["hyperparameters"]
    net_arch = hyperparameters["net_arch"]
    feature_encoder = hyperparameters.get("feature_encoder")
    history_len = hyperparameters.get("history_len")
    
    policy_fpath = f"policies/imitation_rl_policies/bc_policy_{agent_mode}.zip"
    run_trained_bc_policy(policy_fpath)