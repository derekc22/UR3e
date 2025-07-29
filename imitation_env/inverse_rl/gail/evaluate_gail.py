# import torch
# from stable_baselines3 import PPO
# from imitation_env.envs import ImitationEnv

# def run_trained_gail_policy(policy_path="policies/imitation_env_policies/gail_policy.zip"):
#     """
#     Loads and runs a trained policy from GAIL.

#     Args:
#         policy_path (str): The path to the saved policy file.
#     """
#     # Create the environment with rendering enabled
#     env = ImitationEnv(render_mode="human") #

#     # Load the trained policy
#     policy = PPO.load(policy_path, env)

#     # Reset the environment to get the initial observation
#     obs, _ = env.reset() #

#     # Loop indefinitely to run the policy
#     while True:
#         # Get the action from the policy
#         action, _ = policy.predict(obs, deterministic=True) #

#         # Take the action in the environment
#         obs, _, terminated, truncated, _ = env.step(action) #

#         # If the episode is over, reset the environment
#         if terminated or truncated:
#             obs, _ = env.reset() #

# if __name__ == "__main__":
#     run_trained_gail_policy()



import torch
from stable_baselines3 import PPO
from imitation_env.envs import ImitationEnv

def run_trained_gail_policy(policy_path="policies/imitation_env_policies/gail_policy.zip"):
    """
    Loads and runs a trained policy.

    This function bypasses `reconstruct_policy` to use `torch.load` with
    `weights_only=False`, which is necessary for PyTorch 2.6+ when loading
    policy files that contain more than just model weights (e.g., environment specs).
    This is safe because we trust the source of the policy file.
    """
    env = ImitationEnv(render_mode="human")
    
    # 1. Create a policy with the same architecture as the trained one.
    # We do this by creating a dummy PPO agent and extracting its policy.
    policy = PPO("MlpPolicy", env).policy

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
    run_trained_gail_policy()