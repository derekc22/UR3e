import gymnasium as gym
import yaml
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import register_envs

def run_trained_sac_policy(policy_fpath, vecnormalize_fpath):
    """
    Loads and runs a trained SAC policy.
    """

    env_kwargs = {}
    if feature_encoder == "transformer":
        env_kwargs.update(dict(
            wrapper_class=FrameStackObservation,
            wrapper_kwargs=dict(stack_size=history_len)
        ))
            
    # Create a single, vectorized environment
    env = make_vec_env(
        env_id=f"gymnasium_env/ur3e-v2",
        n_envs=1,
        env_kwargs={"render_mode": "human"},
        **env_kwargs
    )

    # Load the saved statistics and wrap the environment
    env = VecNormalize.load(vecnormalize_fpath, env)
    env.training = False # Set to evaluation mode

    # Load the trained agent
    model = SAC.load(policy_fpath, env=env)

    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _ = env.step(action)
        print(f"Reward: {reward[0]:.2f}")

        # In a vectorized environment, 'terminated' is a list.
        if terminated[0]:
            obs = env.reset()


if __name__ == "__main__":
    with open("gymnasium_src/config/config_sac.yml", "r") as f: yml = yaml.safe_load(f)    
    action_mode = yml["action_mode"]
    hyperparameters = yml["hyperparameters"]
    feature_encoder = hyperparameters.get("feature_encoder")
    history_len = hyperparameters.get("history_len")
    
    # Define paths
    policy_fpath = f"policies/rl_policies/sac_policy_{action_mode}.zip"
    vecnormalize_fpath = f"policies/rl_policies/sac_vecnormalize_{action_mode}.pkl"
    run_trained_sac_policy(policy_fpath, vecnormalize_fpath)
