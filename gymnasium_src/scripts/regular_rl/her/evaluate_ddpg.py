import gymnasium as gym
from stable_baselines3 import DDPG
import yaml
import register_envs 

def run_trained_ddpg_policy(policy_fpath):
    """
    Loads and runs a trained DDPG policy.
    """
    env = gym.make(
        id="gymnasium_env/RL-v0",
        render_mode="human"
    )
    
    # Load the trained agent
    # Stable-Baselines3 handles policy reconstruction automatically
    print(f"Loading policy from {policy_fpath}...")
    model = DDPG.load(policy_fpath, env=env)
    print("Policy loaded successfully.")

    obs, _ = env.reset()
    while True:
        # Use the loaded model to predict actions
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        print(f"Reward: {reward:.2f}")

        if terminated or truncated:
            print("Episode finished. Resetting environment.")
            obs, _ = env.reset()

if __name__ == "__main__":
    with open("config_ddpg.yml", "r") as f: yml = yaml.safe_load(f)    
    agent_mode = yml["agent_mode"]
    
    policy_fpath = f"policies/rl_policies/ddpg_policy_{agent_mode}.zip"
    run_trained_ddpg_policy(policy_fpath)