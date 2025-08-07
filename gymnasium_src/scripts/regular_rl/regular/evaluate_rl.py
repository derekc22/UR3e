import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
import register_envs 



def run_trained_rl_policy():
    model = PPO.load("policies/rl_policies/8-sturdy-pick/ur3e_pickplace_model.zip")

    env = gym.make(
        "gymnasium_env/ur3e-v0",
        render_mode="human"
    )

    obs, _ = env.reset()
    for _ in range(1000000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            print("Episode completed!")
            obs, _ = env.reset()

    env.close()
    
if __name__ == "__main__":
    run_trained_rl_policy()