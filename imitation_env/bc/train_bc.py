import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import bc
from imitation.data import rollout
from imitation_env.envs import ImitationEnv
from imitation_env.collect_demonstrations import load_demonstrations
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

def train_behavioral_cloning(expert_trajs):
    # Register environment
    register(
        id="imitation_env/ur3e-v0",
        entry_point="imitation_env.envs.imitation_env:ImitationEnv"
    )    

    # Create environment
    venv = make_vec_env(
        "imitation_env/ur3e-v0",
        n_envs=5,
        env_kwargs={"render_mode": "human"},
        vec_env_cls=DummyVecEnv
    )

    # Create the vectorized environment
    # venv = DummyVecEnv([lambda: ImitationEnv()])
    

    # Convert trajectories to transitions
    transitions = rollout.flatten_trajectories(expert_trajs)
    
    # Initialize BC trainer
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        policy=PPO("MlpPolicy", venv).policy,
        device="auto",
        rng=np.random.default_rng()  # Add this line
    )
        
    # Train BC
    bc_trainer.train(n_epochs=10)
    
    # Save trainer
    bc_trainer.policy.save("policies/imitation_env_policies/bc_policy.zip")
    print("Saved BC trainer to policies/imitation_env_policies/bc_policy.zip")
    
    return bc_trainer

# def evaluate_policy(policy, num_episodes=10):
#     """Evaluate trained policy"""
#     env = ImitationEnv(render_mode="human")
#     total_success = 0
    
#     for i in range(num_episodes):
#         obs, _ = env.reset()
#         terminated = False
#         truncated = False
        
#         while not (terminated or truncated):
#             action, _ = policy.predict(obs, deterministic=True)
#             obs, _, terminated, truncated, _ = env.step(action)
        
#         if terminated:  # Success
#             total_success += 1
    
#     success_rate = total_success / num_episodes
#     print(f"Success rate: {success_rate:.2f}")
#     return success_rate

if __name__ == "__main__":
    # Load expert demonstrations
    expert_trajs = load_demonstrations()
    
    # Train Behavioral Cloning
    bc_trainer = train_behavioral_cloning(expert_trajs)
    
    # Evaluate
    # policy = bc_trainer.policy
    # evaluate_policy(policy)