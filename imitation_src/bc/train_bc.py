import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import bc
from imitation.data import rollout
from imitation_env_direct.envs import ImitationEnv
from imitation_src.collect_demonstrations import load_demonstrations
# from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import torch
import register_envs # Import your new registration file
from stable_baselines3.common.env_util import make_vec_env


def train_behavioral_cloning(expert_trajs, mode):
    # Register environment
    # register(
    #     id="imitation_env/ur3e-v0",
    #     entry_point=f"envs.imitation_env_{mode}:ImitationEnv"
    # )

    # Create environment
    venv = make_vec_env(
        env_id=f"imitation_env/{mode}-v0",
        n_envs=1,
        env_kwargs={"render_mode": "rgb_array"},
        # env_kwargs={"render_mode": "human"},
        vec_env_cls=DummyVecEnv
    )
    
    # Create the vectorized environment
    # venv = DummyVecEnv([lambda: ImitationEnv()])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.)


    # Convert trajectories to transitions
    transitions = rollout.flatten_trajectories(expert_trajs)
    
    # Initialize BC trainer
    # bc_trainer = bc.BC(
    #     observation_space=venv.observation_space,
    #     action_space=venv.action_space,
    #     demonstrations=transitions,
    #     policy=PPO("MlpPolicy", venv).policy,
    #     device="auto",
    #     rng=np.random.default_rng()  # Add this line
    # )
    
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        policy=PPO("MlpPolicy", venv).policy,
        device="auto",
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-4},
        rng=np.random.default_rng()
    )
    
    # Train BC
    bc_trainer.train(n_epochs=10)
    
    # Save trainer
    bc_trainer.policy.save(f"policies/imitation_env_policies/bc_policy_{mode}.zip")
    print(f"Saved BC trainer to policies/imitation_env_policies/bc_policy_{mode}.zip")
    
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
    mode = "direct" 
    
    # Load the expert demonstrations
    fpath = f"imitation_src/data/expert_demos_{mode}.pkl"
    expert_trajectories = load_demonstrations(fpath)

    # Train the BC agent
    trained_policy = train_behavioral_cloning(expert_trajectories, mode)
    
    # Evaluate
    # policy = bc_trainer.policy
    # evaluate_policy(policy)