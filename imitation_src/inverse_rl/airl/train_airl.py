import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicRewardNet
from imitation_env_direct.envs import ImitationEnv
from imitation_src.collect_demonstrations import load_demonstrations
# from gymnasium.envs.registration import register
import register_envs # Import your new registration file
from stable_baselines3.common.env_util import make_vec_env


def train_airl(expert_trajs, mode):
    """
    Trains an AIRL agent.

    Args:
        expert_trajs: A list of expert trajectories.
    """
    # Register the custom environment
    venv = make_vec_env(
        env_id=f"imitation_env/{mode}-v0",
        n_envs=1,
        env_kwargs={"render_mode": "rgb_array"},
        # env_kwargs={"render_mode": "human"},
        vec_env_cls=DummyVecEnv
    )
    
    # Create the vectorized environment
    venv = DummyVecEnv([lambda: ImitationEnv()])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Convert trajectories to transitions, which is the format required by the AIRL trainer
    transitions = rollout.flatten_trajectories(expert_trajs)

    # Initialize the generator, which is a PPO policy
    generator = PPO(
        "MlpPolicy",
        venv,
        n_steps=4096,
    )

    # The reward network that AIRL will learn
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=False,
    )

    # Initialize the AIRL trainer
    airl_trainer = AIRL(
        demonstrations=transitions,
        demo_batch_size=256,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=40,
        venv=venv,
        gen_algo=generator,
        reward_net=reward_net,
    )

    # Train the AIRL agent
    airl_trainer.train(total_timesteps=40000)

    # Save the trained policy and reward network
    airl_trainer.policy.save(f"policies/imitation_env_policies/airl_policy_{mode}.zip")
    torch.save(airl_trainer.reward_train.state_dict(), f"policies/imitation_env_policies/airl_reward_net_{mode}.pt")
    print(f"Saved AIRL policy to policies/imitation_env_policies/airl_policy_{mode}.zip")
    print(f"Saved AIRL reward network to policies/imitation_env_policies/airl_reward_net_{mode}.pt")

    return airl_trainer.policy

if __name__ == "__main__":
    mode = "direct" 
    
    # Load the expert demonstrations
    fpath = f"imitation_src/data/expert_demos_{mode}.pkl"
    expert_trajectories = load_demonstrations(fpath)

    # Train the AIRL agent
    trained_policy = train_airl(expert_trajectories, mode)