import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicRewardNet
from imitation_env.envs import ImitationEnv
from imitation_env.collect_demonstrations import load_demonstrations
from gymnasium.envs.registration import register

def train_airl(expert_trajs):
    """
    Trains an AIRL agent.

    Args:
        expert_trajs: A list of expert trajectories.
    """
    # Register the custom environment
    register(
        id="imitation_env/ur3e-v0",
        # entry_point="envs.imitation_env:ImitationEnv"
        entry_point="envs.imitation_env_direct:ImitationEnv"
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
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=generator,
        reward_net=reward_net,
    )

    # Train the AIRL agent
    airl_trainer.train(total_timesteps=4000000)

    # Save the trained policy and reward network
    airl_trainer.policy.save("policies/imitation_env_policies/airl_policy.zip")
    torch.save(
        airl_trainer.reward_train.state_dict(), "policies/imitation_env_policies/airl_reward_net.pt"
    )
    print("Saved AIRL policy to policies/imitation_env_policies/airl_policy.zip")
    print("Saved AIRL reward network to policies/imitation_env_policies/airl_reward_net.pt")

    return airl_trainer.policy

if __name__ == "__main__":
    # Load the expert demonstrations
    expert_trajectories = load_demonstrations()

    # Train the AIRL agent
    trained_policy = train_airl(expert_trajectories)