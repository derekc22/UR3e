import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicRewardNet
from imitation_env.envs import ImitationEnv
from imitation_env.collect_demonstrations import load_demonstrations
from gymnasium.envs.registration import register

def train_gail(expert_trajs):
    """
    Trains a GAIL agent.

    Args:
        expert_trajs: A list of expert trajectories.
    """
    # Register the custom environment
    register(
        id="imitation_env/ur3e-v0",
        entry_point="envs.imitation_env:ImitationEnv" #
    )

    # Create the vectorized environment
    venv = DummyVecEnv([lambda: ImitationEnv()])

    # Convert trajectories to transitions, the format required by the GAIL trainer
    transitions = rollout.flatten_trajectories(expert_trajs) #

    # Initialize the generator, which is a PPO policy
    generator = PPO(
        "MlpPolicy",
        venv,
        n_steps=1024,
    )

    # The discriminator network that GAIL will train to distinguish expert from generator trajectories
    discriminator = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=False,
    )

    # Initialize the GAIL trainer
    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=256,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=generator,
        reward_net=discriminator,
    )

    # Train the GAIL agent
    gail_trainer.train(total_timesteps=40000)

    # Save the trained policy
    gail_trainer.policy.save("policies/imitation_env_policies/gail_policy.zip")
    print("Saved GAIL policy to policies/imitation_env_policies/gail_policy.zip")

    return gail_trainer.policy

if __name__ == "__main__":
    # Load the expert demonstrations
    expert_trajectories = load_demonstrations() #

    # Train the GAIL agent
    trained_policy = train_gail(expert_trajectories)