import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicRewardNet
from gymnasium_src.scripts.imitation_rl.collect_demos import load_demos
from gymnasium.envs.registration import register
import register_envs # Import your new registration file
from stable_baselines3.common.env_util import make_vec_env
import yaml

def train_gail(expert_trajs):
    """
    Trains a GAIL agent.

    Args:
        expert_trajs: A list of expert trajectories.
    """

    # Create environment
    venv = make_vec_env(
        env_id=f"gymnasium_env/imitation_{agent_mode}-v0",
        n_envs=n_envs,
        env_kwargs={"render_mode": "rgb_array"},
        # env_kwargs={"render_mode": "human"},
        vec_env_cls=SubprocVecEnv
    )
    
    # Create the vectorized environment
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=clip_obs)

    # Convert trajectories to transitions, the format required by the GAIL trainer
    transitions = rollout.flatten_trajectories(expert_trajs) #

    # Initialize the generator, which is a PPO policy
    policy_kwargs = dict(net_arch=net_arch)
    generator = PPO("MlpPolicy", venv, n_steps=n_steps, policy_kwargs=policy_kwargs)

    # The discriminator network that GAIL will train to distinguish expert from generator trajectories
    discriminator = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=False,
        hid_sizes=hid_sizes,
    )

    # Initialize the GAIL trainer
    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=demo_batch_size,
        gen_replay_buffer_capacity=gen_replay_buffer_capacity,
        n_disc_updates_per_round=n_disc_updates_per_round,
        venv=venv,
        gen_algo=generator,
        reward_net=discriminator,
    )

    # Train the GAIL agent
    gail_trainer.train(total_timesteps=total_timesteps)

    # Save the trained policy
    save_dir = "policies/imitation_rl_policies"
    gail_trainer.policy.save(f"{save_dir}/gail_policy_{agent_mode}.zip")
    print(f"Saved GAIL policy to {save_dir}/gail_policy_{agent_mode}.zip")

    return gail_trainer.policy

if __name__ == "__main__":
    with open("gymnasium_src/config/config_gail.yml", "r") as f:  yml = yaml.safe_load(f)    
    agent_mode = yml["agent_mode"]
    device = yml["device"]
    n_envs = yml["n_envs"]
    hyperparameters = yml["hyperparameters"]
    clip_obs = hyperparameters["clip_obs"]
    net_arch = hyperparameters["net_arch"]
    n_steps = hyperparameters["n_steps"]
    hid_sizes = hyperparameters["hid_sizes"]
    demo_batch_size = hyperparameters["demo_batch_size"]
    gen_replay_buffer_capacity = hyperparameters["gen_replay_buffer_capacity"]
    n_disc_updates_per_round = hyperparameters["n_disc_updates_per_round"]
    total_timesteps = hyperparameters["total_timesteps"]
    
    # Load the expert demonstrations
    demos_fpath = f"gymnasium_src/demos/expert_demos_{agent_mode}.pkl"
    expert_trajectories = load_demos(demos_fpath)

    # Train the GAIL agent
    trained_policy = train_gail(expert_trajectories)