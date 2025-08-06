import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicRewardNet
from gymnasium_src.scripts.imitation_rl.collect_demos import load_demos
import register_envs # Import your new registration file
from stable_baselines3.common.env_util import make_vec_env
import yaml
from gymnasium.wrappers import FrameStack


def train_airl(expert_trajs):
    """
    Trains an AIRL agent.

    Args:
        expert_trajs: A list of expert trajectories.
    """
    # Register the custom environment
    venv = make_vec_env(
        env_id=f"gymnasium_env/imitation_{agent_mode}-v0",
        n_envs=n_envs,
        env_kwargs={"render_mode": "rgb_array"},
        # env_kwargs={"render_mode": "human"},
        vec_env_cls=SubprocVecEnv,
        wrapper_class=FrameStack,
        wrapper_kwargs=dict(num_stack=history_len)
    )
    
    # Create the vectorized environment
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=clip_obs)

    # Convert trajectories to transitions, which is the format required by the AIRL trainer
    transitions = rollout.flatten_trajectories(expert_trajs)

    # Initialize the generator, which is a PPO policy    
    policy_kwargs = dict(net_arch=net_arch)
    generator = PPO("MlpPolicy", venv, n_steps=n_steps, policy_kwargs=policy_kwargs)

    # The reward network that AIRL will learn
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=False,
        hid_sizes=hid_sizes,
    )

    # Initialize the AIRL trainer
    airl_trainer = AIRL(
        demonstrations=transitions,
        demo_batch_size=demo_batch_size,
        gen_replay_buffer_capacity=gen_replay_buffer_capacity,
        n_disc_updates_per_round=n_disc_updates_per_round,
        venv=venv,
        gen_algo=generator,
        reward_net=reward_net,
    )

    # Train the AIRL agent
    airl_trainer.train(total_timesteps=total_timesteps)

    # Save the trained policy and reward network
    save_dir = "policies/imitation_rl_policies"
    airl_trainer.policy.save(f"{save_dir}/airl_policy_{agent_mode}.zip")
    torch.save(airl_trainer.reward_train.state_dict(), f"{save_dir}/airl_reward_net_{agent_mode}.pt")
    print(f"Saved AIRL policy to {save_dir}/airl_policy_{agent_mode}.zip")
    print(f"Saved AIRL reward network to {save_dir}/airl_reward_net_{agent_mode}.pt")

    return airl_trainer.policy

if __name__ == "__main__":
    with open("gymnasium_src/config/config_airl.yml", "r") as f:  yml = yaml.safe_load(f)    
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
    history_len = hyperparameters.get("history_len")
    
    # Load the expert demonstrations
    demos_fpath = f"gymnasium_src/demos/expert_demos_{agent_mode}.pkl"
    expert_trajectories = load_demos(demos_fpath)

    # Train the AIRL agent
    trained_policy = train_airl(expert_trajectories)