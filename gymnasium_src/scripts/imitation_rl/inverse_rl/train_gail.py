import os
import numpy as np
import torch
import yaml
from gymnasium.wrappers import FrameStackObservation
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import register_envs
from gymnasium_src.scripts.imitation_rl.collect_demos import load_demos


def train_gail(expert_trajs):
    """
    Trains a GAIL agent.
    """
    
    # Define file paths
    save_dir = "policies/imitation_rl_policies"
    os.makedirs(save_dir, exist_ok=True)
    policy_fpath = f"{save_dir}/gail_policy_{imitation_mode}.zip"
    reward_net_path = f"{save_dir}/gail_reward_net_{imitation_mode}.pt"
    vecnormalize_fpath = f"{save_dir}/gail_vecnormalize_{imitation_mode}.pkl"

    # Setup environment kwargs
    venv_kwargs = {}
    policy_kwargs = dict(net_arch=net_arch)

    if feature_encoder == "transformer":
        venv_kwargs.update(dict(
            wrapper_class=FrameStackObservation,
            wrapper_kwargs=dict(stack_size=history_len)
        ))
        from gymnasium_src.feature_extractors.transformer import TransformerFeatureExtractor
        policy_kwargs.update(dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        ))
        from gymnasium_src.scripts.imitation_rl.collect_demos import stack_expert_trajectories
        expert_trajs = stack_expert_trajectories(expert_trajs, history_len)

    # Create the vectorized environment
    venv = make_vec_env(
        env_id=f"gymnasium_env/imitation_{imitation_mode}-v0",
        n_envs=n_envs,
        env_kwargs={"render_mode": "rgb_array"},
        vec_env_cls=SubprocVecEnv,
        **venv_kwargs
    )
    
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=False,
        hid_sizes=hid_sizes
    ).to(device)

    if resume_training and os.path.exists(policy_fpath):
        print("Loading existing policy, reward net, and env stats to resume GAIL training...")
        # 1. Load VecNormalize statistics
        venv = VecNormalize.load(vecnormalize_fpath, venv)

        # 2. Load PPO generator
        generator = PPO.load(policy_fpath, env=venv, device=device)

        # 3. Load Reward Network
        # state_dict = torch.load(reward_net_path)
        # unprefixed_state_dict = {k.removeprefix("base."): v for k, v in state_dict.items()}
        # reward_net.load_state_dict(unprefixed_state_dict)
        reward_net.load_state_dict(torch.load(reward_net_path))

    else:
        print("Starting GAIL training from scratch...")
        # Normalize observations
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=clip_obs)

        # Initialize PPO generator
        generator = PPO(
            "MlpPolicy",
            venv,
            n_steps=n_steps,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            device=device
        )

    # Convert expert trajectories to transitions
    transitions = rollout.flatten_trajectories(expert_trajs)

    # Initialize the GAIL trainer
    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=demo_batch_size,
        gen_replay_buffer_capacity=gen_replay_buffer_capacity,
        n_disc_updates_per_round=n_disc_updates_per_round,
        venv=venv,
        gen_algo=generator,
        reward_net=reward_net
    )

    # Train the GAIL agent
    gail_trainer.train(total_timesteps=total_timesteps)

    # Save all components
    generator.save(policy_fpath)
    # torch.save(gail_trainer.reward_train.state_dict(), reward_net_path)
    torch.save(gail_trainer._reward_net.state_dict(), reward_net_path)
    venv.save(vecnormalize_fpath)

    print(f"Saved GAIL policy to {policy_fpath}")
    print(f"Saved GAIL reward network to {reward_net_path}")
    print(f"Saved VecNormalize stats to {vecnormalize_fpath}")



if __name__ == "__main__":
    with open("gymnasium_src/config/config_gail.yml", "r") as f: yml = yaml.safe_load(f)
    imitation_mode = yml["imitation_mode"]
    device = yml["device"]
    n_envs = yml["n_envs"]
    resume_training = yml["resume_training"]

    hyperparameters = yml["hyperparameters"]
    clip_obs = hyperparameters["clip_obs"]
    net_arch = hyperparameters["net_arch"]
    n_steps = hyperparameters["n_steps"]
    hid_sizes = hyperparameters["hid_sizes"]
    batch_size = hyperparameters["batch_size"]
    demo_batch_size = hyperparameters["demo_batch_size"]
    gen_replay_buffer_capacity = hyperparameters["gen_replay_buffer_capacity"]
    n_disc_updates_per_round = hyperparameters["n_disc_updates_per_round"]
    total_timesteps = hyperparameters["total_timesteps"]
    feature_encoder = hyperparameters.get("feature_encoder")
    history_len = hyperparameters.get("history_len")

    # Load expert demonstrations
    demos_fpath = f"gymnasium_src/demos/expert_demos_{imitation_mode}.pkl"
    expert_trajectories = load_demos(demos_fpath)

    # Train GAIL agent
    train_gail(expert_trajectories)