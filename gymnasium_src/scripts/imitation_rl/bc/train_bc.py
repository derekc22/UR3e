import os
import numpy as np
import torch
import yaml
from gymnasium.wrappers import FrameStackObservation
from imitation.algorithms import bc
from imitation.data import rollout
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import register_envs
from gymnasium_src.scripts.imitation_rl.collect_demos import load_demos


def train_bc(expert_trajs):
    """
    Trains a BC agent.
    """
    
    # Define file paths
    save_dir = "policies/imitation_rl_policies"
    os.makedirs(save_dir, exist_ok=True)
    policy_fpath = f"{save_dir}/bc_policy_{action_mode}.zip"

    # Setup environment kwargs
    env_kwargs = {}
    policy_kwargs = dict(net_arch=net_arch)

    if feature_encoder == "transformer":
        env_kwargs.update(dict(
            wrapper_class=FrameStackObservation,
            wrapper_kwargs=dict(stack_size=history_len)
        ))
        from gymnasium_src.feature_extractors.transformer import TransformerFeatureExtractor
        policy_kwargs.update(dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256), # Output dimension of the transformer
        ))
        from gymnasium_src.scripts.imitation_rl.collect_demos import stack_expert_trajectories
        expert_trajs = stack_expert_trajectories(expert_trajs, history_len)

    # Create a single, vectorized environment
    env = make_vec_env(
        env_id=f"gymnasium_env/imitation_{action_mode}-v0",
        n_envs=1,
        env_kwargs={"render_mode": "rgb_array"},
        **env_kwargs
    )

    if resume_training and os.path.exists(policy_fpath):
        print(f"Loading existing policy and env stats to resume BC training...")
        # Load PPO agent
        agent = PPO.load(policy_fpath, env=env, device=device)

    else:
        print("Starting BC training from scratch...")
        # Initialize PPO agent
        agent = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            device=device
        )

    # Convert expert trajectories to transitions
    transitions = rollout.flatten_trajectories(expert_trajs)
    print(f"Number of transitions: {len(transitions)}. Total number of batches: {int(len(transitions)/batch_size)*n_epochs}")

    # Initialize the BC trainer
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=agent.policy,
        device=torch.device(device),
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": optimizer_lr},
        rng=np.random.default_rng(),
        l2_weight=l2_weight,
        ent_weight=ent_weight,
        batch_size=batch_size
    )

    # Train the BC agent
    bc_trainer.train(n_epochs=n_epochs)

    # Save all components
    agent.save(policy_fpath)

    print(f"Saved BC policy to {policy_fpath}")

if __name__ == "__main__":
    with open("gymnasium_src/config/config_bc.yml", "r") as f: yml = yaml.safe_load(f)
    action_mode = yml["action_mode"]
    device = yml["device"]
    resume_training = yml["resume_training"]

    hyperparameters = yml["hyperparameters"]
    clip_obs = hyperparameters["clip_obs"]
    net_arch = hyperparameters["net_arch"]
    optimizer_lr = float(hyperparameters["optimizer_lr"])
    l2_weight = float(hyperparameters["l2_weight"])
    ent_weight = float(hyperparameters["ent_weight"])
    batch_size = hyperparameters["batch_size"]
    n_epochs = hyperparameters["n_epochs"]
    feature_encoder = hyperparameters.get("feature_encoder")
    history_len = hyperparameters.get("history_len")

    # Load expert demonstrations
    demos_fpath = f"gymnasium_src/demos/expert_demos_{action_mode}.pkl"
    expert_trajectories = load_demos(demos_fpath)

    # Train BC agent
    train_bc(expert_trajectories)