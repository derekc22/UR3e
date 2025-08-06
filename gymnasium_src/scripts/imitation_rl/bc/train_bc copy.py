import gymnasium as gym
from stable_baselines3 import PPO
from imitation.algorithms import bc
from imitation.data import rollout
from gymnasium_src.scripts.imitation_rl.collect_demos import load_demos
from stable_baselines3 import PPO
import numpy as np
import torch
import register_envs # Import your new registration file
import yaml
from gymnasium.wrappers import FrameStackObservation



def train_behavioral_cloning(expert_trajs):

    # Create environment
    env = gym.make(id=f"gymnasium_env/imitation_{agent_mode}-v0")
    
    policy_kwargs = dict(net_arch=net_arch)
    
    if feature_encoder == "transformer":
        from gymnasium_src.feature_extractors.transformer import TransformerFeatureExtractor
        env = FrameStackObservation(env, stack_size=history_len)
        
        policy_kwargs.update(dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256), # Output dimension of the transformer
        ))
        
        from gymnasium_src.scripts.imitation_rl.collect_demos import stack_expert_trajectories
        expert_trajs = stack_expert_trajectories(expert_trajs, history_len)
    
    # Convert trajectories to transitions
    transitions = rollout.flatten_trajectories(expert_trajs)
    
    if device == "mps":
        from imitation.data.types import Transitions
        transitions = Transitions(
            obs=transitions.obs.astype(np.float32),
            acts=transitions.acts.astype(np.float32),
            infos=transitions.infos,
            next_obs=transitions.next_obs.astype(np.float32),
            dones=transitions.dones,
        )

    policy = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs).policy
    
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=policy,
        device=torch.device(device) if device != "auto" else "auto",
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": optimizer_lr},
        rng=np.random.default_rng(),
        
        l2_weight=l2_weight,  # Pass l2_weight as a direct argument to bc.BC
        ent_weight=ent_weight, # Increase from the default of 1e-3
        batch_size=batch_size
    )
    
    # Train BC
    bc_trainer.train(n_epochs=n_epochs)
    
    # Save trainer
    save_dir = "policies/imitation_rl_policies"
    bc_trainer.policy.save(f"{save_dir}/bc_policy_{agent_mode}.zip")
    print(f"Saved BC trainer to {save_dir}/bc_policy_{agent_mode}.zip")
    
    return bc_trainer



if __name__ == "__main__":
    with open("gymnasium_src/config/config_bc.yml", "r") as f:  yml = yaml.safe_load(f)    
    agent_mode = yml["agent_mode"]
    device = yml["device"]
    hyperparameters = yml["hyperparameters"]
    net_arch = hyperparameters.get("net_arch")
    optimizer_lr = float(hyperparameters.get("optimizer_lr"))
    l2_weight = float(hyperparameters.get("l2_weight"))
    ent_weight = float(hyperparameters.get("ent_weight"))
    batch_size = hyperparameters.get("batch_size")
    n_epochs = hyperparameters.get("n_epochs")
    feature_encoder = hyperparameters.get("feature_encoder")
    history_len = hyperparameters.get("history_len")

    # Load the expert demonstrations
    demos_fpath = f"gymnasium_src/demos/expert_demos_{agent_mode}.pkl"
    expert_trajectories = load_demos(demos_fpath)

    # Train the BC agent
    trained_policy = train_behavioral_cloning(expert_trajectories)
