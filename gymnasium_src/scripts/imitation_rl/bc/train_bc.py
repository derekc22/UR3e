import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import bc
from imitation.data import rollout
from gymnasium_src.scripts.imitation_rl.collect_demonstrations import load_demonstrations
# from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import torch
import register_envs # Import your new registration file
from stable_baselines3.common.env_util import make_vec_env
import yaml


def train_behavioral_cloning(expert_trajs):
    # Register environment
    # register(
    #     id="gymnasium_env/ur3e-v0",
    #     entry_point=f"envs.imitation_env_{agent_mode}:ImitationEnv"
    # )

    # Create environment
    env = gym.make(
        id=f"gymnasium_env/imitation_{agent_mode}-v0",
    )
    
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

    policy_kwargs = dict(net_arch=net_arch)
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
    net_arch = hyperparameters["net_arch"]
    optimizer_lr = float(hyperparameters["optimizer_lr"])
    l2_weight = float(hyperparameters["l2_weight"])
    ent_weight = float(hyperparameters["ent_weight"])
    batch_size = hyperparameters["batch_size"]
    n_epochs = hyperparameters["n_epochs"]

    # Load the expert demonstrations
    demos_fpath = f"gymnasium_src/demos/expert_demos_{agent_mode}.pkl"
    expert_trajectories = load_demonstrations(demos_fpath)

    # Train the BC agent
    trained_policy = train_behavioral_cloning(expert_trajectories)
