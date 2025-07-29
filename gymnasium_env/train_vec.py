import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

num_envs = 15

# Register environment at the module level - outside of any function
register(
    id="gymnasium_env/ur3e-v0",
    entry_point="gymnasium_env.envs.ur3e_env:UR3eEnv"
)

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :return: (Callable)
    """
    def _init():
        # Re-register environment in each worker process
        try:
            register(
                id="gymnasium_env/ur3e-v0",
                entry_point="gymnasium_env.envs.ur3e_env:UR3eEnv"
            )
        except Exception:
            # Ignore if already registered
            pass
            
        env = gym.make("gymnasium_env/ur3e-v0", render_mode="rgb_array")
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    # Create environment initialization functions
    env_fns = [make_env(i) for i in range(num_envs)]
    
    # Create vectorized environment using SubprocVecEnv
    env = SubprocVecEnv(env_fns)

    # Evaluation callback
    eval_callback = EvalCallback(
        env,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ur3e_tensorboard/",
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=30,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )

    # Train the model
    model.learn(
        total_timesteps=100_000,
        callback=eval_callback
    )

    # Save the model
    model.save("gymnasium_env/ur3e_pickplace_model")

if __name__ == "__main__":
    main()