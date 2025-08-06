import gymnasium as gym

# Import the gymnasium_robotics environments to register them with Gymnasium.
# This makes environments like FetchPickAndPlace available.
import gymnasium_robotics

def main():
    # Create the pick and place environment, explicitly setting the
    # render_mode to "human" to display a GUI window.
    env = gym.make("FetchPickAndPlace-v4", render_mode="human")

    # Reset environment to get initial observation and info
    obs, info = env.reset()
    
    total_reward = 0.0
    done = False

    while not done:
        # A random action is chosen from the environment's action space.
        # In a real-world application, this would be replaced by your
        # agent's policy.
        action = env.action_space.sample()
        
        # Take a step in the environment with the chosen action.
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # The environment is automatically rendered due to the `render_mode`
        # being set to "human" during creation.
        # This line is not strictly necessary for rendering but is good practice.
        env.render()
        
        # The episode ends if it is either terminated (goal reached) or
        # truncated (time limit exceeded).
        # done = terminated or truncated

    print(f"Episode finished. Total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()