# UR3e Robot Gymnasium Environment

A Gymnasium environment for the Universal Robots UR3e robotic arm using MuJoCo physics simulation. This environment allows for reinforcement learning and control experiments with the UR3e robot.

<img src="./media/capture.png" alt="UR3e Robot in MuJoCo simulation" width="400"/>

## Installation

To install the UR3e environment, run the following commands:

```bash
git clone https://github.com/yourusername/UR3e.git
cd UR3e
pip install -e .
```

### Prerequisites

- Python 3.7+
- MuJoCo 2.1.0+
- Gymnasium

Install dependencies with:

```bash
pip install gymnasium mujoco
```

## Usage

Basic example of using the environment:

```python
import gymnasium as gym
from gymnasium.envs.registration import register

# Register the environment
register(
    id="gymnasium_env/ur3e-v0",
    entry_point="gymnasium_env.envs.ur3e_env:UR3eEnv"
)

# Create environment
env = gym.make("gymnasium_env/ur3e-v0", render_mode="human")

# Run simulation
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

## Environment Details

- **Observation Space**: Joint positions and velocities
- **Action Space**: Joint torques within defined limits
- **Reward Function**: Currently based on joint positions and velocities
- **Terminal Conditions**: Customizable based on task requirements

## Customization

The environment can be customized by modifying the reward function, terminal conditions, or robot model parameters in the `ur3e_env.py` file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/) for the reinforcement learning framework
- [MuJoCo](https://mujoco.org/) for the physics simulation
- [Universal Robots](https://www.universal-robots.com/) for the UR3e robot specifications

