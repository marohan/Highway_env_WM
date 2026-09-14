import gymnasium as gym
import highway_env
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class IDMAgent:
    """
    A baseline agent that simply returns the 'IDLE' action (1),
    allowing the built-in IDM/MOBIL controllers of highway-env to take over.
    In DiscreteMetaAction space:
    0: LANE_LEFT
    1: IDLE (maintained by IDM)
    2: LANE_RIGHT
    3: FASTER
    4: SLOWER
    """
    def predict(self, obs, deterministic=True):
        return 1, None


class TrainingLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.interactions = []
        self.current_interactions = 0

    def _on_step(self) -> bool:
        self.current_interactions += 1
        if "episode" in self.locals["infos"][0]:
            self.episode_rewards.append(self.locals["infos"][0]["episode"]["r"])
            self.interactions.append(self.current_interactions)
        return True


def train_ppo_baseline(total_timesteps=20000, model_path="ppo_highway"):
    """
    Trains a PPO baseline on highway-v0 for sample efficiency comparison.
    """
    env = gym.make("highway-v0")
    
    # Custom config to match our evaluation config later
    config = {
        "observation": {"type": "Kinematics"},
        "action": {
            "type": "ContinuousAction",
            "acceleration_range": [-5.0, 5.0],
            "steering_range": [-np.pi/4, np.pi/4],
        },
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "duration": 40,
        "vehicles_count": 10,
        "vehicles_density": 1.5,
    }
    env.unwrapped.configure(config)
    env.reset()
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
    
    callback = TrainingLoggerCallback()
    print(f"Training PPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    model.save(model_path)
    env.close()
    
    return callback.interactions, callback.episode_rewards


if __name__ == "__main__":
    # Test PPO training if executed directly
    train_ppo_baseline(total_timesteps=5000)
