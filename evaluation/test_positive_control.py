import gymnasium as gym
import highway_env
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planner import HierarchicalPlanner
from world_model import WorldModel
from controller import EgoController
from ray_sensor import RaySensor

def run_positive_control():
    env_config = {
        "vehicles_count": 50,
        "vehicles_density": 2.0, # high density
        "duration": 60,
        "policy_frequency": 1,
        "simulation_frequency": 15,
        "action": {
            "type": "ContinuousAction",
            "acceleration_range": [-5.0, 5.0],
            "steering_range": [-np.pi/4, np.pi/4],
        },
        "observation": {"type": "Kinematics"},
    }
    
    env = gym.make("highway-v0")
    env.unwrapped.configure(env_config)
    
    num_episodes = 20
    collisions = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=100 + ep)
        planner = HierarchicalPlanner(k_factor=0.0)
        planner.disable_safety = True
        
        world_model = WorldModel()
        controller = EgoController()
        controller.target_speed = 30.0 # Force to go fast
        sensor = RaySensor()
        
        _, _, ego_state_prev = sensor.observe(env)
        action_prev = None
        
        for step in range(env_config["duration"]):
            distances, angles, ego_state_now = sensor.observe(env)
            if action_prev is not None:
                world_model.observe(action_prev, ego_state_prev, ego_state_now, distances, angles)
                
            maneuver, target_speed = planner.plan(ego_state_now, world_model)
            controller.set_maneuver(maneuver, ego_state_now['y'], ego_state_now['vx'], target_speed=target_speed)
            action = controller.compute_action(ego_state_now)
            
            obs, reward, done, truncated, info = env.step(action)
            
            if info.get("crashed", False):
                collisions += 1
                break
                
            if done or truncated:
                break
                
            ego_state_prev = ego_state_now
            action_prev = action
            
    print(f"Positive Control Collision Rate: {collisions / num_episodes:.2f} ({collisions}/{num_episodes})")

if __name__ == "__main__":
    run_positive_control()
