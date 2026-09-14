import gymnasium as gym
import highway_env
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller import EgoController, Maneuver
from planner import HierarchicalPlanner

def test_free_flow():
    env_config = {
        "vehicles_count": 0,
        "duration": 60,
        "policy_frequency": 1,
        "simulation_frequency": 15,
        "action": {
            "type": "ContinuousAction"
        }
    }
    
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(env_config)
    obs, info = env.reset()
    
    controller = EgoController()
    controller.target_speed = 30.0
    
    # Run for 60 steps (or 60 seconds? policy_frequency=1 means 1 step = 1 sec)
    for step in range(60):
        # We don't even need the planner for Test A, just the EgoController to see if it reaches 30
        maneuver = Maneuver.KEEP_SPEED
        ego = env.unwrapped.vehicle
        ego_state = {
            'x': ego.position[0], 'y': ego.position[1],
            'heading': ego.heading, 'vx': ego.velocity[0], 'vy': ego.velocity[1]
        }
        
        controller.set_maneuver(maneuver, ego_state['y'], ego_state['vx'], target_speed=30.0)
        action = controller.compute_action(ego_state)
        
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            break
            
    final_speed = env.unwrapped.vehicle.speed
    print(f"Test A - Final Speed: {final_speed:.2f} m/s")

if __name__ == "__main__":
    test_free_flow()
