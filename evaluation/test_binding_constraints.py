import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model import WorldModel
from planner import HierarchicalPlanner
from controller import EgoController

def test_binding_constraints():
    env_config = {
        "vehicles_count": 50,
        "duration": 100,
        "policy_frequency": 1,
        "simulation_frequency": 15,
        "action": {
            "type": "ContinuousAction"
        }
    }
    
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(env_config)
    
    planner = HierarchicalPlanner(k_factor=2.0)
    world_model = WorldModel()
    
    obs, info = env.reset()
    logs = []
    
    for step in range(100):
        ego = env.unwrapped.vehicle
        ego_state = {
            'x': ego.position[0], 'y': ego.position[1],
            'heading': ego.heading, 'vx': ego.velocity[0], 'vy': ego.velocity[1]
        }
        # Skip world_model observe for this simple math check
        
        # Test v_ego = 25 d_safe check
        # We can temporarily hack ego speed to 25 and v_lead to 0 in _L1_invariant
        if step == 0:
            v = 20.0
            v_lead = 20.0
            # For v=20, a_brake=5.0 (since abs(mu)=5.0 for real driving, but we start with 10.0 default, so we'll just check the formula logic)
            # In L1 invariant, a_brake_min = 5.0, a_lead_max = 5.0
            tau = 1.0
            a_brake = 5.0 
            a_lead_max = 5.0
            d_to_stop = v * tau + (v**2) / (2 * a_brake)
            d_lead = (v_lead**2) / (2 * a_lead_max)
            d_safe_20 = max(10.0, max(0.5 * v, d_to_stop - d_lead))
            print(f"Test 3 - Calculated L1 d_safe for v=20, v_lead=20: {d_safe_20:.2f} m")
            break

        # Track what layer binds
        # We need to modify planner temporarily to return binding info, or we can just capture it by modifying planner.py later.
        # But for now, let's just log what we can.
        
        action, _ = planner.plan(ego_state, world_model)
        
        # In this script we just want to run the environment. We will modify planner.py to dump logs to a file.
        obs, reward, done, truncated, info = env.step([0, 0]) # Just pass dummy action since we only care about planning logs
        
        if done or truncated:
            break

if __name__ == "__main__":
    test_binding_constraints()
