import gymnasium as gym
import highway_env
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller import EgoController, Maneuver
from planner import HierarchicalPlanner
from world_model import WorldModel
from ray_sensor import RaySensor

def test_body_swap(use_online_id=True):
    print(f"--- Running Body Swap Test (Online ID: {use_online_id}) ---")
    config = {
        "observation": {"type": "Kinematics"},
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "duration": 60,
        "lanes_count": 1,
        "vehicles_count": 0,
        "show_trajectories": False
    }
    
    env = gym.make("highway-v0")
    env.unwrapped.configure(config)
    obs, info = env.reset(seed=42)
    
    ego = env.unwrapped.vehicle
    ego.position = np.array([0.0, 0.0])
    ego.speed = 20.0
    
    from highway_env.vehicle.behavior import IDMVehicle
    lead = IDMVehicle(env.unwrapped.road, position=[40.0, 0.0], speed=20.0, heading=0)
    lead.target_speed = 20.0
    env.unwrapped.road.vehicles.append(lead)
    
    world_model = WorldModel(dt=1.0)
    world_model.a_decel.mu = -5.0
    world_model.a_decel.var = 0.1**2
    world_model.a_accel.mu = 2.0
    world_model.a_accel.var = 0.1**2
    
    sensor = RaySensor(num_rays=512, max_dist=150.0)
    planner = HierarchicalPlanner(dt=1.0, k_factor=2.0, t_react=1.0, a_lead_max=5.0)
    planner.lanes_count = 1
    controller = EgoController(dt=1.0)
    controller.target_speed = 20.0
    
    min_dx = float('inf')
    crashed = False
    ego_state_prev = None
    prev_raw_action = np.array([0.0, 0.0])
    
    dx_history = []
    v_ego_history = []
    v_lead_history = []
    sigma_history = []
    
    for step in range(60):
        # Body swap: At step 10, brake power is reduced by 60%
        # Lead vehicle sudden braking at step 15
        
        if step >= 15:
            lead.target_speed = max(0.0, lead.target_speed - 2.0)
            
        distances, angles, ego_state_now = sensor.observe(env)
        if step > 0:
            if use_online_id:
                world_model.observe(prev_raw_action, ego_state_prev, ego_state_now, distances, angles)
            else:
                # Disable surprise feedback, just pass 0 error
                pass 
                
        maneuver, target_speed = planner.plan(ego_state_now, world_model)
        controller.set_maneuver(maneuver, ego_state_now['y'], ego_state_now['vx'], target_speed=target_speed)
        
        # Pure controller output
        raw_action = controller.compute_action(ego_state_now)
        prev_raw_action = raw_action.copy()
        
        # Environmental body swap effect applied to physical actuation
        actual_action = raw_action.copy()
        if step >= 10: 
            if actual_action[0] < 0:
                actual_action[0] *= 0.4 # 60% loss of braking power
                
        ego_state_prev = ego_state_now
        
        # Save before step
        dx = lead.position[0] - ego.position[0]
        dx_history.append(dx)
        v_ego_history.append(ego.speed)
        v_lead_history.append(lead.speed)
        sigma_history.append(world_model.a_decel.sigma)
        
        obs, reward, done, truncated, info = env.step(actual_action)
        
        print(f"Step {step}: dx={dx:.1f}, v_ego={ego.speed:.1f}, v_lead={lead.speed:.1f}, raw_a={raw_action[0]:.2f}, act_a={actual_action[0]:.2f}, sigma={world_model.a_decel.sigma:.2f}")
        
        if info.get("crashed", False) or dx <= 0:
            crashed = True
            break
            
    if crashed:
        print("Result: FAIL (Collision)")
    else:
        print("Result: PASS (No Collision)")
        
    return crashed, dx_history, v_ego_history, sigma_history

if __name__ == "__main__":
    print("=== Testing with Fixed Prior (No Online ID) ===")
    crashed_fixed, _, _, _ = test_body_swap(use_online_id=False)
    
    print("\n=== Testing with Online Identification ===")
    crashed_online, _, _, _ = test_body_swap(use_online_id=True)
    
    print("\n=== SUMMARY ===")
    print(f"Fixed Prior: {'FAIL' if crashed_fixed else 'PASS'}")
    print(f"Online ID  : {'FAIL' if crashed_online else 'PASS'}")
