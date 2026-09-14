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

def test_step_response():
    env_config = {
        "vehicles_count": 1,
        "vehicles_density": 1.0, 
        "duration": 60,
        "lanes_count": 1,
        "policy_frequency": 1,
        "simulation_frequency": 15,
        "action": {
            "type": "ContinuousAction"
        },
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "absolute": True
        }
    }
    
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(env_config)
    obs, info = env.reset()
    
    ego = env.unwrapped.vehicle
    ego.position = np.array([0.0, 0.0])
    ego.speed = 20.0
    
    if len(env.unwrapped.road.vehicles) > 1:
        lead = env.unwrapped.road.vehicles[1]
    else:
        from highway_env.vehicle.behavior import IDMVehicle
        lead = IDMVehicle(env.unwrapped.road, position=[50.0, 0.0], speed=20.0, heading=0)
        env.unwrapped.road.vehicles.append(lead)
        
    lead.position = np.array([50.0, 0.0])
    lead.speed = 20.0
    
    world_model = WorldModel(dt=1.0)
    world_model.a_decel.mu = -5.0
    world_model.a_decel.var = 0.25**2
    
    planner = HierarchicalPlanner(dt=1.0, k_factor=1.0, t_react=1.46, a_lead_max=5.0)
    controller = EgoController(dt=1.0)
    
    distances = []
    ego_speeds = []
    lead_speeds = []
    min_dx = float('inf')
    
    for step in range(60):
        # Step response: Lead brakes hard at t=20
        if step < 20:
            lead.speed = 20.0
            lead.target_speed = 20.0
        elif step < 25:
            # Sudden braking at -4 m/s^2
            lead.speed = max(0.0, lead.speed - 4.0)
            lead.target_speed = lead.speed
        else:
            lead.speed = lead.speed
            lead.target_speed = lead.speed
            
        ego_state = {
            'x': ego.position[0], 'y': ego.position[1],
            'heading': ego.heading, 'vx': ego.velocity[0], 'vy': ego.velocity[1]
        }
        
        from world_model import Track
        trk = Track(0, lead.position[0], lead.position[1])
        trk.x = np.array([lead.position[0], lead.position[1], lead.speed, 0.0])
        trk.hits = 5
        trk = Track(0, lead.position[0], lead.position[1])
        trk.x = np.array([lead.position[0], lead.position[1], lead.speed, 0.0])
        trk.hits = 5
        
        # Add phantom tracks 10m ahead to block lane changes
        trk_left = Track(1, ego.position[0] + 10.0, ego.position[1] - 4.0)
        trk_left.x = np.array([ego.position[0] + 10.0, ego.position[1] - 4.0, 0.0, 0.0])
        trk_left.hits = 5
        
        trk_right = Track(2, ego.position[0] + 10.0, ego.position[1] + 4.0)
        trk_right.x = np.array([ego.position[0] + 10.0, ego.position[1] + 4.0, 0.0, 0.0])
        trk_right.hits = 5
        
        world_model.mot.tracks = [trk, trk_left, trk_right]        
        maneuver, target_speed = planner.plan(ego_state, world_model)
        controller.set_maneuver(maneuver, ego_state['y'], ego_state['vx'], target_speed=target_speed)
        action = controller.compute_action(ego_state)
        
        obs, reward, done, truncated, info = env.step(action)
        
        dx = lead.position[0] - ego.position[0]
        distances.append(dx)
        ego_speeds.append(ego.velocity[0])
        lead_speeds.append(lead.speed)
        min_dx = min(min_dx, dx)
        
        print(f"Step {step}: dx={dx:.2f}, v_ego={ego.velocity[0]:.2f}, v_lead={lead.speed:.2f}, target_v={target_speed:.2f}, maneuver={maneuver}")
        
        if done or truncated or dx <= 0:
            break
            
    print("Step Response Test:")
    print(f"Minimum Distance: {min_dx:.2f} m")
    if min_dx > 5.0:
        print("PASS (No Collision)")
    else:
        print("FAIL (Collision)")

if __name__ == "__main__":
    test_step_response()
