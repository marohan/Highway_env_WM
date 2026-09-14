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

def test_steady_state_follow():
    env_config = {
        "vehicles_count": 1,
        "vehicles_density": 1.0, # Just one vehicle in front
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
    
    # We want to force a vehicle to be exactly in front of ego
    ego = env.unwrapped.vehicle
    # Place ego
    ego.position = np.array([0.0, 0.0])
    ego.speed = 20.0
    
    # Place one lead vehicle
    if len(env.unwrapped.road.vehicles) > 1:
        lead = env.unwrapped.road.vehicles[1]
    else:
        # Create a lead vehicle if env didn't
        from highway_env.vehicle.behavior import IDMVehicle
        lead = IDMVehicle(env.unwrapped.road, position=[50.0, 0.0], speed=20.0, heading=0)
        env.unwrapped.road.vehicles.append(lead)
        
    lead.position = np.array([50.0, 0.0])
    lead.speed = 20.0
    # Force lead vehicle to maintain exactly 20.0 m/s
    lead.target_speed = 20.0
    
    world_model = WorldModel(dt=1.0)
    world_model.a_decel.mu = -5.0 # Set our known brake capability
    world_model.a_decel.var = 0.25**2 # Simulate calibrated agent
    
    planner = HierarchicalPlanner(dt=1.0, k_factor=1.0, t_react=1.46, a_lead_max=5.0)
    controller = EgoController(dt=1.0)
    
    distances = []
    speeds = []
    
    for step in range(60):
        lead.speed = 20.0
        ego_state = {
            'x': ego.position[0], 'y': ego.position[1],
            'heading': ego.heading, 'vx': ego.velocity[0], 'vy': ego.velocity[1]
        }
        from world_model import Track
        trk = Track(0, lead.position[0], lead.position[1])
        trk.x = np.array([lead.position[0], lead.position[1], lead.speed, 0.0])
        trk.hits = 5 # Confirmed
        world_model.mot.tracks = [trk]
        
        maneuver, target_speed = planner.plan(ego_state, world_model)
        controller.set_maneuver(maneuver, ego_state['y'], ego_state['vx'], target_speed=target_speed)
        action = controller.compute_action(ego_state)
        
        obs, reward, done, truncated, info = env.step(action)
        
        dx = lead.position[0] - ego.position[0]
        distances.append(dx)
        speeds.append(ego.velocity[0])
        
        a_brake_pess = max(0.1, abs(world_model.a_decel.mu) - planner.k_factor * world_model.a_decel.sigma)
        print(f"Step {step}: dx={dx:.2f}, v_ego={ego.velocity[0]:.2f}, v_filt={planner.v_ego_filt:.2f}, target_v={target_speed:.2f}, accel={action[0]*5.0:.2f}, maneuver={maneuver}")
        
        if done or truncated:
            break
            
    # Calculate expected target_gap
    v_ego_filt = planner.v_ego_filt
    v_lead = 20.0
    a_brake_pess = max(0.1, abs(world_model.a_decel.mu) - planner.k_factor * world_model.a_decel.sigma)
    expected_gap = v_ego_filt * planner.t_react + (v_ego_filt**2)/(2*a_brake_pess) - (v_lead**2)/(2*planner.a_lead_max) + 5.0

    
    final_dx = distances[-1]
    final_vx = speeds[-1]
    
    print(f"Steady-State Follow Test:")
    print(f"Expected Target Gap: {expected_gap:.2f} m")
    print(f"Final Distance (dx): {final_dx:.2f} m")
    print(f"Final Ego Speed: {final_vx:.2f} m/s")
    
    overshoot = (final_dx - expected_gap) / expected_gap * 100
    print(f"Gap Error: {final_dx - expected_gap:.2f} m ({overshoot:.1f}%)")
    
    if abs(overshoot) < 10.0 and abs(final_vx - 20.0) < 1.0:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    test_steady_state_follow()
