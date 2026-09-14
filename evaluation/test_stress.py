import gymnasium as gym
import highway_env
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller import EgoController, Maneuver
from planner import HierarchicalPlanner
from world_model import WorldModel
from ray_sensor import RaySensor

def test_cut_in():
    print("--- Running Cut-In Stress Test ---")
    config = {
        "observation": {"type": "Kinematics"},
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "duration": 40,
        "lanes_count": 2,
        "vehicles_count": 0,
        "show_trajectories": False
    }
    
    env = gym.make("highway-v0")
    env.unwrapped.configure(config)
    obs, info = env.reset(seed=42)
    
    ego = env.unwrapped.vehicle
    ego.position = np.array([0.0, 4.0]) # right lane
    ego.speed = 20.0
    
    from highway_env.vehicle.behavior import IDMVehicle
    cut_in_veh = IDMVehicle(env.unwrapped.road, position=[60.0, 0.0], speed=15.0, heading=0)
    cut_in_veh.target_speed = 15.0
    env.unwrapped.road.vehicles.append(cut_in_veh)
    
    world_model = WorldModel(dt=1.0)
    world_model.a_decel.mu = -5.0
    world_model.a_decel.var = 0.25**2
    sensor = RaySensor(num_rays=512, max_dist=150.0)
    planner = HierarchicalPlanner(dt=1.0, k_factor=2.0, t_react=1.0, a_lead_max=5.0)
    controller = EgoController(dt=1.0)
    controller.target_speed = 20.0
    
    min_dx = float('inf')
    crashed = False
    
    ego_state_prev = None
    prev_raw_action = np.array([0.0, 0.0])
    for step in range(30):
        # Trigger cut-in at step 2
        if step == 2:
            cut_in_veh.target_lane_index = ("0", "1", 1) # Force it to right lane
            
        distances, angles, ego_state_now = sensor.observe(env)
        if step > 0:
            world_model.observe(prev_raw_action, ego_state_prev, ego_state_now, distances, angles)
            
        maneuver, target_speed = planner.plan(ego_state_now, world_model)
        controller.set_maneuver(maneuver, ego_state_now['y'], ego_state_now['vx'], target_speed=target_speed)
        action = controller.compute_action(ego_state_now)
        prev_raw_action = action.copy()
        ego_state_prev = ego_state_now
        
        obs, reward, done, truncated, info = env.step(action)
        
        dx = cut_in_veh.position[0] - ego.position[0]
        if ego.position[1] > 2.0 and cut_in_veh.position[1] > 2.0 and dx > 0: # both in right lane
            min_dx = min(min_dx, dx)
            
        print(f"Step {step}: cut_in_y={cut_in_veh.position[1]:.1f}, ego_vx={ego.speed:.1f}, dx={dx:.1f}, maneuver={maneuver}")
        
        if info.get("crashed", False):
            crashed = True
            break
            
    print(f"Cut-In Test: min_dx={min_dx:.1f} m")
    if crashed:
        print("FAIL (Collision)")
    else:
        print("PASS (No Collision)")

if __name__ == "__main__":
    test_cut_in()
