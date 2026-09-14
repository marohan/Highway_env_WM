import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller import EgoController, Maneuver
from planner import HierarchicalPlanner
from world_model import WorldModel
from ray_sensor import RaySensor

def render_body_swap_video():
    print("--- Rendering Body Swap Video ---")
    config = {
        "observation": {"type": "Kinematics"},
        "action": {"type": "ContinuousAction"},
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "duration": 40,
        "lanes_count": 1,
        "vehicles_count": 0,
        "show_trajectories": True
    }
    
    # Use rgb_array to allow RecordVideo to capture frames
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(config)
    env = RecordVideo(env, video_folder="evaluation/videos", name_prefix="body_swap_online_id", episode_trigger=lambda e: True)
    
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
    
    ego_state_prev = None
    prev_raw_action = np.array([0.0, 0.0])
    
    for step in range(40):
        if step >= 15:
            lead.target_speed = max(0.0, lead.target_speed - 2.0)
            
        distances, angles, ego_state_now = sensor.observe(env)
        if step > 0:
            world_model.observe(prev_raw_action, ego_state_prev, ego_state_now, distances, angles)
                
        maneuver, target_speed = planner.plan(ego_state_now, world_model)
        controller.set_maneuver(maneuver, ego_state_now['y'], ego_state_now['vx'], target_speed=target_speed)
        
        raw_action = controller.compute_action(ego_state_now)
        prev_raw_action = raw_action.copy()
        
        actual_action = raw_action.copy()
        if step >= 10: 
            if actual_action[0] < 0:
                actual_action[0] *= 0.4 # 60% loss of braking power
                
        ego_state_prev = ego_state_now
        
        obs, reward, done, truncated, info = env.step(actual_action)
        print(f"Step {step}: maneuver={maneuver}, y={ego_state_now['y']:.2f}, heading={ego_state_now['heading']:.2f}, speed={ego_state_now['vx']:.2f}, steer={raw_action[1]:.2f}")
        
        if info.get("crashed", False):
            break
            
    env.close()
    print("Video saved to evaluation/videos/")

if __name__ == "__main__":
    render_body_swap_video()
