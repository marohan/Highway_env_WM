import gymnasium as gym
import highway_env
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model import WorldModel
from planner import HierarchicalPlanner
from controller import EgoController, Maneuver
from ray_sensor import RaySensor

def debug_run():
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
        "vehicles_count": 20,
        "vehicles_density": 1.5,
    }
    
    env = gym.make("highway-v0")
    env.unwrapped.configure(config)
    
    world_model = WorldModel(dt=1.0)
    world_model.a_accel.mu = 2.0; world_model.a_accel.var = 0.1**2
    world_model.a_decel.mu = -5.0; world_model.a_decel.var = 0.1**2
    world_model.kp.mu = 0.4; world_model.kp.var = 0.1**2
    world_model.kd.mu = 0.6; world_model.kd.var = 0.1**2
    
    sensor = RaySensor(num_rays=512, max_dist=150.0)
    planner = HierarchicalPlanner(dt=1.0, k_factor=1.0, t_react=1.0, a_lead_max=5.0)
    controller = EgoController(dt=0.1)
    controller.target_speed = 30.0
    
    obs, info = env.reset(seed=42)
    world_model.reset_episode()
    
    done = truncated = False
    step = 0
    action = [0, 0]
    ego_state_prev = None
    
    while not (done or truncated):
        distances, angles, ego_state_now = sensor.observe(env)
        if step > 0:
            world_model.observe(action, ego_state_prev, ego_state_now, distances, angles)
            
        maneuver, target_speed = planner.plan(ego_state_now, world_model)
        controller.set_maneuver(maneuver, ego_state_now['y'], ego_state_now['vx'], target_speed=target_speed)
        action = controller.compute_action(ego_state_now)
        ego_state_prev = ego_state_now
        
        obs, reward, done, truncated, info = env.step(action)
        
        crashed = info.get("crashed", False)
        print(f"Step {step}: maneuver={maneuver}, target_v={target_speed:.2f}, action={action}, crashed={crashed}")
        if crashed:
            break
            
        step += 1

if __name__ == "__main__":
    debug_run()
