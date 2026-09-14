# -*- coding: utf-8 -*-
"""
evaluate_dense_traffic.py
Tests the agent in a highly complex, dense traffic environment.
Records metrics and renders the video.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from highway_env.vehicle.behavior import IDMVehicle
from ray_sensor import RaySensor
from world_model import WorldModel
from planner import HierarchicalPlanner, Maneuver
from controller import EgoController
from gymnasium.wrappers import RecordVideo

SIM_FREQ    = 15
POLICY_FREQ = 15
PLAN_EVERY  = 15
LANE_WIDTH  = 4.0

def run_dense_traffic(duration_s=40):
    print("\n" + "="*60)
    print("  Evaluating Dense Traffic Scenario")
    print("="*60)

    # Highway config with high density
    config = {
        "observation":          {"type": "Kinematics"},
        "action":               {"type": "ContinuousAction"},
        "simulation_frequency": SIM_FREQ,
        "policy_frequency":     POLICY_FREQ,
        "duration":             duration_s * POLICY_FREQ,
        "lanes_count":          4,
        "vehicles_count":       15,
        "vehicles_density":     0.8,
        "initial_lane_id":      None,
        "ego_spacing":          2,
        "show_trajectories":    True,
        "offscreen_rendering":  True,
    }

    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(config)
    env = RecordVideo(env, video_folder="evaluation/videos/Dense_Traffic", name_prefix="Dense_Traffic")

    obs, info = env.reset(seed=42)

    # Set up our modules
    sensor      = RaySensor(num_rays=512, max_dist=150.0)
    planner     = HierarchicalPlanner(dt=1.0, k_factor=2.0, t_react=1.0, a_lead_max=5.0)
    planner.lanes_count = 4
    controller  = EgoController(dt=1.0 / POLICY_FREQ)
    world_model = WorldModel(dt=1.0)
    
    # Warm-start ID prior
    world_model.a_decel.mu  = -5.0
    world_model.a_decel.var = 0.25 ** 2
    world_model.a_accel.mu  = 3.0
    world_model.a_accel.var = 0.5 ** 2

    # Give ego an initial target speed and lane
    ego = env.unwrapped.vehicle
    ego.speed = 25.0
    controller.target_speed = 30.0
    controller.target_lane_y = round(ego.position[1] / LANE_WIDTH) * LANE_WIDTH

    ego_state_prev = None
    prev_raw_action = np.array([0.0, 0.0])
    
    y_min = 0.0
    y_max = (4 - 1) * LANE_WIDTH

    total_steps = duration_s * POLICY_FREQ
    
    # Metrics
    metrics = {
        "speeds": [],
        "lane_changes": 0,
        "min_ttc": float('inf'),
        "crashed": False,
        "l0_triggers": 0
    }
    
    prev_lane = round(ego.position[1] / LANE_WIDTH)

    for ctrl_step in range(total_steps):
        distances, angles, ego_state_now = sensor.observe(env)
        ego_y  = ego_state_now["y"]
        ego_vx = ego_state_now["vx"]
        metrics["speeds"].append(ego_vx)
        
        cur_lane = int(round(np.clip(ego_y, y_min, y_max) / LANE_WIDTH))
        if cur_lane != prev_lane:
            metrics["lane_changes"] += 1
            prev_lane = cur_lane

        if ctrl_step % PLAN_EVERY == 0:
            if ctrl_step > 0:
                world_model.observe(prev_raw_action, ego_state_prev, ego_state_now, distances, angles)
                
            plan_maneuver, plan_target_speed = planner.plan(ego_state_now, world_model)
            
            if plan_maneuver == Maneuver.DECEL_TO_STOP:
                metrics["l0_triggers"] += 1
                
            if plan_maneuver == Maneuver.CHANGE_LEFT:
                tgt_lane = max(0, cur_lane - 1)
            elif plan_maneuver == Maneuver.CHANGE_RIGHT:
                tgt_lane = min(3, cur_lane + 1)
            else:
                tgt_lane = cur_lane
                
            controller.target_speed     = plan_target_speed
            controller.current_maneuver = plan_maneuver
            controller.target_lane_y    = tgt_lane * LANE_WIDTH
            
            t_s = ctrl_step // PLAN_EVERY
            if t_s % 5 == 0:
                print("t=%3ds maneuver=%d speed=%5.2f y=%5.2f" % (t_s, plan_maneuver, ego_vx, ego_y))

        # Clamp target y
        controller.target_lane_y = np.clip(controller.target_lane_y, y_min, y_max)

        raw_action = controller.compute_action(ego_state_now)
        actual_action = raw_action.copy()
        
        prev_raw_action = actual_action.copy()
        ego_state_prev  = ego_state_now

        # Calculate TTC to closest vehicle ahead for metrics
        for v in env.unwrapped.road.vehicles:
            if v is not ego:
                dx = v.position[0] - ego.position[0]
                dy = abs(v.position[1] - ego.position[1])
                dvx = ego.velocity[0] - v.velocity[0]
                if dx > 0 and dy < 2.0 and dvx > 0:
                    ttc = dx / dvx
                    if ttc < metrics["min_ttc"]:
                        metrics["min_ttc"] = ttc

        obs, reward, done, truncated, info = env.step(actual_action)

        if info.get("crashed", False):
            print("  *** CRASHED at t=%.1fs ***" % (ctrl_step / POLICY_FREQ))
            metrics["crashed"] = True
            break
        if done or truncated:
            break

    env.close()
    
    # Calculate results
    avg_speed = np.mean(metrics["speeds"])
    print("\n--- Dense Traffic Results ---")
    print(f"Crashed:       {metrics['crashed']}")
    print(f"Avg Speed:     {avg_speed:.2f} m/s")
    print(f"Lane Changes:  {metrics['lane_changes']}")
    print(f"Min TTC:       {metrics['min_ttc']:.2f} s")
    print(f"L0 Triggers:   {metrics['l0_triggers']}")
    print("Saved to evaluation/videos/Dense_Traffic")

if __name__ == "__main__":
    run_dense_traffic(duration_s=60)