# -*- coding: utf-8 -*-
"""
render_all_scenarios.py
1Hz Planning / 15Hz Control dual-loop structure.
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


def run_scenario(scenario_name, lanes_count, setup_env_fn, step_fn, duration_s=25):
    print("\n" + "="*60)
    print("  Rendering: " + scenario_name)
    print("="*60)

    config = {
        "observation":          {"type": "Kinematics"},
        "action":               {"type": "ContinuousAction"},
        "simulation_frequency": SIM_FREQ,
        "policy_frequency":     POLICY_FREQ,
        "duration":             duration_s * POLICY_FREQ,
        "lanes_count":          lanes_count,
        "vehicles_count":       0,
        "show_trajectories":    True,
        "offscreen_rendering":  True,
    }

    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(config)
    env = RecordVideo(env,
                      video_folder="evaluation/videos/" + scenario_name,
                      name_prefix=scenario_name)

    obs, info = env.reset(seed=42)

    sensor      = RaySensor(num_rays=512, max_dist=150.0)
    planner     = HierarchicalPlanner(dt=1.0, k_factor=2.0, t_react=1.0, a_lead_max=5.0)
    planner.lanes_count = lanes_count
    controller  = EgoController(dt=1.0 / POLICY_FREQ)
    world_model = WorldModel(dt=1.0)
    # Warm-start ID prior at nominal actuator values
    world_model.a_decel.mu  = -5.0
    world_model.a_decel.var = 0.25 ** 2
    world_model.a_accel.mu  = 3.0
    world_model.a_accel.var = 0.5 ** 2

    y_min =  0.0
    y_max = (lanes_count - 1) * LANE_WIDTH

    setup_env_fn(env, controller)

    ego_state_prev    = None
    prev_raw_action   = np.array([0.0, 0.0])
    plan_maneuver     = Maneuver.KEEP_SPEED
    plan_target_speed = controller.target_speed

    total_steps = duration_s * POLICY_FREQ

    for ctrl_step in range(total_steps):
        distances, angles, ego_state_now = sensor.observe(env)
        ego_y  = ego_state_now["y"]
        ego_vx = ego_state_now["vx"]

        if ctrl_step % PLAN_EVERY == 0:
            if ctrl_step > 0:
                world_model.observe(prev_raw_action, ego_state_prev, ego_state_now,
                                    distances, angles)
            plan_maneuver, plan_target_speed = planner.plan(ego_state_now, world_model)

            cur_lane = int(round(np.clip(ego_y, y_min, y_max) / LANE_WIDTH))
            if plan_maneuver == Maneuver.CHANGE_LEFT:
                tgt_lane = max(0, cur_lane - 1)
            elif plan_maneuver == Maneuver.CHANGE_RIGHT:
                tgt_lane = min(lanes_count - 1, cur_lane + 1)
            else:
                tgt_lane = cur_lane
            plan_target_lane_y = tgt_lane * LANE_WIDTH

            controller.target_speed     = plan_target_speed
            controller.current_maneuver = plan_maneuver
            controller.target_lane_y    = plan_target_lane_y

            t_s = ctrl_step // PLAN_EVERY
            print("t=%3ds  maneuver=%d  speed=%5.2f  y=%5.2f  tgt_y=%.1f" % (
                t_s, plan_maneuver, ego_vx, ego_y, plan_target_lane_y))

        clamped = np.clip(controller.target_lane_y, y_min, y_max)
        controller.target_lane_y = clamped

        raw_action    = controller.compute_action(ego_state_now)
        actual_action = step_fn(ctrl_step, raw_action.copy(), env)
        prev_raw_action = actual_action.copy()
        ego_state_prev  = ego_state_now

        obs, reward, done, truncated, info = env.step(actual_action)

        if info.get("crashed", False):
            print("  *** CRASHED at t=%.1fs ***" % (ctrl_step / POLICY_FREQ))
            break
        if done or truncated:
            break

    env.close()
    print("  Saved to evaluation/videos/" + scenario_name)


# ---- scenario setups ----

def setup_normal(env, controller):
    ego = env.unwrapped.vehicle
    ego.position = np.array([0.0, 4.0])
    ego.speed    = 20.0
    controller.target_speed    = 25.0
    controller.target_lane_y   = 4.0

    lead = IDMVehicle(env.unwrapped.road, position=[50.0, 4.0], speed=20.0, heading=0)
    lead.target_speed = 20.0
    env.unwrapped.road.vehicles.append(lead)

def step_normal(ctrl_step, raw_action, env):
    return raw_action


def setup_cutin(env, controller):
    ego = env.unwrapped.vehicle
    ego.position = np.array([0.0, 4.0])
    ego.speed    = 25.0
    controller.target_speed    = 25.0
    controller.target_lane_y   = 4.0

    cutin = IDMVehicle(env.unwrapped.road, position=[55.0, 0.0], speed=18.0, heading=0)
    cutin.target_speed = 18.0
    env.unwrapped.road.vehicles.append(cutin)

def step_cutin(ctrl_step, raw_action, env):
    if ctrl_step == 5 * PLAN_EVERY:
        env.unwrapped.road.vehicles[1].target_lane_index = ("0", "1", 1)
    return raw_action


def setup_brakes(env, controller):
    ego = env.unwrapped.vehicle
    ego.position = np.array([0.0, 0.0])
    ego.speed    = 20.0
    controller.target_speed    = 25.0
    controller.target_lane_y   = 0.0

    lead = IDMVehicle(env.unwrapped.road, position=[45.0, 0.0], speed=20.0, heading=0)
    lead.target_speed = 20.0
    env.unwrapped.road.vehicles.append(lead)

def step_brakes(ctrl_step, raw_action, env):
    t = ctrl_step / POLICY_FREQ
    if abs(t - 5.0) < 1.0/POLICY_FREQ:
        env.unwrapped.road.vehicles[1].target_speed = 15.0
    elif abs(t - 10.0) < 1.0/POLICY_FREQ:
        env.unwrapped.road.vehicles[1].target_speed = 20.0
    elif abs(t - 18.0) < 1.0/POLICY_FREQ:
        env.unwrapped.road.vehicles[1].target_speed = 0.0

    actual = raw_action.copy()
    if t >= 5.0 and actual[0] < 0:
        actual[0] *= 0.7
    return actual


if __name__ == "__main__":
    run_scenario("Normal_Driving", lanes_count=4, setup_env_fn=setup_normal, step_fn=step_normal, duration_s=25)
    run_scenario("Cut_In",         lanes_count=4, setup_env_fn=setup_cutin,  step_fn=step_cutin,  duration_s=25)
    run_scenario("Brake_Loss",     lanes_count=1, setup_env_fn=setup_brakes, step_fn=step_brakes, duration_s=40)