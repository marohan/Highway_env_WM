import sys, os
sys.path.append(".")
import gymnasium as gym, numpy as np
from highway_env.vehicle.behavior import IDMVehicle
from ray_sensor import RaySensor
from world_model import WorldModel
from planner import HierarchicalPlanner, Maneuver
from controller import EgoController

env = gym.make("highway-v0", render_mode="rgb_array")
env.unwrapped.configure({"observation": {"type": "Kinematics"}, "action": {"type": "ContinuousAction"}, "simulation_frequency": 15, "policy_frequency": 15, "duration": 90, "lanes_count": 4, "vehicles_count": 0})
obs, info = env.reset(seed=42)
ego = env.unwrapped.vehicle
ego.position = np.array([0.0, 4.0])
ego.speed = 20.0

lead = IDMVehicle(env.unwrapped.road, position=[50.0, 4.0], speed=20.0, heading=0)
lead.target_speed = 20.0
env.unwrapped.road.vehicles.append(lead)

sensor = RaySensor(num_rays=512, max_dist=150.0)
planner = HierarchicalPlanner(dt=1.0, k_factor=2.0, t_react=1.0, a_lead_max=5.0)
planner.lanes_count = 4
controller = EgoController(dt=1.0/15)
controller.target_speed = 25.0
controller.target_lane_y = 4.0
world_model = WorldModel(dt=1.0)

ego_state_prev = None
prev_raw_action = np.array([0.0, 0.0])

for ctrl_step in range(30):
    distances, angles, ego_state_now = sensor.observe(env)
    if ctrl_step % 15 == 0:
        if ctrl_step > 0:
            world_model.observe(prev_raw_action, ego_state_prev, ego_state_now, distances, angles)
        plan_maneuver, plan_target_speed = planner.plan(ego_state_now, world_model)
        cur_lane = int(round(np.clip(ego_state_now["y"], 0.0, 12.0) / 4.0))
        if plan_maneuver == Maneuver.CHANGE_LEFT:
            tgt = max(0, cur_lane-1)
        elif plan_maneuver == Maneuver.CHANGE_RIGHT:
            tgt = min(3, cur_lane+1)
        else:
            tgt = cur_lane
        controller.target_lane_y = tgt * 4.0
        controller.target_speed = plan_target_speed
        controller.current_maneuver = plan_maneuver
        n_tracks = len(world_model.mot.tracks)
        y_now = ego_state_now["y"]
        print("t=%ds maneuver=%d tracks=%d y=%.2f tgt_y=%.1f" % (ctrl_step//15, plan_maneuver, n_tracks, y_now, controller.target_lane_y))
    raw = controller.compute_action(ego_state_now)
    prev_raw_action = raw.copy()
    ego_state_prev = ego_state_now
    env.step(raw)
env.close()