import sys
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

# Simulate 1 second to get track data
ego_state_prev = None
prev_raw_action = np.array([0.0, 0.0])
for ctrl_step in range(15):
    distances, angles, ego_state_now = sensor.observe(env)
    if ctrl_step == 0:
        ego_state_prev = ego_state_now
    if ctrl_step > 0:
        world_model.observe(prev_raw_action, ego_state_prev, ego_state_now, distances, angles)
    prev_raw_action = controller.compute_action(ego_state_now)
    ego_state_prev = ego_state_now
    env.step(prev_raw_action)

# At t=1s, inspect what the planner sees
distances, angles, ego_state_now = sensor.observe(env)
world_model.observe(prev_raw_action, ego_state_prev, ego_state_now, distances, angles)
print("Ego:", ego_state_now)
print("Tracks:")
for i, trk in enumerate(world_model.mot.tracks):
    print("  trk[%d]: x=%s y=%.2f vx=%.2f vy=%.2f hits=%d" % (i, trk.x[0], trk.x[1], trk.x[2], trk.x[3], trk.hits))

# Now plan
plan_m, plan_v = planner.plan(ego_state_now, world_model)
print("Plan: maneuver=%d, v=%f" % (plan_m, plan_v))
env.close()