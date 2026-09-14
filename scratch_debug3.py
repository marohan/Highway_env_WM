import sys
sys.path.append(".")
import numpy as np
from planner import HierarchicalPlanner, Maneuver
from world_model import WorldModel

# Simulate exact state
class FakeTrack:
    def __init__(self):
        self.x = [67.5, 4.0, 1.33, 0.0]  # x, y, vx, vy
        self.hits = 15
        self.is_phantom = False

class FakeWorldModel:
    class FakeMOT:
        tracks = [FakeTrack()]
    mot = FakeMOT()
    class FakeParam:
        mu = 5.0
        sigma = 0.1
    a_decel = FakeParam()

ego_state = {"x": 21.4, "y": 4.0, "heading": 0.0, "vx": 22.9, "vy": 0.0}
wm = FakeWorldModel()

planner = HierarchicalPlanner(dt=1.0, k_factor=2.0, t_react=1.0, a_lead_max=5.0)
planner.lanes_count = 4
planner.v_ego_filt = 22.9

# Check L1 for each candidate
print("L1 invariant checks:")
for m in [Maneuver.KEEP_SPEED, Maneuver.CHANGE_LEFT, Maneuver.CHANGE_RIGHT]:
    safe = planner._L1_invariant(ego_state, wm.mot.tracks, m, wm)
    print("  maneuver=%d safe=%s" % (m, safe))

plan_m, plan_v = planner.plan(ego_state, wm)
print("Plan: maneuver=%d v=%.2f" % (plan_m, plan_v))