import sys
sys.path.append(".")
import numpy as np
from planner import HierarchicalPlanner, Maneuver
from controller import Maneuver as M

class FakeTrack:
    def __init__(self):
        self.x = [67.5, 4.0, 1.33, 0.0]
        self.hits = 15
        self.is_phantom = False

class FakeWM:
    class FakeMOT:
        tracks = [FakeTrack()]
    mot = FakeMOT()
    class FakeP:
        mu = 5.0
        sigma = 0.1
    a_decel = FakeP()
    a_accel = FakeP()

ego = {"x": 21.4, "y": 4.0, "heading": 0.0, "vx": 22.9, "vy": 0.0}
wm = FakeWM()

planner = HierarchicalPlanner(dt=1.0, k_factor=2.0, t_react=1.0, a_lead_max=5.0)
planner.lanes_count = 4
planner.v_ego_filt = 22.9
planner.v_limit = 30.0

# Manually trace L2
candidates = [M.KEEP_SPEED, M.CHANGE_LEFT, M.CHANGE_RIGHT]
a_brake_pess = max(0.1, abs(wm.a_decel.mu) - planner.k_factor * wm.a_decel.sigma)
print("a_brake_pess =", a_brake_pess)

for m in candidates:
    target_y = ego["y"]
    if m == M.CHANGE_LEFT: target_y -= 4.0
    elif m == M.CHANGE_RIGHT: target_y += 4.0
    
    v_flow = planner._get_lane_v_flow(target_y, wm.mot.tracks)
    
    closest_trk = None
    min_dx = 1e9
    for trk in wm.mot.tracks:
        if abs(trk.x[1] - target_y) < 2.5 and trk.x[0] > ego["x"]:
            dx = trk.x[0] - ego["x"]
            if dx < min_dx:
                min_dx = dx
                closest_trk = trk
    
    if closest_trk:
        v_lead = closest_trk.x[2]
        target_gap = (planner.v_ego_filt * planner.t_react
                      + planner.v_ego_filt**2/(2*a_brake_pess)
                      - v_lead**2/(2*planner.a_lead_max))
        target_gap = max(target_gap, max(10.0, 0.5*planner.v_ego_filt))
        target_gap += 5.0
        v_target = v_lead + 0.2 * (min_dx - target_gap)
        v_target = np.clip(v_target, 0, planner.v_limit)
        score = v_target - (10.0 if m != M.KEEP_SPEED else 0.0)
        print("maneuver=%d target_y=%.1f min_dx=%.1f v_lead=%.2f target_gap=%.1f v_target=%.2f score=%.2f" % (
            m, target_y, min_dx, v_lead, target_gap, v_target, score))
    else:
        v_target = min(v_flow + 5.0, planner.v_limit)
        score = v_target - (10.0 if m != M.KEEP_SPEED else 0.0)
        print("maneuver=%d target_y=%.1f no_track v_flow=%.2f v_target=%.2f score=%.2f" % (
            m, target_y, v_flow, v_target, score))