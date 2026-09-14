"""S1 — Option trap (spec section 12, never previously implemented).

Geometry. The ego starts in lane 1 inside a 20 m/s platoon. Lane 0 (left) is
EMPTY for D_trap metres and therefore both RSS-clean and faster right now, but a
4 m/s jam waits at the end of it. Lanes 1-3 carry dense 20 m/s platoons spaced
below the RSS return threshold, so once the ego has entered lane 0 it cannot
merge back out before reaching the jam.

Pre-registered predictions (spec section 12, written before this file existed):

  flow-L2, l1-only : ENTER  (lane 0 satisfies RSS now and scores faster)
  viability-L2     : DO NOT ENTER (the option count collapses past the entry)

The scenario records two different things, and the distinction is the point:

  1. the DECISION  - did the agent enter lane 0, and what did it cost
  2. the MECHANISM - what logZ(CHANGE_LEFT) - logZ(KEEP_SPEED) was at the
     decision moment

Only (2) can tell an option-value refusal apart from a refusal produced by the
tie-break's -3.0 lane-change penalty. `viability_myopic` (T=1, cfg.no_option_value)
is the control: it keeps every penalty and every RSS check but cannot see past
one step, so if it refuses too, the refusal is not option value.

Other vehicles are scripted at constant velocity (position overwritten each
step) rather than left as reactive IDM agents, so the trap cannot dissolve
itself and every run is exactly reproducible.

Usage: python evaluation/scenario_s1.py
"""
import os, sys, json, argparse
import numpy as np
import gymnasium as gym
import highway_env

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model import WorldModel
from planner import HierarchicalPlanner
from controller import EgoController, Maneuver
from ray_sensor import RaySensor
import config as cfgmod

LANE_W = 4.0
V_PLATOON = 20.0
V_JAM = 4.0
SPACING_BLOCK = 22.0     # < RSS return threshold at 20 m/s -> merging back is barred
SPACING_LOOSE = 30.0
DURATION = 35


def build_layout(d_trap, jam_speed=V_JAM, hole=70.0):
    """(lane, x_offset, speed) for every scripted vehicle.

    The ego sits in a `hole`-metre gap in the lane-1 platoon so that STAYING is
    RSS-feasible; everywhere else the spacing is below the RSS return threshold
    so that COMING BACK is not. Without the hole the ego's own lane is also
    infeasible at 26 m/s (RSS needs ~43 m there) and every candidate scores
    LOGZ_DEAD -- which conflates "the objective is blind to the trap" with
    "the objective has nothing to say anywhere", two very different findings.
    """
    veh = []
    # lane 0: empty until d_trap, then the jam
    for i in range(8):
        veh.append((0, d_trap + i * 12.0, jam_speed))
    # lane 1 (ego's lane): a hole around the ego, dense platoon beyond it
    for x in np.arange(hole, 320.0, SPACING_BLOCK):
        veh.append((1, float(x), V_PLATOON))
    for x in np.arange(-hole - 45.0, -hole, SPACING_BLOCK):
        veh.append((1, float(x), V_PLATOON))
    # lane 2: dense enough that escaping right is not free
    for x in np.arange(10.0, 300.0, SPACING_BLOCK):
        veh.append((2, float(x), V_PLATOON))
    # lane 3: looser, two lanes away
    for x in np.arange(20.0, 300.0, SPACING_LOOSE):
        veh.append((3, float(x), V_PLATOON))
    return veh


def make_agent(kind, v_limit):
    wm = WorldModel(dt=1.0)
    wm.a_accel.mu, wm.a_accel.var = 2.0, 0.1 ** 2
    wm.a_decel.mu, wm.a_decel.var = -5.0, 0.1 ** 2
    wm.kp.mu, wm.kp.var = 0.4, 0.1 ** 2
    wm.kd.mu, wm.kd.var = 0.6, 0.1 ** 2
    pl = HierarchicalPlanner(dt=1.0, k_factor=2.0)
    pl.v_limit = v_limit
    pl.cfg = cfgmod.ViabilityCfg(
        legacy_L2=(kind == "flow"),
        no_option_value=(kind == "viability_myopic"),
    )
    return wm, pl, EgoController(dt=1.0)


def run(kind, d_trap, v0, v_limit=26.0, seed=7, jam_speed=V_JAM):
    env = gym.make("highway-v0")
    env.unwrapped.configure({
        "observation": {"type": "Kinematics"},
        "action": {"type": "ContinuousAction",
                   "acceleration_range": [-5.0, 5.0],
                   "steering_range": [-np.pi / 4, np.pi / 4]},
        "simulation_frequency": 15, "policy_frequency": 1,
        "duration": DURATION, "lanes_count": 4,
        "vehicles_count": 60, "initial_lane_id": 1,
    })
    obs, info = env.reset(seed=seed)
    ego = env.unwrapped.vehicle
    ego.speed = v0
    x0 = ego.position[0]

    layout = build_layout(d_trap, jam_speed)
    others = [v for v in env.unwrapped.road.vehicles if v is not ego]
    scripted = []
    for i, (lane, dx, spd) in enumerate(layout):
        if i >= len(others):
            break
        v = others[i]
        scripted.append((v, lane, x0 + dx, spd))
    for v in others[len(scripted):]:      # park the surplus far behind, out of play
        v.position = env.unwrapped.road.network.get_lane(('0', '1', 3)).position(x0 - 400.0, 0)
        v.speed = 0.0

    def place(t):
        for v, lane, x_init, spd in scripted:
            lane_idx = ('0', '1', lane)
            ln = env.unwrapped.road.network.get_lane(lane_idx)
            v.position = ln.position(x_init + spd * t, 0)
            v.lane_index = lane_idx
            v.target_lane_index = lane_idx
            v.lane = ln
            v.heading = 0.0
            v.speed = spd
            v.target_speed = spd
    place(0.0)

    wm, planner, ctl = make_agent(kind, v_limit)
    sensor = RaySensor(num_rays=512, max_dist=150.0)

    trace = []
    entered, entry_step = False, None
    delta_at_choice = None     # logZ(LEFT) - logZ(KEEP) on the step CHANGE_LEFT was chosen
    delta_first_legal = None   # ... and on the first step entry was even legal
    first_legal = None
    jam_seen_at_choice = None  # nearest lane-0 track distance when the choice was made
    prev_state, prev_action = None, None
    crashed = False
    done = tr = False
    step = 0

    while not (done or tr) and step < DURATION:
        dists, angles, st = sensor.observe(env)
        if prev_state is not None:
            wm.observe(prev_action, prev_state, st, dists, angles)

        if kind == "l1only":
            reflex = planner._L0_reflex(st, wm.mot.tracks)
            if reflex is not None:
                man, v_t = reflex
            else:
                # greedy: take the left lane whenever L1 permits it
                if planner._L1_invariant(st, wm.mot.tracks, Maneuver.CHANGE_LEFT, wm) and st['y'] > 2.0:
                    man, v_t = Maneuver.CHANGE_LEFT, v_limit
                elif planner._L1_invariant(st, wm.mot.tracks, Maneuver.KEEP_SPEED, wm):
                    man, v_t = Maneuver.KEEP_SPEED, v_limit
                else:
                    man, v_t = Maneuver.KEEP_SPEED, max(0.0, st['vx'] - 5.0)
        else:
            man, v_t = planner.plan(st, wm)

        # mechanism probe: is CHANGE_LEFT legal, and what does the objective say?
        left_legal = (st['y'] > 2.0) and planner._L1_invariant(st, wm.mot.tracks, Maneuver.CHANGE_LEFT, wm)
        sc = getattr(planner, 'last_scores', {}) or {}
        d_lz = None
        if Maneuver.CHANGE_LEFT in sc and Maneuver.KEEP_SPEED in sc:
            d_lz = sc[Maneuver.CHANGE_LEFT] - sc[Maneuver.KEEP_SPEED]
        if left_legal and first_legal is None:
            first_legal = step
            delta_first_legal = d_lz
        if man == Maneuver.CHANGE_LEFT and delta_at_choice is None:
            delta_at_choice = d_lz
            # Could the agent even see the jam? 512 rays reach 150 m; the trap
            # may sit beyond that, in which case refusing is impossible for
            # perception reasons and the planner is not on trial.
            fwd = [t.x[0] - st['x'] for t in wm.mot.tracks
                   if abs(t.x[1] - 0.0) < 2.0 and t.x[0] > st['x'] and not getattr(t, 'is_phantom', False)]
            jam_seen_at_choice = round(min(fwd), 1) if fwd else None

        lane_now = st['y'] / LANE_W
        if lane_now < 0.5 and not entered:
            entered, entry_step = True, step

        trace.append({"t": step, "x": st['x'] - x0, "y": round(st['y'], 2),
                      "v": round(st['vx'], 2), "man": man,
                      "left_legal": bool(left_legal),
                      "d_logz": None if d_lz is None else round(d_lz, 3)})

        ctl.set_maneuver(man, st['y'], st['vx'], target_speed=v_t)
        act = ctl.compute_action(st)
        obs, r, done, tr, info = env.step(act)
        crashed = crashed or bool(info.get("crashed", False))
        step += 1
        place(float(step))
        prev_state, prev_action = st, act

    env.close()
    speeds = [p["v"] for p in trace]
    return {
        "kind": kind, "d_trap": d_trap, "v0": v0, "jam_speed": jam_speed,
        "entered_left": entered, "entry_step": entry_step,
        "crashed": crashed,
        "distance": round(trace[-1]["x"], 1) if trace else 0.0,
        "mean_speed": round(float(np.mean(speeds)), 2),
        "min_speed": round(float(np.min(speeds)), 2),
        "steps_below_10": int(sum(1 for v in speeds if v < 10.0)),
        "left_ever_legal": first_legal is not None,
        "first_legal_step": first_legal,
        "d_logz_at_first_legal": delta_first_legal,
        "d_logz_at_choice": delta_at_choice,
        "nearest_lane0_at_choice": jam_seen_at_choice,
        "trap_within_sensor": d_trap <= 150.0,
        "trace": trace,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/results_s1.json")
    a = ap.parse_args()

    kinds = ["l1only", "flow", "viability", "viability_myopic"]
    rows = []
    SWEEP = [(d, j) for j in (4.0, 12.0) for d in (90.0, 130.0, 170.0)]
    hdr = (f"{'agent':<18}{'trap':>6}{'jam':>5}{'see?':>6}{'enter':>7}{'crash':>7}"
           f"{'dist':>7}{'meanV':>7}{'minV':>7}{'jam@ch':>8}{'dlogZ@ch':>10}")
    print(hdr); print("-" * len(hdr))
    for d_trap, jam_v in SWEEP:
        for v0 in (20.0,):
            for kind in kinds:
                r = run(kind, d_trap, v0, jam_speed=jam_v)
                rows.append(r)
                dz = r["d_logz_at_choice"]
                jam = r["nearest_lane0_at_choice"]
                print(f"{kind:<18}{d_trap:>6.0f}{jam_v:>5.0f}"
                      f"{('in' if r['trap_within_sensor'] else 'OUT'):>6}"
                      f"{('YES' if r['entered_left'] else '-'):>7}"
                      f"{('YES' if r['crashed'] else '-'):>7}"
                      f"{r['distance']:>7.0f}{r['mean_speed']:>7.2f}{r['min_speed']:>7.2f}"
                      f"{('-' if jam is None else format(jam, '.0f')):>8}"
                      f"{('n/a' if dz is None else format(dz, '+.2f')):>10}", flush=True)
            print()
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {a.out}")
