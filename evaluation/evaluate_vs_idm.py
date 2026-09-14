"""Pareto re-measurement against an OMNISCIENT IDM/MOBIL oracle.

Pre-registered acceptance criterion (written before the run):

  IDM/MOBIL is not a peer. It reads the exact position and velocity of every
  vehicle from the simulator, has no sensing pipeline, no self-model
  uncertainty, and is the same model that drives the surrounding traffic, so it
  is effectively coordinating with copies of itself. Parity with it is the wrong
  bar. The viability agent is required to reach a THRESHOLD relative to it:

      PASS if   collision_rate <= idm_collision_rate + 0.05   (absolute)
           and  mean_speed     >= 0.70 * idm_mean_speed       (70% retention)

  The full curve is printed so any other threshold can be read off directly.

Baselines (spec section 10.2): idm, viability(k), flow-L2 (legacy heuristic),
l1-only (no deliberation layer at all).

Usage: python evaluation/evaluate_vs_idm.py [--episodes N] [--agents a,b,c]
"""
import os, sys, json, time, argparse
import numpy as np
import gymnasium as gym
import highway_env

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model import WorldModel
from planner import HierarchicalPlanner
from controller import EgoController, Maneuver
from ray_sensor import RaySensor
import config as cfgmod

BASE_CONFIG = {
    "observation": {"type": "Kinematics"},
    "action": {"type": "ContinuousAction",
               "acceleration_range": [-5.0, 5.0],
               "steering_range": [-np.pi / 4, np.pi / 4]},
    "simulation_frequency": 15,
    "policy_frequency": 1,
    "duration": 40,
    "vehicles_count": 20,
    "vehicles_density": 1.5,
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def make_planner(kind, k_factor, v_limit=30.0):
    pl = HierarchicalPlanner(dt=1.0, k_factor=k_factor)
    pl.v_limit = v_limit
    pl.cfg = cfgmod.ViabilityCfg(legacy_L2=(kind == "flow"))
    return pl


def run_episode(kind, seed, k_factor, rear_rss=True, shared=None, v_limit=30.0):
    env = gym.make("highway-v0")
    cfg = dict(BASE_CONFIG)
    if kind == "idm":
        cfg = dict(cfg, action={"type": "DiscreteMetaAction"})
    env.unwrapped.configure(cfg)
    obs, info = env.reset(seed=seed)

    if kind == "idm":
        from highway_env.vehicle.behavior import IDMVehicle
        ego = env.unwrapped.vehicle
        new_ego = IDMVehicle.create_from(ego)
        env.unwrapped.vehicle = new_ego
        env.unwrapped.road.vehicles.remove(ego)
        env.unwrapped.road.vehicles.append(new_ego)
        speeds = []
        done = tr = False
        while not (done or tr):
            speeds.append(env.unwrapped.vehicle.speed)
            obs, r, done, tr, info = env.step(None)
        env.close()
        return bool(info.get("crashed", False)), float(np.mean(speeds)), len(speeds), None

    # Model-based agent. Self-model priors are pinned so this measures the L2
    # layer, not the calibration phase (calibration is evaluated separately).
    wm = WorldModel(dt=1.0)
    wm.a_accel.mu, wm.a_accel.var = 2.0, 0.1 ** 2
    wm.a_decel.mu, wm.a_decel.var = -5.0, 0.1 ** 2
    wm.kp.mu, wm.kp.var = 0.4, 0.1 ** 2
    wm.kd.mu, wm.kd.var = 0.6, 0.1 ** 2

    sensor = RaySensor(num_rays=512, max_dist=150.0)
    planner = make_planner("flow" if kind == "flow" else "viability", k_factor, v_limit)
    planner.cfg.rear_rss = rear_rss
    if shared is not None:
        planner.tie_stats = shared
    ctl = EgoController(dt=1.0)

    speeds = []
    done = tr = False
    prev_state, prev_action = None, None
    while not (done or tr):
        dists, angles, st = sensor.observe(env)
        if prev_state is not None:
            wm.observe(prev_action, prev_state, st, dists, angles)

        if kind == "l1only":
            # No deliberation: L0 reflex + L1 invariant on KEEP_SPEED only,
            # fixed target speed. This is spec section 10.2's "adaptive-RSS only".
            reflex = planner._L0_reflex(st, wm.mot.tracks)
            if reflex is not None:
                man, v_t = reflex
            elif planner._L1_invariant(st, wm.mot.tracks, Maneuver.KEEP_SPEED, wm):
                man, v_t = Maneuver.KEEP_SPEED, planner.v_limit
            else:
                man, v_t = Maneuver.KEEP_SPEED, max(0.0, st['vx'] - 5.0)
        else:
            man, v_t = planner.plan(st, wm)

        ctl.set_maneuver(man, st['y'], st['vx'], target_speed=v_t)
        act = ctl.compute_action(st)
        speeds.append(st['vx'])
        obs, r, done, tr, info = env.step(act)
        prev_state, prev_action = st, act

    env.close()
    return bool(info.get("crashed", False)), float(np.mean(speeds)), len(speeds), planner.tie_stats


def evaluate(kind, episodes, k_factor=2.0, rear_rss=True, seed0=1000, v_limit=30.0):
    shared = {'n_plans': 0, 'n_safe_gt1': 0, 'logz_decided': 0, 'heuristic_decided': 0,
              'all_dead': 0, 'all_tied': 0, 'tie_sizes': [], 'safe_sizes': [], 'score_spread': []}
    crashes, sp, steps = 0, [], 0
    per_seed = {}
    for i in range(episodes):
        seed = seed0 + i
        c, s, n, _ = run_episode(kind, seed, k_factor, rear_rss,
                                 shared=None if kind in ("idm", "l1only") else shared,
                                 v_limit=v_limit)
        crashes += int(c); sp.append(s); steps += n
        per_seed[seed] = s
    lo, hi = wilson(crashes, episodes)
    return {
        "kind": kind, "k": k_factor, "v_limit": v_limit, "episodes": episodes,
        "collisions": crashes, "collision_rate": crashes / episodes,
        "collision_ci": [lo, hi],
        "collisions_per_100_steps": 100.0 * crashes / max(1, steps),
        "mean_speed": float(np.mean(sp)), "speed_sd": float(np.std(sp)),
        "steps": steps, "per_seed_speed": per_seed,
        "tie_stats": {k: v for k, v in shared.items() if not isinstance(v, list)},
        "mean_tie_size": float(np.mean(shared['tie_sizes'])) if shared['tie_sizes'] else None,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--agents", type=str,
                    default="idm,l1only,flow,viability:1.0,viability:2.0,viability:3.0")
    ap.add_argument("--out", type=str, default="evaluation/results_vs_idm.json")
    ap.add_argument("--rear-rss", type=int, default=1,
                    help="1 = symmetric rear RSS in the alive predicate (fixed), 0 = pre-fix")
    a = ap.parse_args()

    results = []
    for spec in a.agents.split(","):
        parts = spec.split(":")
        kind = parts[0]
        kf = float(parts[1]) if len(parts) > 1 and parts[1] else 2.0
        vl = float(parts[2]) if len(parts) > 2 and parts[2] else 30.0
        t0 = time.time()
        r = evaluate(kind, a.episodes, k_factor=kf, rear_rss=bool(a.rear_rss), v_limit=vl)
        r["wall_s"] = time.time() - t0
        results.append(r)
        print(f"{spec:<18} col={r['collision_rate']:.3f} "
              f"[{r['collision_ci'][0]:.2f},{r['collision_ci'][1]:.2f}] "
              f"speed={r['mean_speed']:5.2f}  ({r['wall_s']:.0f}s)", flush=True)

    # Reference point: measured IDM oracle from the paired full run (30 episodes,
    # seeds 1000-1029) so that partial sweeps can still be scored.
    IDM_REF = {"collision_rate": 0.000, "mean_speed": 20.40}
    idm = next((r for r in results if r["kind"] == "idm"), IDM_REF)
    print("\n" + "=" * 78)
    print(f"{'agent':<18}{'coll.rate':>11}{'95% CI':>16}{'speed':>9}{'ret.':>8}{'verdict':>10}")
    print("-" * 78)
    for r in results:
        ret = r["mean_speed"] / idm["mean_speed"] if idm else float('nan')
        if r["kind"] == "idm":
            verdict = "oracle"
        else:
            ok = (r["collision_rate"] <= idm["collision_rate"] + 0.05) and (ret >= 0.70)
            verdict = "PASS" if ok else "FAIL"
        tag = r['kind'] + (f":k{r['k']:g}/v{r['v_limit']:g}" if r['kind'] == 'viability' else '')
        print(f"{tag:<18}"
              f"{r['collision_rate']:>11.3f}"
              f"{'['+format(r['collision_ci'][0],'.2f')+','+format(r['collision_ci'][1],'.2f')+']':>16}"
              f"{r['mean_speed']:>9.2f}{ret:>8.2f}{verdict:>10}")
    print("=" * 78)
    print("criterion: collision_rate <= IDM+0.05  AND  speed retention >= 0.70")

    for r in results:
        ts = r["tie_stats"]
        if ts.get("n_safe_gt1"):
            print(f"\n{r['kind']}:{r['k']} tie-set: {ts['n_plans']} plans, "
                  f"{ts['n_safe_gt1']} with >1 safe option -> "
                  f"logZ decided {ts['logz_decided']} "
                  f"({100*ts['logz_decided']/ts['n_safe_gt1']:.0f}%), "
                  f"heuristic decided {ts['heuristic_decided']}, "
                  f"all-dead {ts['all_dead']}, mean tie size {r['mean_tie_size']:.2f}")

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {a.out}")
