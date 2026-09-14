"""Body swap — does online SELF identification actually earn its keep?

Every previously reported number in this repository was produced with the
self-model priors PINNED to ground truth (mu = -5.0, var = 0.1^2), in order to
isolate the ranking layer. That also means the adaptive/robust machinery was
running in a near-degenerate regime: the world ensemble spanned 4.8-5.0 m/s^2,
a 4% spread, and `k_factor` was consequently a dead knob. So two of the
project's headline claims -- that epistemic uncertainty converts into
conservatism, and that a changed body is recovered zero-shot -- have never been
measured. This measures them.

The ego's physical authority is changed through the environment's
`acceleration_range`, symmetrically so the zero-command point stays at zero.
The controller still emits normalized commands and still divides by the nominal
5.0, so a command means something different in each body -- which is exactly
what a body swap is.

Three arms:

  adaptive       priors start at "I don't know" (mu = -10, sigma = 10), then
                 active calibration probes on an empty road until sigma drops
                 below target, then drives. The claim under test.
  pinned_nominal priors pinned to the NOMINAL body regardless of the real one --
                 an agent that believes it still has its old brakes.
  pinned_oracle  priors pinned to the TRUE body. Upper bound: what knowing buys.

adaptive vs pinned_nominal answers "does identification earn its keep".
adaptive vs pinned_oracle answers "how close does it get to knowing".

Pre-registered prediction: on the weak body, pinned_nominal computes RSS
distances that are too short and collides materially more often; adaptive
recovers most of that gap; on the nominal body all three coincide.

Usage: python evaluation/body_swap.py --workers 10 --episodes 30
"""
import os, sys, json, time, argparse
import multiprocessing as mp
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BODIES = [2.5, 3.5, 5.0, 7.0]     # symmetric |accel| = |decel| authority, m/s^2
NOMINAL = 5.0
ARMS = ["adaptive", "pinned_nominal", "pinned_oracle"]
SIGMA_TARGET = 0.5
V_LIMIT = 26.0


def _make_config(body, vehicles=20, duration=40, density=1.5):
    import numpy as _np
    return {
        "observation": {"type": "Kinematics"},
        "action": {"type": "ContinuousAction",
                   "acceleration_range": [-body, body],
                   "steering_range": [-_np.pi / 4, _np.pi / 4]},
        "simulation_frequency": 15, "policy_frequency": 1,
        "duration": duration, "vehicles_count": vehicles,
        "vehicles_density": density,
    }


def calibrate(body, wm, sigma_target=SIGMA_TARGET, max_steps=60):
    """Active excitation on an empty road until the posteriors are tight."""
    import gymnasium as gym, highway_env
    from ray_sensor import RaySensor
    env = gym.make("highway-v0")
    env.unwrapped.configure(_make_config(body, vehicles=0, duration=max_steps, density=0.0))
    env.reset(seed=99)
    sensor = RaySensor(num_rays=64, max_dist=150.0)   # empty road: rays are cheap
    _, _, prev = sensor.observe(env)
    prev_action = None
    steps = 0
    done = tr = False
    while not (done or tr) and steps < max_steps:
        d, a, now = sensor.observe(env)
        if prev_action is not None:
            wm.observe(prev_action, prev, now, d, a)
        if wm.a_accel.sigma < sigma_target and wm.a_decel.sigma < sigma_target:
            break
        if wm.a_decel.sigma > sigma_target and now['vx'] > 5.0:
            act = np.array([-1.0, 0.0], dtype=np.float32)     # full brake probe
        elif wm.a_accel.sigma > sigma_target:
            act = np.array([0.6, 0.0], dtype=np.float32)      # thrust probe
        else:
            act = np.array([0.0, 0.0], dtype=np.float32)
        env.step(act)
        prev, prev_action = now, act
        steps += 1
    env.close()
    return steps


def _worker(task):
    body, arm, seed, kind = task
    import gymnasium as gym, highway_env
    from world_model import WorldModel
    from controller import EgoController
    from ray_sensor import RaySensor
    from evaluate_vs_idm import make_planner

    wm = WorldModel(dt=1.0)
    calib_steps = 0
    if arm == "adaptive":
        calib_steps = calibrate(body, wm)          # starts from mu=-10, sigma=10
    else:
        believed = NOMINAL if arm == "pinned_nominal" else body
        wm.a_accel.mu, wm.a_accel.var = believed * 0.6, 0.1 ** 2
        wm.a_decel.mu, wm.a_decel.var = -believed, 0.1 ** 2
    wm.kp.mu, wm.kp.var = 0.4, 0.1 ** 2
    wm.kd.mu, wm.kd.var = 0.6, 0.1 ** 2
    identified = float(abs(wm.a_decel.mu))

    env = gym.make("highway-v0")
    env.unwrapped.configure(_make_config(body))
    env.reset(seed=seed)
    wm.reset_episode()

    sensor = RaySensor(num_rays=512, max_dist=150.0)
    planner = make_planner(kind, 2.0, V_LIMIT)
    ctl = EgoController(dt=1.0)

    speeds, spreads = [], []
    prev_state = prev_action = None
    done = tr = False
    info = {}
    while not (done or tr):
        d, a, st = sensor.observe(env)
        if prev_state is not None:
            wm.observe(prev_action, prev_state, st, d, a)
        man, v_t = planner.plan(st, wm)
        w = planner._build_worlds(wm, wm.mot.tracks, planner.cfg)
        vals = [x['a_decel'] for x in w]
        spreads.append(max(vals) - min(vals))
        ctl.set_maneuver(man, st['y'], st['vx'], target_speed=v_t)
        act = ctl.compute_action(st)
        speeds.append(st['vx'])
        _, _, done, tr, info = env.step(act)
        prev_state, prev_action = st, act
    env.close()
    return (body, arm, seed, int(bool(info.get("crashed", False))),
            float(np.mean(speeds)), len(speeds), calib_steps, identified,
            float(np.mean(spreads)))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--kind", type=str, default="viability")
    ap.add_argument("--seed0", type=int, default=40000)
    ap.add_argument("--out", type=str, default="evaluation/results_body_swap.json")
    a = ap.parse_args()

    tasks = [(b, arm, a.seed0 + i, a.kind)
             for b in BODIES for arm in ARMS for i in range(a.episodes)]
    print(f"{len(tasks)} episodes: {len(BODIES)} bodies x {len(ARMS)} arms x {a.episodes}, "
          f"ranker={a.kind}", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(processes=a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1)):
            rows.append(r)
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{len(tasks)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"wall {time.time()-t0:.0f}s\n", flush=True)

    agg = {}
    for body, arm, seed, crashed, spd, steps, calib, ident, spread in rows:
        d = agg.setdefault((body, arm), {"n": 0, "c": 0, "sp": [], "calib": [],
                                         "ident": [], "spread": []})
        d["n"] += 1; d["c"] += crashed; d["sp"].append(spd)
        d["calib"].append(calib); d["ident"].append(ident); d["spread"].append(spread)

    hdr = (f"{'body |a|':>9}{'arm':>16}{'believed':>10}{'calib':>7}"
           f"{'coll.':>8}{'95% CI':>14}{'speed':>8}{'ens.spread':>12}")
    print(hdr); print("-" * len(hdr))
    for body in BODIES:
        for arm in ARMS:
            d = agg[(body, arm)]
            lo, hi = wilson(d["c"], d["n"])
            tag = f"{body:.1f}" + ("  (nominal)" if body == NOMINAL else "")
            print(f"{tag:>9}{arm:>16}{np.mean(d['ident']):>10.2f}"
                  f"{np.mean(d['calib']):>7.1f}"
                  f"{d['c']/d['n']:>8.3f}"
                  f"{'['+format(lo,'.2f')+','+format(hi,'.2f')+']':>14}"
                  f"{np.mean(d['sp']):>8.2f}{np.mean(d['spread']):>12.2f}")
        print()

    with open(a.out, "w") as f:
        json.dump([{"body": b, "arm": ar, "seed": s, "crashed": c, "speed": sp,
                    "steps": st, "calib_steps": ca, "identified_decel": idn,
                    "ensemble_spread": esp}
                   for b, ar, s, c, sp, st, ca, idn, esp in rows], f, indent=2)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
