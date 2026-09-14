"""Event-count protocol (spec section 10.3), run in parallel.

Drives each condition until at least TARGET_EVENTS collisions have been observed
(or an episode cap is hit), then reports exposure-time collision rates and a
Poisson RATE RATIO with an exact 95% CI -- not a proportion comparison.

Conditions are seed-paired: every condition sees the same traffic seeds in the
same order, so the flow-vs-viability contrast is not confounded by traffic draw.

Usage:
  python evaluation/evaluate_events.py --workers 10 --target-events 30 \
      --conditions flow:26,viability:26,flow:30,viability:30
"""
import os, sys, json, time, argparse
import multiprocessing as mp
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET_EVENTS = 30
EPISODE_CAP = 400
SEED0 = 20000


def _worker(task):
    """One episode. Imported lazily so spawn-based workers pay the cost once."""
    kind, seed, k_factor, v_limit, rear_rss = task
    from evaluate_vs_idm import run_episode
    crashed, speed, steps, _ = run_episode(kind, seed, k_factor,
                                           rear_rss=rear_rss, v_limit=v_limit)
    return (kind, v_limit, seed, int(crashed), float(speed), int(steps))


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def rate_ratio_ci(x, tx, y, ty):
    """Exact CI for (x/tx)/(y/ty) via the binomial conditional on x+y."""
    n = x + y
    if n == 0:
        return float('nan'), (0.0, float('inf'))
    rr = (x / tx) / (y / ty) if y > 0 else float('inf')
    lo_p, hi_p = clopper_pearson(x, n)
    def tr(p):
        if p >= 1.0: return float('inf')
        if p <= 0.0: return 0.0
        return (p / (1 - p)) * (ty / tx)
    return rr, (tr(lo_p), tr(hi_p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--target-events", type=int, default=TARGET_EVENTS)
    ap.add_argument("--episode-cap", type=int, default=EPISODE_CAP)
    ap.add_argument("--chunk", type=int, default=40, help="episodes per condition per round")
    ap.add_argument("--conditions", type=str,
                    default="flow:26,viability:26,flow:30,viability:30")
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--out", type=str, default="evaluation/results_events.json")
    a = ap.parse_args()

    conds = []
    for spec in a.conditions.split(","):
        kind, _, vl = spec.partition(":")
        conds.append((kind, float(vl) if vl else 30.0))

    state = {c: {"episodes": 0, "events": 0, "steps": 0, "speeds": [],
                 "crash_seeds": []} for c in conds}
    active = list(conds)
    next_seed = {c: SEED0 for c in conds}

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=a.workers) as pool:
        rnd = 0
        while active:
            rnd += 1
            tasks = []
            for c in active:
                kind, vl = c
                for i in range(a.chunk):
                    tasks.append((kind, next_seed[c] + i, a.k, vl, True))
                next_seed[c] += a.chunk

            for kind, vl, seed, crashed, speed, steps in pool.imap_unordered(_worker, tasks, chunksize=1):
                st = state[(kind, vl)]
                st["episodes"] += 1
                st["events"] += crashed
                st["steps"] += steps
                st["speeds"].append(speed)
                if crashed:
                    st["crash_seeds"].append(seed)

            still = []
            for c in active:
                st = state[c]
                done = st["events"] >= a.target_events or st["episodes"] >= a.episode_cap
                print(f"[round {rnd}] {c[0]}:v{c[1]:g}  ep={st['episodes']:4d} "
                      f"events={st['events']:3d} exposure={st['steps']:6d}s "
                      f"speed={np.mean(st['speeds']):.2f}"
                      f"{'  DONE' if done else ''}", flush=True)
                if not done:
                    still.append(c)
            active = still

    print(f"\nwall time {time.time()-t0:.0f}s")
    print("=" * 92)
    print(f"{'condition':<18}{'episodes':>9}{'events':>8}{'exposure(s)':>13}"
          f"{'s/collision':>13}{'per-100s':>10}{'speed':>9}")
    print("-" * 92)
    rows = {}
    for c in conds:
        st = state[c]
        spc = st["steps"] / st["events"] if st["events"] else float('inf')
        rows[c] = st
        print(f"{c[0]+':v'+format(c[1],'g'):<18}{st['episodes']:>9}{st['events']:>8}"
              f"{st['steps']:>13}{spc:>13.1f}{100*st['events']/max(1,st['steps']):>10.3f}"
              f"{np.mean(st['speeds']):>9.2f}")
    print("=" * 92)

    for vl in sorted({c[1] for c in conds}):
        vk, fk = ("viability", vl), ("flow", vl)
        if vk in rows and fk in rows:
            v, f = rows[vk], rows[fk]
            rr, (lo, hi) = rate_ratio_ci(v["events"], v["steps"], f["events"], f["steps"])
            verdict = ("viability SAFER" if hi < 1.0 else
                       "flow SAFER" if lo > 1.0 else "NOT SEPARABLE")
            print(f"\nv_limit={vl:g}  rate ratio viability/flow = {rr:.2f} "
                  f"[{lo:.2f}, {hi:.2f}]  -> {verdict}")
            print(f"           speed  viability {np.mean(v['speeds']):.2f} "
                  f"vs flow {np.mean(f['speeds']):.2f} "
                  f"({np.mean(v['speeds'])/np.mean(f['speeds']):.3f}x)")

    with open(a.out, "w") as fp:
        json.dump({f"{k[0]}:{k[1]}": {kk: vv for kk, vv in v.items() if kk != "speeds"} |
                   {"mean_speed": float(np.mean(v["speeds"]))}
                   for k, v in state.items()}, fp, indent=2)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
