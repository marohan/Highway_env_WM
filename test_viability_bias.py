"""Spec section 11 — pre-registered falsifiable prediction.

Braking authority (|a_decel| ~ 5 m/s^2) exceeds accelerating authority
(a_accel ~ 3 m/s^2). Risk from the leader is resolvable with the strong
actuator, risk from the follower only with the weak one. A pure option count
should therefore prefer a longitudinal position biased TOWARD THE LEADER
(positive offset), and adding the pain gradient (beta) should shrink that
offset toward zero.

Two things were wrong with the previous version of this file:

1. Frame mismatch. Other vehicles were propagated at (v_traffic - v_ego) while
   viability.get_succ advances ego x by its ABSOLUTE speed. That made FASTER
   score Z=0 (certain death) from the middle of a 60 m gap.
2. It rebuilt the successor tables with pure-python triple loops on every step
   (~2.8e8 iterations for the three-beta sweep), so it never finished.

Both fixed here; the successor tables now come from viability.build_succ_tables.
Run: python test_viability_bias.py
"""
import numpy as np
from config import ViabilityCfg
from occupancy import OccField
from viability import dp_single_world, IDLE, FASTER, SLOWER, N_V, V_MIN
from occupancy import N_CELLS

V_TRAFFIC = 20.0
A_DECEL = 5.0
A_ACCEL = 3.0


def run_test(beta, gap=60.0, start_offset=0.0, steps=150, settle=40, rear_rss=True, verbose=False):
    cfg = ViabilityCfg(beta=beta, T=8, K=1, tie_eps=0.0,
                       x_range=(-200.0, 400.0), rear_rss=rear_rss)
    N_x = int((cfg.x_range[1] - cfg.x_range[0]) / cfg.dx)
    x_grid = cfg.x_range[0] + np.arange(N_x) * cfg.dx

    x_lead = gap / 2.0
    x_follow = -gap / 2.0
    x_ego = start_offset
    v_ego = V_TRAFFIC
    hist = []

    for step in range(steps):
        occ = OccField(1, cfg.T, N_CELLS, N_x, cfg.x_range[0], cfg.dx)
        rel_lead = x_lead - x_ego
        rel_follow = x_follow - x_ego

        for t in range(cfg.T):
            # Absolute-frame propagation, matching get_succ's absolute ego motion.
            p_lead = rel_lead + V_TRAFFIC * t * cfg.dt
            p_follow = rel_follow + V_TRAFFIC * t * cfg.dt
            i_lead = int((p_lead - cfg.x_range[0]) / cfg.dx)
            i_follow = int((p_follow - cfg.x_range[0]) / cfg.dx)
            if 0 <= i_lead < N_x:
                occ.blocked[0, t, 0, i_lead:] = True
            if 0 <= i_follow < N_x:
                occ.blocked[0, t, 0, :i_follow + 1] = True
            occ.gap_ahead[0, t, 0, :] = np.where(p_lead > x_grid,
                                                 np.maximum(0.0, p_lead - x_grid), 1000.0)
            occ.gap_behind[0, t, 0, :] = np.where(x_grid > p_follow,
                                                  np.maximum(0.0, x_grid - p_follow), 1000.0)

        Z, succ, log_offset = dp_single_world(0, A_DECEL, A_ACCEL, occ, cfg)

        v_idx = int(np.clip(round((v_ego - V_MIN) / cfg.dv), 0, N_V - 1))
        x_idx = int(round(-cfg.x_range[0] / cfg.dx))
        s_flat = np.ravel_multi_index((0, v_idx, x_idx), (N_CELLS, N_V, N_x))

        best_a, best_z = IDLE, -1.0
        for a in (IDLE, FASTER, SLOWER):
            s_next = succ[a][s_flat]
            z_val = Z[1, s_next] if s_next >= 0 else -1.0
            if z_val > best_z:
                best_z, best_a = z_val, a

        if verbose and step < 5:
            print(f"    step{step} v={v_ego:4.1f} off={x_ego - (x_lead + x_follow)/2:+6.1f} "
                  f"-> {['IDLE','FASTER','SLOWER'][[IDLE,FASTER,SLOWER].index(best_a)]}")

        if best_a == FASTER:
            v_ego = min(v_ego + A_ACCEL * cfg.dt, 30.0)
        elif best_a == SLOWER:
            v_ego = max(v_ego - A_DECEL * cfg.dt, 10.0)

        x_ego += v_ego * cfg.dt
        x_lead += V_TRAFFIC * cfg.dt
        x_follow += V_TRAFFIC * cfg.dt

        if step >= steps - settle:
            hist.append(x_ego - (x_lead + x_follow) / 2.0)

    return float(np.mean(hist))


def sweep(rear_rss, starts=(-12.0, -6.0, 0.0, 6.0, 12.0), betas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    label = "rear_rss=ON (symmetric DEATH)" if rear_rss else "rear_rss=OFF (pre-fix: gap_behind > 1.0)"
    print(f"\n{label}   gap=60 m, T=8, K=1, tie_eps=0")
    print(f"  {'beta':<6}" + "".join(f"{s:>9.0f}" for s in starts) + f"{'mean':>10}")
    rows = []
    for b in betas:
        offs = [run_test(b, start_offset=s, rear_rss=rear_rss) for s in starts]
        rows.append((b, float(np.mean(offs))))
        print(f"  {b:<6.2f}" + "".join(f"{o:>+9.1f}" for o in offs) + f"{np.mean(offs):>+10.2f}")
    return rows


def logz_profile(beta, gap=60.0, rear_rss=True, span=None, v_ego=V_TRAFFIC):
    """Direct measurement: logZ as a function of ego position inside the gap.

    The closed-loop sweep above can only servo in +-2 m/s steps, so it reports a
    continuum of fixed points rather than a preferred position. This measures
    the objective itself: place the ego at each offset, count surviving futures,
    and read off the argmax. That is what section 11 actually predicts about.
    """
    cfg = ViabilityCfg(beta=beta, T=8, K=1, tie_eps=0.0,
                       x_range=(-200.0, 400.0), rear_rss=rear_rss)
    N_x = int((cfg.x_range[1] - cfg.x_range[0]) / cfg.dx)
    x_grid = cfg.x_range[0] + np.arange(N_x) * cfg.dx
    if span is None:
        span = np.arange(-gap/2 + 4.0, gap/2 - 3.0, 2.0)

    v_idx = int(np.clip(round((v_ego - V_MIN) / cfg.dv), 0, N_V - 1))
    x_idx = int(round(-cfg.x_range[0] / cfg.dx))
    s_flat = np.ravel_multi_index((0, v_idx, x_idx), (N_CELLS, N_V, N_x))

    out = []
    for off in span:
        rel_lead = gap/2 - off
        rel_follow = -gap/2 - off
        occ = OccField(1, cfg.T, N_CELLS, N_x, cfg.x_range[0], cfg.dx)
        for t in range(cfg.T):
            p_lead = rel_lead + V_TRAFFIC * t * cfg.dt
            p_follow = rel_follow + V_TRAFFIC * t * cfg.dt
            i_l = int((p_lead - cfg.x_range[0]) / cfg.dx)
            i_f = int((p_follow - cfg.x_range[0]) / cfg.dx)
            if 0 <= i_l < N_x: occ.blocked[0, t, 0, i_l:] = True
            if 0 <= i_f < N_x: occ.blocked[0, t, 0, :i_f+1] = True
            occ.gap_ahead[0, t, 0, :] = np.where(p_lead > x_grid, np.maximum(0.0, p_lead - x_grid), 1000.0)
            occ.gap_behind[0, t, 0, :] = np.where(x_grid > p_follow, np.maximum(0.0, x_grid - p_follow), 1000.0)
        Z, succ, log_offset = dp_single_world(0, A_DECEL, A_ACCEL, occ, cfg)
        z0 = Z[0, s_flat]
        lz = np.log(z0) + log_offset[0] if z0 > 0 else float('-inf')
        out.append((float(off), float(lz)))
    return out


def report_profiles(rear_rss):
    label = "rear_rss=ON " if rear_rss else "rear_rss=OFF"
    print(f"\n  logZ(position) profile, {label}, gap=60 m, v_ego=20")
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        prof = logz_profile(beta, rear_rss=rear_rss)
        finite = [(o, z) for o, z in prof if np.isfinite(z)]
        if not finite:
            print(f"    beta={beta:<5} (no alive position in gap)")
            continue
        best = max(finite, key=lambda t: t[1])
        zs = np.array([z for _, z in finite]); os_ = np.array([o for o, _ in finite])
        # logZ-weighted centroid: robust to plateaus that a bare argmax hides
        w = np.exp(zs - zs.max())
        centroid = float((w * os_).sum() / w.sum())
        mid = float((os_.min() + os_.max()) / 2.0)
        # Report the bias RELATIVE TO THE ALIVE-BAND MIDPOINT. The band itself is
        # slightly asymmetric because the forward and rear RSS margins differ, and
        # that is a constraint artifact, not the actuation-authority effect under test.
        print(f"    beta={beta:<5} band [{os_.min():+.0f},{os_.max():+.0f}] mid={mid:+5.1f}  "
              f"argmax={best[0]:+6.1f}  BIAS={best[0]-mid:+6.1f} m  "
              f"centroid-bias={centroid-mid:+6.2f} m  spread={zs.max()-zs.min():.3f} nats")


if __name__ == "__main__":
    report_profiles(rear_rss=False)
    report_profiles(rear_rss=True)
    off_rows = sweep(rear_rss=False)
    on_rows = sweep(rear_rss=True)
    print("\nSection-11 prediction: mean offset significantly POSITIVE at beta=0,")
    print("shrinking monotonically toward 0 as beta increases.")
    for name, rows in (("rear_rss=OFF", off_rows), ("rear_rss=ON ", on_rows)):
        b0 = rows[0][1]
        mono = all(abs(rows[i][1]) <= abs(rows[i-1][1]) + 1e-6 for i in range(1, len(rows)))
        print(f"  {name}: beta=0 offset {b0:+.2f} m | sign "
              f"{'POSITIVE' if b0 > 0.5 else 'NEGATIVE' if b0 < -0.5 else 'ZERO'} "
              f"| monotone shrink: {mono}")
