import numpy as np
from config import ViabilityCfg
from occupancy import OccField, N_CELLS, y_to_cell

# Actions
IDLE = 0
FASTER = 1
SLOWER = 2
LANE_LEFT = 3
LANE_RIGHT = 4
ACTIONS = [IDLE, FASTER, SLOWER, LANE_LEFT, LANE_RIGHT]

N_V = 11
V_MIN = 10.0
V_MAX = 30.0
LOGZ_DEAD = -690.0

def get_succ(lane, v, x, a, a_decel, a_accel, cfg):
    v2 = v
    if a == FASTER: v2 = min(v + a_accel * cfg.dt, V_MAX)
    elif a == SLOWER: v2 = max(v - a_decel * cfg.dt, V_MIN)

    lane2 = lane
    if a == LANE_LEFT: lane2 = lane - 1
    elif a == LANE_RIGHT: lane2 = lane + 1

    x2 = x + 0.5 * (v + v2) * cfg.dt
    return lane2, v2, x2

def clamp_to_lattice(lane, v, x, cfg):
    l_idx = int(lane)
    if l_idx < 0 or l_idx >= N_CELLS: return None

    v_idx = int(round((v - V_MIN) / cfg.dv))
    v_idx = max(0, min(N_V - 1, v_idx))

    x_idx = int(round((x - cfg.x_range[0]) / cfg.dx))
    if x_idx < 0 or x_idx >= int((cfg.x_range[1] - cfg.x_range[0])/cfg.dx): return None

    return l_idx, v_idx, x_idx


# ── Successor tables ──────────────────────────────────────────────────────────
# The gather tables depend only on (a_decel, a_accel, lattice geometry), which
# change slowly across plan steps. Building them with python loops on every
# world of every step was the dominant cost; this is the vectorized + cached
# version (identical semantics to get_succ/clamp_to_lattice above).
_SUCC_CACHE = {}
_SUCC_CACHE_MAX = 64

def build_succ_tables(a_decel, a_accel, N_lane, N_x, cfg):
    key = (round(float(a_decel), 4), round(float(a_accel), 4), N_lane, N_x,
           cfg.dt, cfg.dv, cfg.dx, cfg.x_range)
    hit = _SUCC_CACHE.get(key)
    if hit is not None:
        return hit

    l = np.arange(N_lane)[:, None, None]
    vi = np.arange(N_V)[None, :, None]
    xi = np.arange(N_x)[None, None, :]
    v = (V_MIN + vi * cfg.dv).astype(float)
    x = (cfg.x_range[0] + xi * cfg.dx).astype(float)

    tables = {}
    for a in ACTIONS:
        if a == FASTER:
            v2 = np.minimum(v + a_accel * cfg.dt, V_MAX)
        elif a == SLOWER:
            v2 = np.maximum(v - a_decel * cfg.dt, V_MIN)
        else:
            v2 = v
        v2 = np.broadcast_to(v2, (N_lane, N_V, N_x))

        l2 = np.broadcast_to(l, (N_lane, N_V, N_x)).copy()
        if a == LANE_LEFT: l2 = l2 - 1
        elif a == LANE_RIGHT: l2 = l2 + 1

        x2 = x + 0.5 * (v + v2) * cfg.dt

        ok = (l2 >= 0) & (l2 < N_lane)
        v_idx = np.clip(np.round((v2 - V_MIN) / cfg.dv).astype(int), 0, N_V - 1)
        x_idx = np.round((x2 - cfg.x_range[0]) / cfg.dx).astype(int)
        ok &= (x_idx >= 0) & (x_idx < N_x)

        flat = (np.clip(l2, 0, N_lane - 1) * N_V + v_idx) * N_x + np.clip(x_idx, 0, N_x - 1)
        tables[a] = np.where(ok, flat, -1).reshape(-1)

    if len(_SUCC_CACHE) > _SUCC_CACHE_MAX:
        _SUCC_CACHE.clear()
    _SUCC_CACHE[key] = tables
    return tables


def safe_distance_grids(a_decel, cfg):
    """Forward and rear RSS margins as a function of ego speed (spec section 5).

    DEATH must cover damage symmetrically: being rear-ended is damage too.
    Forward assumes the worst plausible leader (slower than ego, capped at the
    traffic nominal); rear assumes the worst plausible follower (faster than
    ego, floored at the traffic nominal). Ego braking authority enters with
    opposite sign in the two directions, which is exactly the actuation
    asymmetry the section-11 prediction is about.
    """
    a_ego = max(0.1, float(a_decel))
    v_grid = V_MIN + np.arange(N_V) * cfg.dv

    v_lead = np.minimum(v_grid, cfg.v_traffic_nominal)
    d_fwd = (v_grid * cfg.t_react
             + v_grid ** 2 / (2.0 * a_ego)
             - v_lead ** 2 / (2.0 * cfg.a_other_max))
    d_fwd = np.maximum(d_fwd, np.maximum(12.0, v_grid * 0.7))

    if not cfg.rear_rss:
        return d_fwd, np.full(N_V, 1.0)

    v_rear = np.maximum(v_grid, cfg.v_traffic_nominal)
    d_rear = (v_rear * cfg.t_react
              + v_rear ** 2 / (2.0 * cfg.a_other_max)
              - v_grid ** 2 / (2.0 * a_ego))
    d_rear = np.maximum(d_rear, np.maximum(10.0, v_rear * 0.5))
    return d_fwd, d_rear


def dp_single_world(k, a_decel, a_accel, occ: OccField, cfg: ViabilityCfg):
    """Exact backward count of surviving action sequences.

    Returns (Z, succ_tables, log_offset) where Z[t] is normalized to max 1 and
    log_offset[t] is the accumulated log normalizer, so that

        log Z_true[t][s] = log(Z[t][s]) + log_offset[t]

    Without log_offset the per-level normalization makes logZ incomparable
    ACROSS worlds, which silently broke the maximin aggregation in
    score_actions (each world was normalized by its own trajectory of maxima).
    """
    N_lane = occ.N_lane
    N_x = occ.N_x

    succ_tables = build_succ_tables(a_decel, a_accel, N_lane, N_x, cfg)

    Z = np.zeros((cfg.T, N_lane * N_V * N_x), dtype=float)
    log_offset = np.zeros(cfg.T, dtype=float)

    d_safe_grid, d_rear_grid = safe_distance_grids(a_decel, cfg)

    # occ.blocked: [K, T, N_lane, N_x] -> expand to [T, N_lane, N_v, N_x]
    blocked_expand = occ.blocked[k][:, :, np.newaxis, :]
    gap_ahead_expand = occ.gap_ahead[k][:, :, np.newaxis, :]
    gap_behind_expand = occ.gap_behind[k][:, :, np.newaxis, :]
    d_safe_expand = d_safe_grid[np.newaxis, np.newaxis, :, np.newaxis]
    d_rear_expand = d_rear_grid[np.newaxis, np.newaxis, :, np.newaxis]
    v_expand = (V_MIN + np.arange(N_V) * cfg.dv)[np.newaxis, np.newaxis, :, np.newaxis]

    alive = (~blocked_expand) & (gap_ahead_expand > d_safe_expand) & (gap_behind_expand > d_rear_expand)

    # pain (symmetric nociception: min over the two directions)
    min_gap = np.minimum(gap_ahead_expand, gap_behind_expand)
    h = min_gap / np.maximum(1.0, v_expand)
    pain = np.clip(1.0 - h / 2.0, 0.0, 1.0)
    pain = np.where(min_gap > 900.0, 0.0, pain)

    alive_flat = alive.reshape((cfg.T, -1)).astype(float)
    pain_flat = pain.reshape((cfg.T, -1))

    # DP backward
    Z[cfg.T-1] = alive_flat[cfg.T-1]
    log_offset[cfg.T-1] = 0.0

    for t in range(cfg.T-2, -1, -1):
        z_next = Z[t+1]
        z_sum = np.zeros_like(z_next)

        for a in ACTIONS:
            table = succ_tables[a]
            valid = table >= 0
            # Swept volume needs no explicit gate now: LANE_LEFT/RIGHT land on a
            # straddle cell, which predict_occupancy marks blocked whenever
            # either adjacent lane is blocked.
            z_sum[valid] += z_next[table[valid]]

        Z[t] = alive_flat[t] * np.exp(-cfg.beta * pain_flat[t]) * z_sum

        max_z = np.max(Z[t])
        if max_z > 0:
            Z[t] /= max_z
            log_offset[t] = np.log(max_z) + log_offset[t+1]
        else:
            log_offset[t] = log_offset[t+1]

    return Z, succ_tables, log_offset


def score_actions(s0, actions, occ, worlds, cfg: ViabilityCfg):
    scores = {a: 0.0 for a in actions}
    logZ_k = {a: [] for a in actions}

    l_idx = y_to_cell(s0['y'])
    v_idx = int(round((s0['vx'] - V_MIN) / cfg.dv))
    x_idx = int(round(-cfg.x_range[0] / cfg.dx))

    v_idx = max(0, min(N_V - 1, v_idx))

    s0_flat = np.ravel_multi_index((l_idx, v_idx, x_idx), (N_CELLS, N_V, occ.N_x))

    # Read level 1 when the horizon allows it, level 0 otherwise. The level is
    # the same for every candidate, so the offset applied is the same too.
    t_read = 1 if cfg.T > 1 else 0

    for k, w in enumerate(worlds):
        Z, succ_tables, log_offset = dp_single_world(k, w['a_decel'], w['a_accel'], occ, cfg)
        off = log_offset[t_read]

        for a in actions:
            # Map Maneuver constants to DP action semantics
            # Maneuver: FOLLOW=0, KEEP_SPEED=1, CHANGE_LEFT=2, CHANGE_RIGHT=3, DECEL_TO_STOP=4
            if a == 1:
                # KEEP_SPEED: holding speed or easing off are both "stay in lane"
                s_idle = succ_tables[IDLE][s0_flat]
                s_slow = succ_tables[SLOWER][s0_flat]
                z_idle = Z[t_read, s_idle] if s_idle >= 0 else 0.0
                z_slow = Z[t_read, s_slow] if s_slow >= 0 else 0.0
                z_val = max(z_idle, z_slow)
            elif a == 2:
                s_next = succ_tables[LANE_LEFT][s0_flat]
                z_val = Z[t_read, s_next] if s_next >= 0 else 0.0
            elif a == 3:
                s_next = succ_tables[LANE_RIGHT][s0_flat]
                z_val = Z[t_read, s_next] if s_next >= 0 else 0.0
            elif a == 4:
                s_next = succ_tables[SLOWER][s0_flat]
                z_val = Z[t_read, s_next] if s_next >= 0 else 0.0
            else:
                s_next = succ_tables[IDLE][s0_flat]
                z_val = Z[t_read, s_next] if s_next >= 0 else 0.0

            if z_val <= 0.0:
                lz = LOGZ_DEAD
            else:
                lz = float(np.log(z_val) + off)
                lz = max(lz, LOGZ_DEAD)
            logZ_k[a].append(lz)

    w_pess = cfg.pessimism_mean_weight
    for a in actions:
        lz_list = logZ_k[a]
        score = (1 - w_pess) * np.min(lz_list) + w_pess * np.mean(lz_list)
        scores[a] = score

    return scores
