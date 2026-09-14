import numpy as np
from config import ViabilityCfg

# ── Half-lane lattice ────────────────────────────────────────────────────────
# The lane axis is discretised at HALF-lane resolution: 4 lane centres and the
# 3 straddle positions between them, 7 cells total.
#
#   cell 2l     = centre of lane l
#   cell 2l + 1 = straddling lanes l and l+1
#
# This exists to fix the grid/controller mismatch. The DP used to model
# LANE_LEFT as completing within one 1 s step, while EgoController caps
# heading_ref at 0.08 rad, giving ~1.8 m/s lateral at highway speed and a real
# lane change of roughly 2.5-3 s. The agent was counting the futures of a
# teleporting body it does not have. Routing every lane change through a
# straddle cell makes the manoeuvre cost two steps and makes the swept volume
# structural rather than a bolted-on gate: a straddle cell is blocked whenever
# EITHER adjacent lane is blocked, which is exactly what "the body occupies both
# lanes during the change" means.
N_LANES = 4
N_CELLS = 2 * N_LANES - 1   # 7


def lane_to_cells(lane, sweep=True):
    """Cells a vehicle sitting in `lane` makes unavailable."""
    c = 2 * lane
    if not sweep:
        return [c]
    return [x for x in (c - 1, c, c + 1) if 0 <= x < N_CELLS]


def y_to_cell(y, lane_width=4.0):
    """Metric lateral position -> half-lane cell index."""
    return int(np.clip(round(y / (lane_width / 2.0)), 0, N_CELLS - 1))


class OccField:
    def __init__(self, K, T, N_lane, N_x, x_min, dx):
        self.K = K
        self.T = T
        self.N_lane = N_lane
        self.N_x = N_x
        self.x_min = x_min
        self.dx = dx
        self.blocked = np.zeros((K, T, N_lane, N_x), dtype=bool)
        self.gap_ahead = np.full((K, T, N_lane, N_x), 1000.0, dtype=float)
        self.gap_behind = np.full((K, T, N_lane, N_x), 1000.0, dtype=float)

def predict_occupancy(tracks, ego_x, cfg: ViabilityCfg, world_k_intents=None) -> OccField:
    N_lane = N_CELLS
    N_x = int((cfg.x_range[1] - cfg.x_range[0]) / cfg.dx)
    x_min = cfg.x_range[0]
    sweep = not cfg.no_sweep

    occ = OccField(cfg.K, cfg.T, N_lane, N_x, x_min, cfg.dx)

    L_other = 5.0
    L_ego = 5.0
    half_L = (L_other + L_ego) / 2.0
    x_grid = x_min + np.arange(N_x) * cfg.dx

    for trk in tracks:
        if trk.time_since_update * 1.0 > cfg.t_forget and not cfg.no_memory:
            continue
        cur_y = trk.x[1]
        cur_lane = int(round(np.clip(cur_y / 4.0, 0, N_LANES - 1)))
        trk_x = trk.x[0] - ego_x
        trk_vx = trk.x[2]

        for k in range(cfg.K):
            intent = 0
            if world_k_intents is not None and k < len(world_k_intents):
                intent = world_k_intents[k].get(trk.id, 0)

            pred_x = trk_x
            pred_v = trk_vx
            for t in range(cfg.T):
                pred_x += pred_v * cfg.dt
                if intent == 3: pred_v = max(0, pred_v - 5.0 * cfg.dt)
                occ_lanes = [cur_lane]
                if intent == 1 and not cfg.no_sweep:
                    if cur_lane > 0:
                        occ_lanes.append(cur_lane - 1)
                        if t > 1: occ_lanes = [cur_lane - 1]
                elif intent == 2 and not cfg.no_sweep:
                    if cur_lane < N_LANES - 1:
                        occ_lanes.append(cur_lane + 1)
                        if t > 1: occ_lanes = [cur_lane + 1]

                x_front = pred_x + half_L
                x_rear = pred_x - half_L
                idx_start = int((x_rear - x_min) / cfg.dx)
                idx_end = int((x_front - x_min) / cfg.dx) + 1
                idx_start = max(0, min(N_x-1, idx_start))
                idx_end = max(0, min(N_x, idx_end))

                cells = set()
                for l in occ_lanes:
                    cells.update(lane_to_cells(l, sweep))
                for c in cells:
                    occ.blocked[k, t, c, idx_start:idx_end] = True

    # Longitudinal gaps, vectorized over x (this axis grew 1.75x with the
    # half-lane cells, so the old python inner loop is no longer free).
    for k in range(cfg.K):
        for t in range(cfg.T):
            for l in range(N_lane):
                b = occ.blocked[k, t, l, :]
                if not np.any(b): continue
                idx = np.arange(N_x)

                # nearest blocked cell at or ahead of i
                nxt = np.where(b, idx, N_x)
                nxt = np.minimum.accumulate(nxt[::-1])[::-1]
                has_ahead = nxt < N_x
                gap_a = np.where(has_ahead,
                                 (np.clip(nxt, 0, N_x - 1) - idx) * cfg.dx - half_L,
                                 1000.0)
                occ.gap_ahead[k, t, l, :] = np.minimum(
                    occ.gap_ahead[k, t, l, :], np.maximum(0.0, gap_a))

                # nearest blocked cell at or behind i
                prv = np.where(b, idx, -1)
                prv = np.maximum.accumulate(prv)
                has_behind = prv >= 0
                gap_b = np.where(has_behind,
                                 (idx - np.clip(prv, 0, N_x - 1)) * cfg.dx - half_L,
                                 1000.0)
                occ.gap_behind[k, t, l, :] = np.minimum(
                    occ.gap_behind[k, t, l, :], np.maximum(0.0, gap_b))

    return occ
