import os
import numpy as np
from controller import Maneuver

# Long event-count runs produce hundreds of thousands of AEB lines; keep the
# trace but make it opt-in.
VERBOSE = bool(os.environ.get("WM_VERBOSE"))

class HierarchicalPlanner:
    """
    3-Layer Safety Architecture replacing the old Lexicographic Planner.
    L0: Reflex (TTC < t_react -> Max Brake)
    L1: Invariant (RSS safety distance check - Hard Bounds)
    L1.5: Soft Cost (Phantoms, Unconfirmed tracks, Flow Matching, Pessimism)
    L2: Deliberation (Lane selection based on flow and safety)
    """
    def __init__(self, dt=1.0, k_factor=2.0, t_react=0.6, a_lead_max=5.0):
        self.dt = dt
        self.t_react = t_react
        self.a_lead_max = a_lead_max
        self.k_factor = k_factor
        self.disable_safety = False
        self.v_limit = 30.0
        self.v_ego_filt = 0.0
        self.first_step = True
        self.last_scores = {}
        # Instrumentation: how often does logZ actually decide the maneuver,
        # versus handing an unresolved tie to the flow heuristic?
        self.tie_stats = {
            'n_plans': 0,          # L2 viability deliberations reached
            'n_safe_gt1': 0,       # ... of which L1 left more than one option
            'logz_decided': 0,     # ... and logZ narrowed it to a singleton
            'heuristic_decided': 0,# ... and the tie-break heuristic chose
            'all_dead': 0,         # every candidate scored LOGZ_DEAD (Z flat at 0)
            'all_tied': 0,         # every candidate inside tie_eps
            'tie_sizes': [],
            'safe_sizes': [],
            'score_spread': [],    # max(logZ) - min(logZ) across candidates
        }

    def _L0_reflex(self, ego_state, mot_tracks):
        """Reflex layer: Emergency braking based on TTC."""
        min_ttc = float('inf')
        for trk in mot_tracks:
            dx = trk.x[0] - ego_state['x']
            dy = abs(trk.x[1] - ego_state['y'])
            dvx = ego_state['vx'] - trk.x[2]

            # Skip unconfirmed tracks unless they are close (< 20m) ??AEB must fire immediately
            if trk.hits < 3 and dx > 20.0:
                continue

            # Physical collision corridor: vehicle width is 2.0m, so dy < 1.8m is direct path overlap.
            # Vehicles in adjacent lanes (4.0m separation) must never trigger emergency AEB.
            in_lane = (dy < 1.8)

            if dx > 0 and in_lane and dvx > 0:
                ttc = dx / dvx
                if ttc < min_ttc:
                    min_ttc = ttc
                    min_ttc_dx = dx

        if min_ttc < 1.5:
            if VERBOSE:
                print(f"AEB Triggered! dx={min_ttc_dx:.1f}m ttc={min_ttc:.2f}s")
            return Maneuver.DECEL_TO_STOP, 0.0
        return None

    def _L1_invariant(self, ego_state, mot_tracks, maneuver, world_model):
        """Invariant layer: Closed-form RSS safety check. Prunes unsafe actions."""
        if self.disable_safety:
            return True

        # Pure World Model authority: use estimated decel, clamped only to avoid div-by-zero
        a_brake = max(0.1, abs(world_model.a_decel.mu))
        a_lead_max = self.a_lead_max

        current_lane_idx = int(round(np.clip(ego_state['y'] / 4.0, 0, 3)))
        if maneuver == Maneuver.CHANGE_LEFT:
            target_lane_idx = max(0, current_lane_idx - 1)
        elif maneuver == Maneuver.CHANGE_RIGHT:
            target_lane_idx = min(3, current_lane_idx + 1)
        else:
            target_lane_idx = current_lane_idx
        target_lane_y = target_lane_idx * 4.0

        for trk in mot_tracks:
            if getattr(trk, 'is_phantom', False):
                continue

            dy = abs(trk.x[1] - target_lane_y)
            dx = trk.x[0] - ego_state['x']

            # Vehicles in target lane corridor (dy < 2.0m)
            if dy < 2.0:
                # Require at least 2 consecutive observations so single-frame noise does not block
                if trk.hits < 2:
                    continue

                v_ego = ego_state['vx']
                v_lead = max(0.0, trk.x[2])

                if dx > 0:  # Vehicle ahead in target lane
                    # RSS longitudinal safe distance
                    d_rss = (v_ego * self.t_react
                             + (v_ego ** 2) / (2.0 * a_brake)
                             - (v_lead ** 2) / (2.0 * a_lead_max))
                    d_min = max(12.0, v_ego * 0.7)

                    if maneuver == Maneuver.KEEP_SPEED:
                        # KEEP_SPEED: only block if critically inside d_min
                        if dx < d_min:
                            return False
                    else:
                        # Lane changes: apply RSS d_safe
                        d_safe = max(d_rss, d_min)
                        if dx < d_safe:
                            return False
                else:  # Vehicle behind in target lane (dx < 0)
                    if maneuver != Maneuver.KEEP_SPEED:
                        # Rear vehicle in target lane must have stopping distance
                        v_rear = max(0.0, trk.x[2])
                        d_safe_rear = max(12.0, v_rear * self.t_react)
                        if abs(dx) < d_safe_rear:
                            return False

        # Swept Volume Invariant: during a lane change, the ego occupies BOTH
        # the origin lane and target lane. If a vehicle ahead in the CURRENT lane
        # is too close (< 8.0m), turning would clip its corner before clearing the lane.
        if maneuver in [Maneuver.CHANGE_LEFT, Maneuver.CHANGE_RIGHT]:
            curr_lane_y = current_lane_idx * 4.0
            for trk in mot_tracks:
                if getattr(trk, 'is_phantom', False) or trk.hits < 2:
                    continue
                dy_curr = abs(trk.x[1] - curr_lane_y)
                dx_curr = trk.x[0] - ego_state['x']
                if dy_curr < 2.0 and 0 < dx_curr < 8.0:
                    return False

        return True

    def _L1_5_soft_cost(self, ego_state, target_lane_y, mot_tracks, world_model, v_flow):
        """Soft cost for pessimistic bounds, phantoms, and rear RSS."""
        cost = 0.0
        
        # Pessimistic braking
        a_brake_pessimistic = max(0.1, abs(world_model.a_decel.mu) - self.k_factor * world_model.a_decel.sigma)
        a_lead_max = self.a_lead_max
        
        for trk in mot_tracks:
            dy = abs(trk.x[1] - target_lane_y)
            dx = trk.x[0] - ego_state['x']
            
            if dy < 2.5:
                # Forward tracks (including phantoms)
                if dx > 0:
                    v_ego = ego_state['vx']
                    if getattr(trk, 'is_phantom', False):
                        # For phantoms, assume v_lead is slightly below flow speed
                        v_lead = max(0.0, v_flow - 5.0)
                    else:
                        v_lead = trk.x[2]
                        
                    d_safe = v_ego * self.t_react + (v_ego**2) / (2 * a_brake_pessimistic) - (v_lead**2) / (2 * a_lead_max)
                    d_min = max(10.0, 0.5 * v_ego)
                    d_safe = max(d_safe, d_min)
                    
                    if dx < d_safe:
                        # Soft penalty based on how much we violate the pessimistic bound
                        cost += (d_safe - dx) * 2.0 
                        
                # Rear tracks (Rear RSS)
                elif dx < 0:
                    v_ego = ego_state['vx']
                    v_rear = trk.x[2]
                    
                    # Assume rear vehicle has average braking and 1s reaction
                    a_rear_brake = abs(world_model.a_decel.mu)
                    a_ego_max_brake = abs(world_model.a_decel.mu) + world_model.a_decel.sigma
                    
                    d_safe_rear = v_rear * 1.0 + (v_rear**2) / (2 * a_rear_brake) - (v_ego**2) / (2 * a_ego_max_brake)
                    d_safe_rear = max(d_safe_rear, max(10.0, 0.5 * v_rear))
                    
                    if abs(dx) < d_safe_rear:
                        # If we brake hard, the car behind will hit us
                        cost += (d_safe_rear - abs(dx)) * 1.5
                        
        return cost

    def _get_lane_v_flow(self, target_lane_y, mot_tracks):
        """Calculate median flow speed for a given lane."""
        lane_speeds = []
        for trk in mot_tracks:
            if abs(trk.x[1] - target_lane_y) < 2.5 and not getattr(trk, 'is_phantom', False):
                lane_speeds.append(trk.x[2])
                
        if len(lane_speeds) >= 2:
            return np.median(lane_speeds)
        elif len(lane_speeds) == 1:
            return lane_speeds[0]
        else:
            return self.v_limit # Fallback to limit if empty

    def _L2_deliberation_legacy(self, ego_state, mot_tracks, world_model):
        if self.disable_safety:
            return Maneuver.KEEP_SPEED, self.v_limit

        candidates = [Maneuver.KEEP_SPEED, Maneuver.CHANGE_LEFT, Maneuver.CHANGE_RIGHT]
        safe_candidates = []
        
        # Determine valid lane bounds. 
        # Assume self.lanes_count is set, defaulting to 4.
        max_y = (getattr(self, 'lanes_count', 4) - 1) * 4.0
        
        for m in candidates:
            if m == Maneuver.CHANGE_LEFT and ego_state['y'] < 2.0:
                continue
            if m == Maneuver.CHANGE_RIGHT and ego_state['y'] > max_y - 2.0:
                continue
                
            if self._L1_invariant(ego_state, mot_tracks, m, world_model):
                safe_candidates.append(m)
                
        # 1. Fallback Path: L1 Violated -> FOLLOW (Decel Bias)
        if not safe_candidates:
            target_y = ego_state['y']
            closest_trk = None
            min_dx = float('inf')
            for trk in mot_tracks:
                if getattr(trk, 'is_phantom', False): continue
                # Intent prediction: consider vehicle in our lane if its predicted y in t_react will be in our lane
                pred_y = trk.x[1] + trk.x[3] * self.t_react
                if (abs(trk.x[1] - target_y) < 2.5 or (abs(trk.x[1] - target_y) < 4.5 and abs(pred_y - target_y) < 2.5)):
                    if trk.x[0] > ego_state['x']:
                        dx = trk.x[0] - ego_state['x']
                        if dx < min_dx:
                            min_dx = dx
                            closest_trk = trk
                        
            if closest_trk:
                v_target = max(0.0, closest_trk.x[2] - 5.0) # Decel Bias
            else:
                v_target = max(0.0, ego_state['vx'] - 5.0) # Decel Bias
            return Maneuver.KEEP_SPEED, v_target
            
        best_maneuver = Maneuver.KEEP_SPEED
        best_score = -float('inf')
        best_v_target = 30.0

        kp_dist = 0.2  # Stable proportional gap gain

        # Pure World Model authority: use pessimistic decel estimate (clamped to 0.1 to avoid div-by-zero)
        a_brake_pess = max(0.1, abs(world_model.a_decel.mu) - self.k_factor * world_model.a_decel.sigma)

        for m in safe_candidates:
            target_y = ego_state['y']
            if m == Maneuver.CHANGE_LEFT:  target_y -= 4.0
            elif m == Maneuver.CHANGE_RIGHT: target_y += 4.0

            # Find closest vehicle in target lane
            closest_trk = None
            min_dx = float('inf')
            for trk in mot_tracks:
                if getattr(trk, 'is_phantom', False): continue
                pred_y = trk.x[1] + trk.x[3] * self.t_react
                in_lane = (abs(trk.x[1] - target_y) < 2.5 or
                           (abs(trk.x[1] - target_y) < 4.5 and abs(pred_y - target_y) < 2.5))
                if in_lane and trk.x[0] > ego_state['x']:
                    dx = trk.x[0] - ego_state['x']
                    if dx < min_dx:
                        min_dx = dx
                        closest_trk = trk

            # Gap control: use v_ego_filt for target_gap to decouple from instant braking loops
            v_flow = self._get_lane_v_flow(target_y, mot_tracks)
            if closest_trk:
                v_lead = max(0.0, closest_trk.x[2])
                target_gap = (self.v_ego_filt * self.t_react
                              + (self.v_ego_filt ** 2) / (2.0 * a_brake_pess)
                              - (v_lead ** 2) / (2.0 * self.a_lead_max))
                target_gap = max(target_gap, max(8.0, self.v_ego_filt * 0.5))
                target_gap += 5.0  # Equilibrium slightly above hard bound
                v_target = v_lead + kp_dist * (min_dx - target_gap)
            else:
                v_target = min(v_flow + 5.0, self.v_limit)

            v_target = np.clip(v_target, 0.0, self.v_limit)

            # Score = target speed (higher is better for progress)
            score = v_target

            # L1.5 Soft Cost
            l15_cost = self._L1_5_soft_cost(ego_state, target_y, mot_tracks, world_model, v_flow)
            score -= l15_cost

            # Moderate lane-change penalty to encourage proactive overtaking
            if m != Maneuver.KEEP_SPEED:
                score -= 5.0

            if score > best_score:
                best_score = score
                best_maneuver = m
                best_v_target = v_target

        return best_maneuver, best_v_target

    def plan(self, ego_state, world_model):
        if self.first_step:
            self.v_ego_filt = ego_state['vx']
            self.first_step = False
        else:
            # Low pass filter for v_ego (time constant ~ 2.0s) to decouple from instant braking loops
            alpha = 1.0 - np.exp(-self.dt / 2.0)
            self.v_ego_filt = (1.0 - alpha) * self.v_ego_filt + alpha * ego_state['vx']

        if self.disable_safety:
            return self._L2_deliberation(ego_state, world_model.mot.tracks, world_model)
            
        mot_tracks = world_model.mot.tracks
        
        # 1. Reflex check
        reflex_result = self._L0_reflex(ego_state, mot_tracks)
        if reflex_result is not None:
            return reflex_result
            
        # 2. & 3. Invariant check + Deliberation
        return self._L2_deliberation(ego_state, mot_tracks, world_model)

    def _build_worlds(self, world_model, mot_tracks, cfg):
        import numpy as np
        mu_d = world_model.a_decel.mu
        sig_d = world_model.a_decel.sigma
        mu_a = world_model.a_accel.mu
        sig_a = world_model.a_accel.sigma
        
        worlds = []
        if cfg.K >= 1: worlds.append({'a_decel': abs(mu_d), 'a_accel': abs(mu_a), 'intent_rank': 0})
        if cfg.K >= 2: worlds.append({'a_decel': max(0.1, abs(mu_d) - sig_d), 'a_accel': max(0.1, abs(mu_a) - sig_a), 'intent_rank': 0})
        if cfg.K >= 3: worlds.append({'a_decel': max(0.1, abs(mu_d) - 2*sig_d), 'a_accel': abs(mu_a), 'intent_rank': 0})
        if cfg.K >= 4: worlds.append({'a_decel': abs(mu_d), 'a_accel': abs(mu_a), 'intent_rank': 1})
        if cfg.K >= 5: worlds.append({'a_decel': max(0.1, abs(mu_d) - sig_d), 'a_accel': max(0.1, abs(mu_a) - sig_a), 'intent_rank': 1})
        
        out_worlds = []
        for w in worlds:
            intents = {}
            for trk in mot_tracks:
                if getattr(trk, 'is_phantom', False): continue
                sorted_idx = np.argsort(trk.intent_probs)[::-1]
                if w['intent_rank'] < len(sorted_idx):
                    intents[trk.id] = sorted_idx[w['intent_rank']]
                else:
                    intents[trk.id] = sorted_idx[0]
            out_worlds.append({'a_decel': w['a_decel'], 'a_accel': w['a_accel'], 'intents': intents})
            
        return out_worlds

    def _L2_deliberation(self, ego_state, mot_tracks, world_model):
        import numpy as np
        import config
        from occupancy import predict_occupancy
        from viability import score_actions
        
        if not hasattr(self, 'cfg'):
            self.cfg = config.ViabilityCfg()
            
        if self.cfg.legacy_L2:
            return self._L2_deliberation_legacy(ego_state, mot_tracks, world_model)
            
        if self.disable_safety:
            return Maneuver.KEEP_SPEED, self.v_limit

        # ── Two-level direction memory (biological motor inertia) ─────────────
        # Level 1: commit_steps_left — short-term "in progress" lock (resets fast)
        # Level 2: last_lc_dir / steps_since_lc — persistent direction memory
        #          KEEP_SPEED does NOT erase recent direction memory.
        #
        # flappy_WM principle: if you just moved right, the cost to reverse left
        # is high regardless of a brief pause — muscles haven't reset.
        if not hasattr(self, 'last_lc_dir'):
            self.last_lc_dir = Maneuver.KEEP_SPEED
            self.steps_since_lc = 999
            self.committed_maneuver = Maneuver.KEEP_SPEED
            self.commit_steps_left = 0
        if self.commit_steps_left > 0:
            self.commit_steps_left -= 1
        self.steps_since_lc += 1

        candidates = [Maneuver.KEEP_SPEED, Maneuver.CHANGE_LEFT, Maneuver.CHANGE_RIGHT]
        safe_candidates = []
        max_y = (getattr(self, 'lanes_count', 4) - 1) * 4.0
        
        for m in candidates:
            if m == Maneuver.CHANGE_LEFT and ego_state['y'] < 2.0: continue
            if m == Maneuver.CHANGE_RIGHT and ego_state['y'] > max_y - 2.0: continue
            if self._L1_invariant(ego_state, mot_tracks, m, world_model):
                safe_candidates.append(m)
                
        if not safe_candidates:
            self.committed_maneuver = Maneuver.KEEP_SPEED
            self.commit_steps_left = 0
            target_y = ego_state['y']
            closest_trk = None
            min_dx = float('inf')
            for trk in mot_tracks:
                if getattr(trk, 'is_phantom', False): continue
                pred_y = trk.x[1] + trk.x[3] * self.t_react
                if (abs(trk.x[1] - target_y) < 2.5 or (abs(trk.x[1] - target_y) < 4.5 and abs(pred_y - target_y) < 2.5)):
                    if trk.x[0] > ego_state['x']:
                        dx = trk.x[0] - ego_state['x']
                        if dx < min_dx:
                            min_dx = dx
                            closest_trk = trk
            if closest_trk:
                v_target = max(0.0, closest_trk.x[2] - 5.0)
            else:
                v_target = max(0.0, ego_state['vx'] - 5.0)
            return Maneuver.KEEP_SPEED, v_target
            
        # ── Viability deliberation ────────────────────────────────────────────
        worlds = self._build_worlds(world_model, mot_tracks, self.cfg)
        occ = predict_occupancy(mot_tracks, ego_state['x'], self.cfg, [w['intents'] for w in worlds])
        
        scores = score_actions(ego_state, safe_candidates, occ, worlds, self.cfg)
        max_score = max(scores.values())
        # Expose the raw objective so a scenario can test the MECHANISM
        # (did logZ actually say the left lane is worse?) and not merely the
        # decision, which the tie-break penalties could equally explain.
        self.last_scores = dict(scores)

        # ── Tie-set instrumentation ──────────────────────────────────────────
        from viability import LOGZ_DEAD
        ts = self.tie_stats
        tie_set = [m for m in safe_candidates if scores[m] >= max_score - self.cfg.tie_eps]
        ts['n_plans'] += 1
        ts['safe_sizes'].append(len(safe_candidates))
        ts['tie_sizes'].append(len(tie_set))
        ts['score_spread'].append(float(max_score - min(scores.values())))
        if all(scores[m] <= LOGZ_DEAD + 1e-6 for m in safe_candidates):
            ts['all_dead'] += 1
        if len(tie_set) == len(safe_candidates) and len(safe_candidates) > 1:
            ts['all_tied'] += 1
        if len(safe_candidates) > 1:
            ts['n_safe_gt1'] += 1
            if len(tie_set) == 1:
                ts['logz_decided'] += 1
            else:
                ts['heuristic_decided'] += 1
        
        # ── Commitment override: force current maneuver into tie-set if viable ─
        if (self.commit_steps_left > 0
                and self.committed_maneuver in safe_candidates
                and scores[self.committed_maneuver] >= max_score - self.cfg.tie_eps * 3):
            forced_tie = self.committed_maneuver
        else:
            forced_tie = None
        
        best_maneuver = Maneuver.KEEP_SPEED
        best_flow = -float('inf')
        best_v_target = ego_state['vx']
        
        kp_dist = 0.2
        a_brake_pess = max(0.1, abs(world_model.a_decel.mu) - self.k_factor * world_model.a_decel.sigma)
        
        # Direction memory window: reversal cost decays linearly over 8 steps
        reversal_memory_steps = 8
        
        for m in safe_candidates:
            # Tie-set filter
            if scores[m] < max_score - self.cfg.tie_eps and m != forced_tie:
                continue
                
            target_y = ego_state['y']
            if m == Maneuver.CHANGE_LEFT:  target_y -= 4.0
            elif m == Maneuver.CHANGE_RIGHT: target_y += 4.0
            
            closest_trk = None
            min_dx = float('inf')
            for trk in mot_tracks:
                if getattr(trk, 'is_phantom', False): continue
                pred_y = trk.x[1] + trk.x[3] * self.t_react
                in_lane = (abs(trk.x[1] - target_y) < 2.5 or
                           (abs(trk.x[1] - target_y) < 4.5 and abs(pred_y - target_y) < 2.5))
                if in_lane and trk.x[0] > ego_state['x']:
                    dx = trk.x[0] - ego_state['x']
                    if dx < min_dx:
                        min_dx = dx
                        closest_trk = trk
                        
            v_flow_m = self._get_lane_v_flow(target_y, mot_tracks)
            if closest_trk:
                v_lead = max(0.0, closest_trk.x[2])
                target_gap = (self.v_ego_filt * self.t_react
                              + (self.v_ego_filt ** 2) / (2.0 * a_brake_pess)
                              - (v_lead ** 2) / (2.0 * self.a_lead_max))
                target_gap = max(target_gap, max(8.0, self.v_ego_filt * 0.5))
                target_gap += 5.0
                v_target = v_lead + kp_dist * (min_dx - target_gap)
            else:
                v_target = min(v_flow_m + 5.0, self.v_limit)
                
            v_target = np.clip(v_target, 0.0, self.v_limit)
            
            flow_score = v_target
            if m != Maneuver.KEEP_SPEED:
                # Base lane-change cost
                flow_score -= 3.0

                # Stabilization cooldown: require settling in newly acquired lane for at least 3 steps
                if self.steps_since_lc < 4:
                    flow_score -= 4.0

                # ── Biological motor switching cost ───────────────────────────
                # Memory-weighted reversal penalty:
                # Even after a KEEP_SPEED pause, reversing a recent direction
                # costs energy — muscles haven't fully reset.
                # Cost decays linearly from 5.0 → 0.0 over reversal_memory_steps.
                is_reversal = (
                    (self.last_lc_dir == Maneuver.CHANGE_LEFT  and m == Maneuver.CHANGE_RIGHT) or
                    (self.last_lc_dir == Maneuver.CHANGE_RIGHT and m == Maneuver.CHANGE_LEFT)
                )
                if is_reversal and self.steps_since_lc < reversal_memory_steps:
                    # Linear decay: full cost immediately after reversal, zero after memory window
                    decay = 1.0 - self.steps_since_lc / reversal_memory_steps
                    flow_score -= 5.0 * decay
                
            if flow_score > best_flow:
                best_flow = flow_score
                best_maneuver = m
                best_v_target = v_target
                
        # ── Update state ──────────────────────────────────────────────────────
        if best_maneuver != Maneuver.KEEP_SPEED:
            if best_maneuver != self.committed_maneuver:
                self.committed_maneuver = best_maneuver
                self.commit_steps_left = 2
            # Always update persistent direction memory when lane-changing
            if best_maneuver in [Maneuver.CHANGE_LEFT, Maneuver.CHANGE_RIGHT]:
                self.last_lc_dir = best_maneuver
                self.steps_since_lc = 0
        else:
            if self.commit_steps_left == 0:
                self.committed_maneuver = Maneuver.KEEP_SPEED
            # last_lc_dir persists (steps_since_lc already incremented at top)
                
        return best_maneuver, best_v_target


