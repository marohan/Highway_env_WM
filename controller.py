import numpy as np

class Maneuver:
    FOLLOW = 0
    KEEP_SPEED = 1
    CHANGE_LEFT = 2
    CHANGE_RIGHT = 3
    DECEL_TO_STOP = 4
    YIELD = 5

class EgoController:
    """
    Translates high-level Maneuver units into ContinuousAction [acceleration, steering].
    Implements a simple PID for speed and a PD for lane tracking.
    """
    def __init__(self, dt=1.0):
        self.dt = dt
        self.target_speed = 30.0
        self.target_lane_y = None
        self.LANE_WIDTH = 4.0
        
        # Controller gains (will be identified online in Phase 5, but here are initial priors)
        self.kp_steer = 0.1
        self.kd_steer = 0.2
        self.kp_accel = 1.0
        
        self.current_maneuver = Maneuver.KEEP_SPEED
        self.prev_accel = None
        
    def set_maneuver(self, maneuver, ego_y, ego_vx, target_speed=30.0):
        # Current discrete lane (0, 1, 2, 3 for y = 0, 4, 8, 12)
        current_lane_idx = int(round(np.clip(ego_y / self.LANE_WIDTH, 0, 3)))
        
        # Hysteresis: if already changing lane, do not interrupt unless completed
        if self.current_maneuver in [Maneuver.CHANGE_LEFT, Maneuver.CHANGE_RIGHT]:
            # Check if maneuver is complete (within 0.3m of target lane center)
            if self.target_lane_y is not None and abs(ego_y - self.target_lane_y) < 0.3:
                self.current_maneuver = Maneuver.KEEP_SPEED
            else:
                # Keep executing current lane change (can update target_speed)
                self.target_speed = target_speed
                return
                
        self.current_maneuver = maneuver
        self.target_speed = target_speed
        
        if maneuver == Maneuver.CHANGE_LEFT:
            target_idx = max(0, current_lane_idx - 1)
            self.target_lane_y = target_idx * self.LANE_WIDTH
        elif maneuver == Maneuver.CHANGE_RIGHT:
            target_idx = min(3, current_lane_idx + 1)
            self.target_lane_y = target_idx * self.LANE_WIDTH
        elif maneuver == Maneuver.DECEL_TO_STOP:
            self.target_speed = 0.0
        else: # KEEP_SPEED, FOLLOW, YIELD
            self.target_lane_y = current_lane_idx * self.LANE_WIDTH
            
    def compute_action(self, ego_state):
        ego_y  = ego_state['y']
        ego_vy = ego_state['vy']
        ego_vx = ego_state['vx']
        ego_heading = ego_state.get('heading', 0.0)

        # ── Longitudinal control ──────────────────────────────────────────
        target_accel = self.kp_accel * (self.target_speed - ego_vx)
        target_accel = np.clip(target_accel, -5.0, 3.0)

        # Prevent reverse gear
        if target_accel < 0.0 and ego_vx <= 0.1:
            target_accel = 0.0

        # Slew-rate limiter (bypass during AEB)
        if self.prev_accel is not None and self.current_maneuver != Maneuver.DECEL_TO_STOP:
            max_d_accel = 5.0 * self.dt   # m/s² per step
            if target_accel < self.prev_accel - max_d_accel:
                target_accel = self.prev_accel - max_d_accel

        self.prev_accel = target_accel

        # ── Lateral control (Cascaded Position -> Heading -> Steer) ───────
        if self.target_lane_y is None:
            current_lane_idx = int(round(np.clip(ego_y / self.LANE_WIDTH, 0, 3)))
            self.target_lane_y = current_lane_idx * self.LANE_WIDTH

        y_error = self.target_lane_y - ego_y

        # Cascaded control:
        # Outer loop: lateral position error -> desired heading angle
        # Max heading angle reference = 0.08 rad (~4.6 degrees) for gentle, stable lane tracking
        heading_ref = np.clip(0.035 * y_error, -0.08, 0.08)

        # Inner loop: heading error -> steering angle
        # Damper on heading with 1Hz discrete stability
        steer = np.clip(0.3 * (heading_ref - ego_heading), -0.025, 0.025)

        # ── Normalise to [-1, 1] for ContinuousAction ────────────────────
        norm_accel = np.clip(target_accel / 5.0, -1.0, 1.0)
        norm_steer = steer / (np.pi / 4)

        return np.array([norm_accel, norm_steer], dtype=np.float32)

