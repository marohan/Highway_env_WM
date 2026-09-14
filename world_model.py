import numpy as np
from scipy.spatial.distance import cdist

class Scalar:
    def __init__(self, mu, sigma, obs_var=0.5, floor=1e-3):
        self.mu = float(mu)
        self.var = float(sigma) ** 2
        self.obs_var = obs_var
        self.floor = floor

    def observe(self, z):
        K = self.var / (self.var + self.obs_var)
        self.mu += K * (z - self.mu)
        self.var = max(self.var * (1.0 - K), self.floor)

    def inflate(self, amount):
        self.var += amount

    @property
    def sigma(self):
        return np.sqrt(self.var)


class Track:
    """EKF-based Multi-Object Tracker for a single vehicle with IMM Intent Estimation."""
    def __init__(self, id, x, y, dt=1.0, is_phantom=False):
        self.id = id
        self.dt = dt
        self.is_phantom = is_phantom
        # State: [x, y, vx, vy]. Prior vx is set to traffic flow speed (22.0 m/s)
        # instead of 0.0 (stopped wall) to avoid false emergency stops on new tracks.
        self.x = np.array([x, y, 22.0, 0.0])
        self.P = np.eye(4) * 10.0
        self.Q = np.diag([0.1, 0.1, 1.0, 1.0])
        self.R = np.diag([0.5, 0.5])
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.intent_probs = np.array([0.7, 0.1, 0.1, 0.1])
        
    def predict(self):
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.x
        
    def update(self, z):
        self.time_since_update = 0
        self.hits += 1
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        vy = self.x[3]
        if vy > 0.5:
            likelihood = np.array([0.1, 0.8, 0.05, 0.05])
        elif vy < -0.5:
            likelihood = np.array([0.1, 0.05, 0.8, 0.05])
        else:
            likelihood = np.array([0.8, 0.05, 0.05, 0.1])
        self.intent_probs = self.intent_probs * likelihood
        self.intent_probs /= np.sum(self.intent_probs)


class MOT:
    """Multi-Object Tracker using RaySensor outputs."""
    def __init__(self, dt=1.0):
        self.dt = dt
        self.tracks = []
        self.next_id = 0
        self.max_age = 5
        self.total_phantoms_generated = 0
        
    def _cluster_rays(self, distances, angles, ego_x, ego_y, ego_heading):
        valid = distances < 149.0
        if not np.any(valid):
            return [], []
        dists = distances[valid]
        angs = angles[valid]
        lx = ego_x + dists * np.cos(angs)
        ly = ego_y + dists * np.sin(angs)
        points = np.column_stack((lx, ly))
        clusters = []
        if len(points) > 0:
            clusters.append([points[0]])
            for p in points[1:]:
                added = False
                for c in clusters:
                    if np.min(np.linalg.norm(np.array(c) - p, axis=1)) < 3.0:
                        c.append(p)
                        added = True
                        break
                if not added:
                    clusters.append([p])
        centroids = [np.mean(c, axis=0) for c in clusters]
        return centroids, clusters
        
    def _add_occlusion_phantoms(self, distances, angles, ego_x, ego_y, centroids):
        phantoms = []
        for i in range(1, len(distances)):
            d1, d2 = distances[i-1], distances[i]
            if abs(d1 - d2) > 5.0 and min(d1, d2) < 149.0:
                if d1 < d2:
                    px = ego_x + d1 * np.cos(angles[i-1])
                    py = ego_y + d1 * np.sin(angles[i-1])
                else:
                    px = ego_x + d2 * np.cos(angles[i])
                    py = ego_y + d2 * np.sin(angles[i])
                if not (-2.0 <= py <= 14.0):
                    continue
                if abs(py - ego_y) > 6.0:
                    continue
                is_new = True
                for c in centroids:
                    if np.linalg.norm(c - [px, py]) < 4.0:
                        is_new = False
                        break
                if is_new:
                    phantoms.append(np.array([px, py]))
                    self.total_phantoms_generated += 1
        return phantoms

    def update(self, distances, angles, ego_state):
        centroids, clusters = self._cluster_rays(distances, angles, ego_state['x'], ego_state['y'], ego_state['heading'])
        phantoms = self._add_occlusion_phantoms(distances, angles, ego_state['x'], ego_state['y'], centroids)
        all_measurements = centroids + phantoms
        for trk in self.tracks:
            trk.predict()
        if not all_measurements:
            self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
            return
        if not self.tracks:
            for i, z in enumerate(all_measurements):
                is_p = i >= len(centroids)
                self.tracks.append(Track(self.next_id, z[0], z[1], self.dt, is_phantom=is_p))
                self.next_id += 1
            return
        track_preds = np.array([[t.x[0], t.x[1]] for t in self.tracks])
        meas = np.array(all_measurements)
        
        # Anisotropic highway distance: enforce lane corridor (dy <= 2.5m) and allow longitudinal speed variation
        dx = track_preds[:, [0]] - meas[:, 0]
        dy = track_preds[:, [1]] - meas[:, 1]
        lane_penalty = np.where(np.abs(dy) > 2.5, 1000.0, 0.0)
        dist_matrix = np.sqrt(dx**2 + (2.5 * dy)**2) + lane_penalty

        matched_tracks = set()
        matched_meas = set()
        while True:
            if dist_matrix.size == 0 or np.min(dist_matrix) > 12.0:
                break
            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            t_idx, m_idx = min_idx
            self.tracks[t_idx].update(meas[m_idx])
            self.tracks[t_idx].is_phantom = (m_idx >= len(centroids))
            matched_tracks.add(t_idx)
            matched_meas.add(m_idx)
            dist_matrix[t_idx, :] = np.inf
            dist_matrix[:, m_idx] = np.inf
        for i, z in enumerate(meas):
            if i not in matched_meas:
                is_p = i >= len(centroids)
                self.tracks.append(Track(self.next_id, z[0], z[1], self.dt, is_phantom=is_p))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]


class WorldModel:
    """Unified World Model replacing SelfModel and incorporating MOT."""
    def __init__(self, dt=1.0):
        self.dt = dt
        self.mot = MOT(dt)
        self.a_accel = Scalar(10.0, 10.0, obs_var=1.0, floor=0.1)
        self.a_decel = Scalar(-10.0, 10.0, obs_var=0.5, floor=0.1)
        self.kp = Scalar(1.0, 5.0, obs_var=1.0, floor=0.05)
        self.kd = Scalar(1.0, 5.0, obs_var=0.4, floor=0.05)
        self.pred_error_history = []
        
    def reset_episode(self):
        self.mot.tracks = []
        self.mot.next_id = 0
        
    def observe(self, action, state_prev, state_now, distances, angles):
        # 1. Update MOT
        self.mot.update(distances, angles, state_now)
        
        dvx = state_now['vx'] - state_prev['vx']
        
        # Surprise Loop: physically scale predicted velocity change by commanded effort
        pred_vx = state_prev['vx']
        if action[0] > 0.1:
            pred_vx += action[0] * self.a_accel.mu * self.dt
        elif action[0] < -0.1:
            # action[0] is in [-1, 0], abs(a_decel.mu) is positive (~5.0)
            pred_vx += action[0] * abs(self.a_decel.mu) * self.dt
        
        error = abs(state_now['vx'] - pred_vx)
        self.pred_error_history.append(error)
        if error > 2.5:  # Genuine Surprise (e.g. friction loss or power loss)
            if action[0] > 0.1:
                self.a_accel.inflate(1.0)
            elif action[0] < -0.1:
                self.a_decel.inflate(1.0)

        # 2. Update Self Dynamics
        # action[0] is normalized in [-1,1] -> actual range [-5, 3] m/s^2
        # a_accel: update on positive thrust events
        # a_decel: ONLY update on FULL/NEAR-FULL emergency braking (action[0] <= -0.85, i.e. > 4.25 m/s^2 commanded)
        # This prevents gentle following deceleration from corrupting the physical emergency braking estimate.
        # SELF philosophy: a_decel tracks true physical friction/braking limit, not average gentle brake
        if action[0] > 0.1 and dvx > 0:
            self.a_accel.observe(dvx / self.dt)
        elif action[0] <= -0.85 and dvx < 0:   # full emergency braking events only
            self.a_decel.observe(dvx / self.dt)