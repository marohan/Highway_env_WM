import numpy as np
import itertools

class LexicographicPlanner:
    V_MAX = 30.0  # highway-v0 최고속도 (m/s). 플래너 시뮬레이터는 이 값을 알고 있음.

    def __init__(self, horizon=4, dt=1.0, ego_length=5.0, ego_width=2.0):
        self.H = horizon
        self.dt = dt
        self.actions = [0, 1, 2, 3, 4]  # LANE_L, IDLE, LANE_R, FASTER, SLOWER
        self.ego_length = ego_length
        self.ego_width = ego_width

    def generate_sequences(self):
        return list(itertools.product(self.actions, repeat=self.H))

    def _simulate_trajectory(self, seq, base_obstacles, obs_dy_initial,
                              ego_state, world, lateral_model):
        """
        Phase 6: 장애물를 초기 횡방향 오프셋으로 분류하여 각기 다른 속도 모델을 적용합니다.
        - 동일 차선 (|obs_dy_initial| < 2.5m): world['v_fwd_obs'] 사용
        - 인접 차선 (2.5 <= |obs_dy_initial| < 7m): world['v_adj_obs'] 사용
        """
        is_same_lane = np.abs(obs_dy_initial) < 2.5

        ego_x = 0.0
        ego_y = ego_state['y']
        ego_vx = ego_state['vx']
        ego_vy = ego_state['vy']

        y_target = ego_y
        in_lc = lateral_model._in_lc
        if in_lc and lateral_model._y_target is not None:
            y_target = lateral_model._y_target

        for t in range(self.H):
            act = seq[t]

            # 종방향 동역학 (최고속도 제한 포함)
            if act == 3:
                ego_vx = ego_vx + world['a_accel'] * self.dt
            elif act == 4:
                ego_vx = ego_vx + world['a_decel'] * self.dt
            ego_vx = max(0.0, min(ego_vx, self.V_MAX))  # 속도 제한 적용

            # 차선 변경 명령
            if act == 0:
                y_target = ego_y - 4.0
                in_lc = True
            elif act == 2:
                y_target = ego_y + 4.0
                in_lc = True

            # 횡방향 동역학 (스프링-댐퍼)
            if in_lc:
                a_y = world['kp'] * (y_target - ego_y) - world['kd'] * ego_vy
                ego_vy = ego_vy + a_y * self.dt
                if abs(y_target - ego_y) < 0.1 and abs(ego_vy) < 0.1:
                    in_lc = False
                    ego_vy = 0.0
            else:
                ego_vy = 0.0

            ego_x += ego_vx * self.dt
            ego_y += ego_vy * self.dt

            # Phase 6: 장애물 class별 미래 위치 예측
            step_t = (t + 1) * self.dt
            obs_x_displace = np.where(
                is_same_lane,
                world['v_fwd_obs'] * step_t,    # 동일 차선: v_rel_fwd로 추정된 절대 속도
                world['v_adj_obs'] * step_t,    # 인접 차선: 추월 가능성 포함한 절대 속도
            )
            obs_dx = base_obstacles[:, 0] + obs_x_displace - ego_x
            obs_dy = base_obstacles[:, 1] - ego_y  # 절대 y 기준

            # 충돌 확인
            buf = 0.5
            collision = np.any(
                (np.abs(obs_dx) < (self.ego_length / 2 + buf)) &
                (np.abs(obs_dy) < (self.ego_width / 2 + buf))
            )
            if ego_y < -2.0 or ego_y > 14.0:
                collision = True

            if collision:
                return t / self.H

        # Terminal state safety check (beyond horizon)
        buf = 0.5
        same_lane_mask = np.abs(obs_dy) < (self.ego_width / 2 + buf)
        if np.any(same_lane_mask):
            dx_same_lane = obs_dx[same_lane_mask]
            fwd_mask = dx_same_lane > 0
            if np.any(fwd_mask):
                fwd_dx = dx_same_lane[fwd_mask]
                closest_idx = np.argmin(fwd_dx)
                closest_dx = fwd_dx[closest_idx] - self.ego_length / 2 - buf
                if closest_dx > 0:
                    a_decel = abs(world['a_decel'])
                    obs_vx = world['v_fwd_obs'] if is_same_lane[same_lane_mask][fwd_mask][closest_idx] else world['v_adj_obs']
                    # ego_vx must be able to brake to obs_vx within closest_dx
                    safe_rel_vx = np.sqrt(max(0, 2 * a_decel * closest_dx))
                    if ego_vx > obs_vx + safe_rel_vx:
                        return 0.99  # Almost 1.0, but still a failure beyond horizon
        return 1.0

    def plan(self, distances, angles, ego_state, ensemble, lateral_model):
        """
        distances, angles: RaySensor output
        ego_state: dict with x, y, vx, vy, heading
        ensemble: List of world-dicts (a_accel, a_decel, kp, kd, v_fwd_obs, v_adj_obs)
        lateral_model: LateralDynamics 인스턴스
        """
        valid = distances < 149.0
        if valid.sum() == 0:
            return 1  # IDLE

        lx = distances[valid] * np.cos(angles[valid] - ego_state['heading'])
        ly_abs = ego_state['y'] + distances[valid] * np.sin(angles[valid] - ego_state['heading'])
        base_obstacles = np.column_stack((lx, ly_abs))
        # Phase 6: 초기 횡방향 오프셋으로 장애물 분류 (차선 변경 시 변경되지 않는 고정값)
        obs_dy_initial = ly_abs - ego_state['y']

        best_action = 1  # 기본값: IDLE
        best_tuple = (-np.inf, -np.inf, -np.inf)
        num_worlds = len(ensemble)

        for seq in self.generate_sequences():
            world_survivals = []
            base_task_score = -np.inf

            for wi, world in enumerate(ensemble):
                survival = self._simulate_trajectory(
                    seq, base_obstacles, obs_dy_initial,
                    ego_state, world, lateral_model
                )
                world_survivals.append(survival)

                if wi == 0:
                    lane_changed = any(a in (0, 2) for a in seq)
                    lane_change_count = sum(1 for a in seq if a in (0, 2))
                    # 속도 제한을 반영한 H스텝 이동 거리
                    vx_sim = ego_state['vx']
                    dist_sim = 0.0
                    for a in seq:
                        if a == 3: vx_sim = min(vx_sim + world['a_accel'] * self.dt, self.V_MAX)
                        elif a == 4: vx_sim = max(vx_sim + world['a_decel'] * self.dt, 0.0)
                        dist_sim += vx_sim * self.dt
                    # 미래 속도 보너스: 차선 변경의 장기적 가치를 포착
                    # - 같은 차선 유지: 결국 전방 차량 속도(v_fwd_obs)에 맞춰야 함 → 손실 발생
                    # - 차선 변경 후: 새 차선에서 최고속도 유지 가능
                    FUTURE_H = 4
                    future_speed = self.V_MAX if lane_changed else min(world['v_fwd_obs'], self.V_MAX)
                    future_bonus = future_speed * FUTURE_H * self.dt
                    base_task_score = dist_sim + future_bonus - 1.0 * lane_change_count

            min_survival = min(world_survivals)
            mean_survival = sum(world_survivals) / num_worlds
            current_tuple = (min_survival, mean_survival, base_task_score)

            if current_tuple > best_tuple:
                best_tuple = current_tuple
                best_action = seq[0]

        return best_action
