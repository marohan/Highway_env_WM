import numpy as np

class Scalar:
    """스칼라 칼만 필터 (Recursive Least Squares)"""
    def __init__(self, mu, sigma, obs_var=0.5, floor=1e-3):
        self.mu = float(mu)
        self.var = float(sigma) ** 2
        self.obs_var = obs_var
        self.floor = floor

    def observe(self, z):
        K = self.var / (self.var + self.obs_var)
        self.mu += K * (z - self.mu)
        self.var = max(self.var * (1.0 - K), self.floor)

    @property
    def sigma(self):
        return np.sqrt(self.var)


class LateralDynamics:
    """
    Phase 4: 차선 변경 2차 동역학 (스프링-댐퍼) 모델 식별
        a_y = Kp * (y_target - y) - Kd * vy
    """
    LANE_WIDTH = 4.0

    def __init__(self, dt=1.0):
        self.dt = dt
        self.Kp = Scalar(1.0, 5.0, obs_var=1.0, floor=0.05)
        self.Kd = Scalar(1.0, 5.0, obs_var=0.4, floor=0.05)  # obs_var 낮춤: 더 적극 학습
        self._in_lc = False
        self._y_target = None
        self._prev_vy = 0.0

    def reset_episode(self):
        """새 에피소드 시작 전 호출: 에피소드 간 상태 누수 방지"""
        self._in_lc = False
        self._y_target = None
        self._prev_vy = 0.0
        self._prev_y = 0.0

    def start_lane_change(self, y_now, direction):
        if direction == 0:
            self._y_target = y_now - self.LANE_WIDTH
        else:
            self._y_target = y_now + self.LANE_WIDTH
        self._in_lc = True
        self._prev_vy = 0.0
        self._prev_y = y_now

    def observe(self, y_now, vy_now):
        if not self._in_lc or self._y_target is None:
            return
        a_y_obs = (vy_now - self._prev_vy) / self.dt
        dy = self._y_target - self._prev_y
        if abs(dy) > 0.1:
            residual_kp = a_y_obs + self.Kd.mu * self._prev_vy
            self.Kp.observe(residual_kp / dy)
        if abs(self._prev_vy) > 0.05:
            residual_kd = self.Kp.mu * dy - a_y_obs
            self.Kd.observe(residual_kd / self._prev_vy)
        self._prev_vy = vy_now
        self._prev_y = y_now
        if abs(dy) < 0.1 and abs(vy_now) < 0.1:
            self._in_lc = False
            self._y_target = None

    def simulate(self, y0, vy0, direction, steps, world_kp=None, world_kd=None):
        kp = world_kp if world_kp is not None else self.Kp.mu
        kd = world_kd if world_kd is not None else self.Kd.mu
        if direction == 0:
            y_target = y0 - self.LANE_WIDTH
        elif direction == 2:
            y_target = y0 + self.LANE_WIDTH
        else:
            y_target = y0
        trajectory_y = []
        y, vy = y0, vy0
        for _ in range(steps):
            a_y = kp * (y_target - y) - kd * vy
            vy += a_y * self.dt
            y += vy * self.dt
            trajectory_y.append(y)
        return trajectory_y


class RelativeObstacleModel:
    """
    Phase 6: 장애물 2-class 상대 속도 모델

    동일 차선(forward) 장애물과 인접 차선(adjacent) 장애물을 구분하여
    각기 다른 속도 모델을 식별합니다.

    - v_rel_fwd: 전방 동일 차선 장애물과의 상대 접근 속도 (양수 = 빠르게 다가옴)
                 전방 Ray의 연속 변화량으로 온라인 추정합니다.
    - v_adj_surplus: 인접 차선 차량이 에고보다 빠를 가능성 (추월 위험).
                     현재는 Prior에 의존하며 넓은 sigma로 앙상블 비관주의를 유발합니다.
    """
    FORWARD_HALF_ANGLE = np.radians(25)  # ±25° 이내 = 전방 동일 차선 영역

    def __init__(self, dt=1.0):
        self.dt = dt
        # 전방 동일 차선 상대 접근 속도: 양수 = 에고가 더 빠름 (앞차를 따라잡는 중)
        # Prior: 5 m/s (에고가 약간 더 빠른 상황이 일반적)
        self.v_rel_fwd = Scalar(5.0, 8.0, obs_var=2.0, floor=0.3)
        # 인접 차선 추월 위험: 상대방이 에고보다 빠를 수 있는 속도 여유
        # Prior: 0 ± 3 m/s (추월하는 차가 있을 수 있음, 하지만 확신 없음)
        self.v_adj_surplus = Scalar(0.0, 3.0, obs_var=1.5, floor=0.3)
        self._prev_fwd_min = None

    def observe(self, distances, angles, ego_heading, ego_vy=0.0):
        """
        전방 Ray 최솟값의 연속 변화로 v_rel_fwd를 온라인 추정합니다.
        [Version fix] 횡방향 운동(y=-2, y=14)이 전방 Ray에 진입하는
        차선 변경 중에는 업데이트를 건너뜁니다. (오염 방지)
        """
        if abs(ego_vy) > 0.1:
            # 횡방향 운동 중에는 전방 Ray가 도로 경계를 향해 오염 샘플이 생김 → 스킵
            # prev를 None으로 지워서 다음 직진 스텝에서 지속되지 않도록 합니다
            self._prev_fwd_min = None
            return
        rel_angles = angles - ego_heading
        rel_angles = (rel_angles + np.pi) % (2 * np.pi) - np.pi  # [-pi, pi] 정규화
        fwd_mask = np.abs(rel_angles) < self.FORWARD_HALF_ANGLE
        if not fwd_mask.any():
            self._prev_fwd_min = None
            return
        fwd_min = float(distances[fwd_mask].min())
        if fwd_min < 149.0:
            if self._prev_fwd_min is not None:
                dv = (self._prev_fwd_min - fwd_min) / self.dt
                if abs(dv) < 15.0:
                    self.v_rel_fwd.observe(dv)
            self._prev_fwd_min = fwd_min
        else:
            self._prev_fwd_min = None  # 전방에 장애물 없음 → 이전 관측 무효

    def reset_episode(self):
        self._prev_fwd_min = None


class SelfModel:
    """에이전트 자신의 동역학(Dynamics)과 주변 장애물 속도를 예측 오차로 식별하는 모델"""
    def __init__(self, dt=1.0):
        self.dt = dt
        # 종방향 동역학
        self.a_accel = Scalar(10.0, 10.0, obs_var=1.0, floor=0.1)
        self.a_decel = Scalar(-10.0, 10.0, obs_var=0.5, floor=0.1)  # 낮은 obs_var: 적극 학습
        # Phase 4: 횡방향 동역학 (2차 스프링-댐퍼)
        self.lateral = LateralDynamics(dt=dt)
        # Phase 6: 장애물 2-class 속도 모델
        self.obs_model = RelativeObstacleModel(dt=dt)

    def notify_action(self, action, state_now):
        """행동 직전 호출: 차선 변경 시작 여부를 LateralDynamics에 알림"""
        if action == 0 or action == 2:
            self.lateral.start_lane_change(state_now['y'], action)

    def reset_episode(self):
        """새 에피소드(새 env) 시작 전 호출 — 차선변경 상태 초기화"""
        self.lateral.reset_episode()
        self.obs_model.reset_episode()

    def observe(self, action, state_prev, state_now, distances, angles):
        dvx = state_now['vx'] - state_prev['vx']
        if action == 3:
            if dvx > 0:  # IDM 개입으로 인한 감속은 가속 능력 식별에서 제외
                self.a_accel.observe(dvx / self.dt)
        elif action == 4:
            if dvx < 0:
                self.a_decel.observe(dvx / self.dt)
        # 횡방향 동역학 관측
        self.lateral.observe(state_now['y'], state_now['vy'])
        # Phase 6: 장애물 상대 속도 관측 (횡방향 속도를 전달해 오염 방지)
        self.obs_model.observe(distances, angles, state_now['heading'], ego_vy=state_now['vy'])

    def get_ensemble(self, ego_vx, k=2.0):
        """
        Phase 6: 장애물 2-class 속도 모델이 포함된 앙상블 생성
        ego_vx: 현재 에고 종방향 속도 (장애물 절대 속도 계산 기준)
        """
        obs = self.obs_model
        # 동일 차선 장애물 절대 속도: 에고보다 v_rel_fwd 만큼 느림
        v_fwd = ego_vx - obs.v_rel_fwd.mu
        # 인접 차선 장애물 기본 속도: 에고와 동일 (중립 가정)
        v_adj = ego_vx

        base = {
            'a_accel': self.a_accel.mu,
            'a_decel': self.a_decel.mu,
            'kp': self.lateral.Kp.mu,
            'kd': self.lateral.Kd.mu,
            'v_fwd_obs': v_fwd,   # 동일 차선 장애물 절대 속도
            'v_adj_obs': v_adj,   # 인접 차선 장애물 절대 속도
        }

        converged = (
            self.a_accel.sigma < 0.5 and
            self.a_decel.sigma < 0.5 and
            self.lateral.Kp.sigma < 0.5 and
            self.lateral.Kd.sigma < 0.6 and
            obs.v_rel_fwd.sigma < 2.0
        )
        if converged:
            return [base]

        def var(**kw):
            d = dict(base)
            d.update(kw)
            return d

        k_kd = min(k, 1.5)
        k_adj = 1.0  # 인접 차선 추월 앙상블은 1-sigma로 완화 (너무 보수적이면 차선 변경 불가)

        return [
            base,
            # W1: 제동력 부족 → 앞차 추돌 위험
            var(a_decel=min(0.0, self.a_decel.mu + k * self.a_decel.sigma)),
            # W2: 가속력 부족
            var(a_accel=self.a_accel.mu - k * self.a_accel.sigma),
            # W3: 복원력(Kp) 약함 → 차선 변경 느림
            var(kp=max(0.1, self.lateral.Kp.mu - k * self.lateral.Kp.sigma)),
            # W4: 저항(Kd) 약함 → 오버슈트 (1.5-sigma로 완화)
            var(kd=max(0.2, self.lateral.Kd.mu - k_kd * self.lateral.Kd.sigma)),
            # W5: 앞차가 예상보다 빠르게 다가옴 (전방 차량 실제론 더 느림)
            var(v_fwd_obs=ego_vx - (obs.v_rel_fwd.mu + k * obs.v_rel_fwd.sigma)),
            # W6: 인접 차선에 추월 차량 (에고보다 빠름) → 차선 변경 시 후방 추돌
            var(v_adj_obs=ego_vx + obs.v_adj_surplus.mu + k_adj * obs.v_adj_surplus.sigma),
            # W7: 인접 차선 차량이 에고보다 느림 → 차선 변경 시 전방 추돌
            var(v_adj_obs=max(0.0, ego_vx - (obs.v_adj_surplus.mu + k_adj * obs.v_adj_surplus.sigma))),
        ]
