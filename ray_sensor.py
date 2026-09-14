import numpy as np

def ray_intersect_segment(ray_origin, ray_dir, p1, p2):
    v1 = ray_origin - p1
    v2 = p2 - p1
    v3 = np.array([-ray_dir[1], ray_dir[0]])
    dot = np.dot(v2, v3)
    if abs(dot) < 1e-6:
        return float('inf')
    t1 = (v2[0] * v1[1] - v2[1] * v1[0]) / dot
    t2 = np.dot(v1, v3) / dot
    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return t1
    return float('inf')

def get_vehicle_corners(x, y, length, width, heading):
    c, s = np.cos(heading), np.sin(heading)
    dx = length / 2
    dy = width / 2
    corners = np.array([
        [dx, -dy],
        [dx, dy],
        [-dx, dy],
        [-dx, -dy]
    ])
    rot = np.array([[c, -s], [s, c]])
    return np.array([x, y]) + np.dot(corners, rot.T)

class RaySensor:
    def __init__(self, num_rays=32, max_dist=50.0):
        self.num_rays = num_rays
        self.max_dist = max_dist
        
    def observe(self, env):
        ego = env.unwrapped.vehicle
        # Phase 4: vy (횡방향 속도)도 함께 추출 — IMU/휠속도 센서에 해당하는 내부 관측
        ego_state = {
            'x': ego.position[0],
            'y': ego.position[1],
            'heading': ego.heading,
            'vx': ego.velocity[0],
            'vy': ego.velocity[1],   # 횡방향 속도 (2차 동역학 식별에 필요)
        }
        
        other_vehicles = []
        for v in env.unwrapped.road.vehicles:
            if v is not ego:
                dist = np.hypot(v.position[0] - ego.position[0], v.position[1] - ego.position[1])
                if dist < self.max_dist + 10.0:
                    other_vehicles.append({
                        'x': v.position[0], 'y': v.position[1],
                        'length': v.LENGTH, 'width': v.WIDTH, 'heading': v.heading
                    })
        
        # Road edges removed
        angles = np.linspace(0, 2*np.pi, self.num_rays, endpoint=False) + ego_state['heading']
        ray_dirs = np.column_stack((np.cos(angles), np.sin(angles)))
        
        distances = np.full(self.num_rays, self.max_dist)
        origin = np.array([ego_state['x'], ego_state['y']])
                        
        for ov in other_vehicles:
            corners = get_vehicle_corners(ov['x'], ov['y'], ov['length'], ov['width'], ov['heading'])
            segments = [
                (corners[0], corners[1]),
                (corners[1], corners[2]),
                (corners[2], corners[3]),
                (corners[3], corners[0])
            ]
            for i in range(self.num_rays):
                rd = ray_dirs[i]
                for p1, p2 in segments:
                    t = ray_intersect_segment(origin, rd, p1, p2)
                    if t < distances[i]:
                        distances[i] = t
                        
        return distances, angles, ego_state
