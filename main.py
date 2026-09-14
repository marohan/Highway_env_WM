import gymnasium as gym
import highway_env
import numpy as np
import imageio
import os

from ray_sensor import RaySensor
from world_model import WorldModel
from planner import HierarchicalPlanner
from controller import EgoController, Maneuver

def run_simulation(env_config, world_model, num_steps_or_active_calib=False, render_frames=None, phase_name="", sigma_target=0.5):
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(env_config)
    obs, info = env.reset(seed=42)

    world_model.reset_episode()
    sensor = RaySensor(num_rays=512, max_dist=150.0)
    planner = HierarchicalPlanner(dt=1.0, k_factor=2.0)
    planner.v_limit = 26.0
    controller = EgoController(dt=1.0)

    if not num_steps_or_active_calib:
        ego = env.unwrapped.vehicle
        ego_x = ego.position[0]
        lane_offsets = {
            0: [48.0, 82.0, 115.0],
            1: [35.0, 70.0, 105.0],
            2: [26.0, 60.0,  95.0],
            3: [42.0, 75.0, 110.0],
        }
        other_vehs = [v for v in env.unwrapped.road.vehicles if v is not ego]
        idx = 0
        for lane_id in range(4):
            for offset in lane_offsets[lane_id]:
                if idx < len(other_vehs):
                    v = other_vehs[idx]
                    lane_idx = ('0', '1', lane_id)
                    lane = env.unwrapped.road.network.get_lane(lane_idx)
                    v.position = lane.position(ego_x + offset, 0)
                    v.lane_index = lane_idx
                    v.target_lane_index = lane_idx
                    v.lane = lane
                    v.heading = 0.0
                    v.speed = float(np.random.uniform(18.5, 21.0))
                    v.target_speed = v.speed
                    idx += 1

        for i in range(idx, len(other_vehs)):
            v = other_vehs[i]
            lane_idx = ('0', '1', 3)
            lane = env.unwrapped.road.network.get_lane(lane_idx)
            v.position = lane.position(ego_x + 140.0 + i * 25.0, 0)
            v.lane_index = lane_idx
            v.target_lane_index = lane_idx
            v.lane = lane
            v.speed = 20.0

    done = truncated = False
    _, _, ego_state_prev = sensor.observe(env)
    action_prev = None
    step = 0
    recycle_turn = 0

    print(f"\n{'='*65}")
    print(f"  Starting: {phase_name}")
    print(f"{'='*65}")

    while not (done or truncated):
        if not num_steps_or_active_calib and step >= env_config.get("duration", 40):
            break
            
        distances, angles, ego_state_now = sensor.observe(env)

        if action_prev is not None:
            world_model.observe(action_prev, ego_state_prev, ego_state_now, distances, angles)

        if num_steps_or_active_calib:
            if world_model.a_accel.sigma < sigma_target and world_model.a_decel.sigma < sigma_target:
                print(f"Active Calibration completed at step {step}!")
                break

        # Action selection
        if not num_steps_or_active_calib:
            # WARMUP FOR REAL DRIVING (2 steps for MOT initialization)
            if step < 2:
                maneuver = Maneuver.KEEP_SPEED
                target_speed = ego_state_now['vx']
            else:
                maneuver, target_speed = planner.plan(ego_state_now, world_model)
            controller.set_maneuver(maneuver, ego_state_now['y'], ego_state_now['vx'], target_speed=target_speed)
            action = controller.compute_action(ego_state_now)
        else:
            # Active Calibration: systematically calibrate braking, then acceleration
            if world_model.a_decel.sigma > sigma_target and ego_state_now['vx'] > 5.0:
                maneuver = Maneuver.DECEL_TO_STOP
                action = np.array([-1.0, 0.0], dtype=np.float32)
            elif world_model.a_accel.sigma > sigma_target:
                maneuver = Maneuver.KEEP_SPEED
                action = np.array([0.6, 0.0], dtype=np.float32)
            else:
                maneuver = Maneuver.KEEP_SPEED
                action = np.array([0.0, 0.0], dtype=np.float32)

        m_name = {0:'FOLLOW', 1:'KEEP_SPEED', 2:'CHANGE_LEFT', 3:'CHANGE_RIGHT', 4:'DECEL_TO_STOP', 5:'YIELD'}.get(maneuver, str(maneuver))
        curr_lane = int(round(ego_state_now['y'] / 4.0))
        print(f"Step {step:02d} | "
              f"x:{ego_state_now['x']:6.1f} | y:{ego_state_now['y']:4.1f} (L{curr_lane}) | "
              f"v:{ego_state_now['vx']:4.1f} | "
              f"Maneuver: {m_name}")

        obs, reward, done, truncated, info = env.step(action)

        # ── Continuous Traffic Stream (Active Traffic Circulation Bubble) ───
        if not num_steps_or_active_calib:
            curr_ego_x = ego_state_now['x']
            for v in env.unwrapped.road.vehicles:
                if v is not env.unwrapped.vehicle and v.position[0] < curr_ego_x - 25.0:
                    target_lane_id = curr_lane if (recycle_turn % 2 == 0) else ((curr_lane + (1 if recycle_turn % 4 == 1 else -1)) % 4)
                    recycle_turn += 1
                    cand_x = curr_ego_x + float(np.random.uniform(55.0, 75.0))
                    lane_idx = ('0', '1', target_lane_id)
                    lane = env.unwrapped.road.network.get_lane(lane_idx)
                    v.position = lane.position(cand_x, 0)
                    v.lane_index = lane_idx
                    v.target_lane_index = lane_idx
                    v.lane = lane
                    v.heading = 0.0
                    v.speed = float(np.random.uniform(18.5, 21.0))
                    v.target_speed = v.speed

        if render_frames is not None and not num_steps_or_active_calib:
            render_frames.append(env.render())

        ego_state_prev = ego_state_now
        action_prev = action
        step += 1

    print(f"{phase_name} ended. Done: {done}, Truncated: {truncated}")
    env.close()
    return step

def main():
    world_model = WorldModel(dt=1.0)
    frames = []

    practice_config = {
        "action": {"type": "ContinuousAction", "acceleration_range": [-5.0, 5.0], "steering_range": [-np.pi/4, np.pi/4]},
        "observation": {"type": "Kinematics"},
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "duration": 40,
        "vehicles_count": 0,
        "vehicles_density": 0.0,
    }
    
    run_simulation(practice_config, world_model, num_steps_or_active_calib=True, render_frames=frames,
                   phase_name="Active Calibration", sigma_target=0.5)

    real_config = {
        "action": {"type": "ContinuousAction", "acceleration_range": [-5.0, 5.0], "steering_range": [-np.pi/4, np.pi/4]},
        "observation": {"type": "Kinematics"},
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "duration": 80,
        "vehicles_count": 16,
        "vehicles_density": 1.0,
        "initial_lane_id": 1,
        "collision_reward": -1.0,
    }
    run_simulation(real_config, world_model, num_steps_or_active_calib=False, render_frames=frames, phase_name="Dense Traffic with Viability")

    vid_path = os.path.join(os.path.dirname(__file__), "Viability_Driving.mp4")
    imageio.mimsave(vid_path, frames, fps=4)
    print(f"\nVideo saved to {vid_path}")

if __name__ == "__main__":
    main()