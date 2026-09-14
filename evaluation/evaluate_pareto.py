import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats.qmc import Sobol

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model import WorldModel
from planner import HierarchicalPlanner
from controller import EgoController
from ray_sensor import RaySensor
from stable_baselines3 import PPO

def evaluate_agent(env_config, agent_type="our", k_factor=2.0, t_react=1.0, a_lead_max=5.0, num_episodes=20, ppo_model_path=None):
    env = gym.make("highway-v0")
    env.unwrapped.configure(env_config)
    
    total_collisions = 0
    total_speed = 0.0
    total_steps = 0
    
    if agent_type == "our":
        world_model = WorldModel(dt=1.0)
        world_model.a_accel.mu = 2.0; world_model.a_accel.var = 0.1**2
        world_model.a_decel.mu = -5.0; world_model.a_decel.var = 0.1**2
        world_model.kp.mu = 0.4; world_model.kp.var = 0.1**2
        world_model.kd.mu = 0.6; world_model.kd.var = 0.1**2
        
        sensor = RaySensor(num_rays=512, max_dist=150.0)
        planner = HierarchicalPlanner(dt=1.0, k_factor=k_factor, t_react=t_react, a_lead_max=a_lead_max)
        controller = EgoController(dt=1.0)
        controller.target_speed = 30.0
    elif agent_type == "ppo":
        if ppo_model_path and os.path.exists(ppo_model_path + ".zip"):
            model = PPO.load(ppo_model_path)
        else:
            return 0.0, 0.0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        
        if agent_type == "our":
            world_model.reset_episode()
        elif agent_type == "idm":
            from highway_env.vehicle.behavior import IDMVehicle
            ego = env.unwrapped.vehicle
            new_ego = IDMVehicle.create_from(ego)
            env.unwrapped.vehicle = new_ego
            env.unwrapped.road.vehicles.remove(ego)
            env.unwrapped.road.vehicles.append(new_ego)
        
        done = truncated = False
        step = 0
        episode_speed = 0.0
        
        while not (done or truncated):
            if agent_type == "our":
                distances, angles, ego_state_now = sensor.observe(env)
                world_model.observe(action, ego_state_prev, ego_state_now, distances, angles) if step > 0 else None
                maneuver, target_speed = planner.plan(ego_state_now, world_model)
                controller.set_maneuver(maneuver, ego_state_now['y'], ego_state_now['vx'], target_speed=target_speed)
                action = controller.compute_action(ego_state_now)
                ego_state_prev = ego_state_now
                speed = ego_state_now['vx']
            elif agent_type == "ppo":
                action, _ = model.predict(obs, deterministic=True)
                speed = env.unwrapped.vehicle.speed
            else:
                action = None
                speed = env.unwrapped.vehicle.speed

            obs, reward, done, truncated, info = env.step(action)
            episode_speed += speed
            step += 1
            
        if info.get("crashed", False):
            total_collisions += 1
            
        total_speed += (episode_speed / max(1, step))
        total_steps += 1
        
    env.close()
    
    return total_collisions / num_episodes, total_speed / num_episodes

def run_2stage_sweep():
    config = {
        "observation": {"type": "Kinematics"},
        "action": {
            "type": "ContinuousAction",
            "acceleration_range": [-5.0, 5.0],
            "steering_range": [-np.pi/4, np.pi/4],
        },
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "duration": 40,
        "vehicles_count": 20,
        "vehicles_density": 1.5,
    }

    # Evaluate IDM
    print("Evaluating IDM...", flush=True)
    idm_config = config.copy()
    idm_config["action"] = {"type": "DiscreteMetaAction"}
    idm_col, idm_spd = evaluate_agent(idm_config, agent_type="idm", num_episodes=20)
    print(f"IDM - Collision: {idm_col:.2f}, Speed: {idm_spd:.2f}", flush=True)

    # Stage 1: Coarse Sweep
    # Parameters: t_react in [0.5, 1.5], a_lead_max in [5.0, 9.0], k in [0.0, 4.0]
    num_samples = 16
    sampler = Sobol(d=3, scramble=True, seed=42)
    sample_points = sampler.random(n=num_samples)
    
    # Scale samples
    t_reacts = 0.5 + sample_points[:, 0] * 1.0
    a_lead_maxs = 5.0 + sample_points[:, 1] * 4.0
    ks = sample_points[:, 2] * 4.0

    print(f"\n--- Stage 1: Coarse Sweep (20 episodes each, {num_samples} samples) ---", flush=True)
    stage1_results = []
    
    for i in range(num_samples):
        tr = t_reacts[i]
        al = a_lead_maxs[i]
        k = ks[i]
        
        col, spd = evaluate_agent(config, agent_type="our", k_factor=k, t_react=tr, a_lead_max=al, num_episodes=20)
        stage1_results.append((tr, al, k, col, spd))
        print(f"Sample {i+1}/{num_samples}: t={tr:.2f}, a_lead={al:.2f}, k={k:.2f} -> Col={col:.2f}, Spd={spd:.2f}", flush=True)
        
    # Find frontier points
    candidates = []
    for (tr, al, k, col, spd) in stage1_results:
        if col <= max(idm_col, 0.1) and spd > 10.0:
            candidates.append((tr, al, k, col, spd))
            
    if not candidates:
        print("No good candidates found in Stage 1! Taking safest 3", flush=True)
        candidates = sorted(stage1_results, key=lambda x: x[3])[:3]
        
    # Sort candidates by a mix of safety and speed to pick a diverse subset for stage 2
    candidates = sorted(candidates, key=lambda x: x[4] - x[3]*50, reverse=True)[:3]
    
    print(f"\n--- Stage 2: Fine Evaluation ({len(candidates)} candidates, 50 episodes each) ---", flush=True)
    stage2_results = []
    for (tr, al, k, _, _) in candidates:
        col, spd = evaluate_agent(config, agent_type="our", k_factor=k, t_react=tr, a_lead_max=al, num_episodes=50)
        stage2_results.append((tr, al, k, col, spd))
        print(f"t={tr:.2f}, a_lead={al:.2f}, k={k:.2f} -> Col: {col:.2f}, Spd: {spd:.2f}", flush=True)
        
    # Plotting Stage 2 results
    plt.figure(figsize=(10, 7))
    cols = [r[3] for r in stage2_results]
    spds = [r[4] for r in stage2_results]
    
    plt.scatter(cols, spds, color='blue', label='Our Agent (Stage 2 Candidates)')
    for i, r in enumerate(stage2_results):
        plt.annotate(f"k={r[2]:.1f}", (r[3], r[4]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
    plt.scatter([idm_col], [idm_spd], color='green', s=150, label='IDM/MOBIL', marker='s')
    
    plt.xlabel('Collision Rate (100 episodes)')
    plt.ylabel('Mean Speed (m/s)')
    plt.title('Pareto Frontier Refinement')
    plt.grid(True)
    plt.legend()
    
    save_path = os.path.join(os.path.dirname(__file__), "pareto_curve.png")
    plt.savefig(save_path)
    print(f"\nSaved Stage 2 Pareto curve to {save_path}", flush=True)
    
    # Save best config
    best_config = min(stage2_results, key=lambda x: (x[3], -x[4]))
    print(f"\nBest Config found: t_react={best_config[0]:.2f}, a_lead_max={best_config[1]:.2f}, k={best_config[2]:.2f}", flush=True)
    print(f"Result: Col={best_config[3]:.2f}, Speed={best_config[4]:.2f}", flush=True)

if __name__ == "__main__":
    run_2stage_sweep()
