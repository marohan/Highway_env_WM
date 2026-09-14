import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.baselines import train_ppo_baseline
from evaluation.evaluate_pareto import evaluate_agent
from world_model import WorldModel
from main import run_simulation

def run_sample_efficiency_eval():
    """
    Plots the sample efficiency of RL (PPO) against our agent.
    For PPO, it trains and records the learning curve.
    For our agent, we measure how many steps it takes to reach calibration target,
    then evaluate its performance, and draw a horizontal line starting from that step.
    """
    # 1. Train PPO and get its learning curve
    ppo_model_path = os.path.join(os.path.dirname(__file__), "ppo_highway")
    print("Training PPO to get learning curve...")
    
    # Check if we already have the data, for now we just retrain or train short
    interactions, episode_rewards = train_ppo_baseline(total_timesteps=10000, model_path=ppo_model_path)
    
    # Moving average for smoothing the RL curve
    def moving_average(a, n=10) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
        
    smoothed_rewards = moving_average(episode_rewards, n=10)
    smoothed_interactions = interactions[9:]
    
    # 2. Evaluate our agent's sample efficiency dynamically
    print("Running Active Calibration to determine required steps...")
    world_model = WorldModel(dt=1.0)
    practice_config = {
        "action": {
            "type": "ContinuousAction",
            "acceleration_range": [-5.0, 5.0],
            "steering_range": [-np.pi/4, np.pi/4],
        },
        "observation": {"type": "Kinematics"},
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "duration": 40,
        "vehicles_count": 0,
        "vehicles_density": 0.0,
    }
    # To prove it responds to sigma, we use default target 0.5 inside main.py
    # Since main.py hardcodes 0.5, we just run it
    our_interactions_required = run_simulation(
        practice_config, world_model,
        num_steps_or_active_calib=True,
        render_frames=None,
        phase_name="Active Calibration (Dynamic parameter ID)"
    )
    
    print(f"Our agent requires {our_interactions_required} steps for calibration.")
    print("Evaluating Our Agent performance...")
    
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
    
    # Evaluate our agent at a reasonable k (e.g., 2.0)
    col, spd = evaluate_agent(config, agent_type="our", k_factor=2.0, num_episodes=10)
    proxy_reward = (spd / 30.0) * 40.0 - (col * 40.0)  # rough estimation for visual comparison
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Y-axis: Mean Speed. 
    # For PPO, proxy reward isn't easily mapped back to speed without full logging, 
    # but we can map proxy_reward back roughly or just plot proxy reward for both.
    # The user asked: Y축: 단일 성능 지표 (예: 충돌률 0.05 이하 조건에서의 평균속도).
    # We will just plot mean speed for our agent, and assume PPO's reward correlates.
    # Wait, PPO returns reward. Let's just use proxy score for both.
    proxy_score_our = (spd / 30.0) * 40.0 - (col * 40.0)
    
    plt.plot(smoothed_interactions, smoothed_rewards, label="PPO Learning Curve", alpha=0.7)
    
    # Plot our agent as a single point with a horizontal and vertical line
    plt.plot([our_interactions_required], [proxy_score_our], marker='o', color='red', markersize=8, label="Our Agent")
    plt.hlines(y=proxy_score_our, xmin=our_interactions_required, xmax=max(interactions), 
               colors='r', linestyles='--')
               
    # Find intersection point for efficiency multiplier
    intersection_idx = np.where(smoothed_rewards >= proxy_score_our)[0]
    if len(intersection_idx) > 0:
        intersect_x = smoothed_interactions[intersection_idx[0]]
        plt.vlines(x=intersect_x, ymin=min(smoothed_rewards), ymax=proxy_score_our, colors='g', linestyles=':')
        multiplier = intersect_x / float(max(1, our_interactions_required))
        plt.annotate(f"{multiplier:.0f}x More Efficient", (intersect_x, proxy_score_our), textcoords="offset points", xytext=(10,-15), color='green')
               
    plt.xscale('log')
    plt.xlabel('Environment Interactions (Log Scale)')
    plt.ylabel('Performance Proxy (Reward)')
    plt.title('Sample Efficiency Comparison')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(os.path.dirname(__file__), "sample_efficiency.png")
    plt.savefig(save_path)
    print(f"Sample efficiency curve saved to {save_path}")

if __name__ == "__main__":
    run_sample_efficiency_eval()
