"""
experiments/run_ablation.py

Quantitative Ablation Study for CANDY Diffusion.
Bypasses the Web UI to compute exact convergence metrics over time.
"""

import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from manifold_flow.systems.neural import CANDYDiffusionSystem
from manifold_flow.solvers.rk4_solver import RK4Solver

def compute_convergence_error(system, state):
    """Mean Euclidean distance from each particle to its ground-truth target."""
    Y = state.reshape(system.num_particles, system.base_dim)
    distances = np.linalg.norm(Y - system.targets, axis=1)
    return np.mean(distances)

def run_single_experiment(params, t_span=(0.0, 5.0), dt=0.01):
    print(f"[*] Running experiment with params: {params}")
    
    system = CANDYDiffusionSystem(num_particles=500)
    system.update_parameters(params)

    np.random.seed(42)
    y0 = system.get_initial_conditions()

    solver = RK4Solver()
    result = solver.solve(system, y0, t_span, dt)

    times = result.times
    errors = []

    sample_indices = range(0, len(times), 10)
    sampled_times = [times[i] for i in sample_indices]
    
    for i in sample_indices:
        state = result.states[i]
        err = compute_convergence_error(system, state)
        errors.append(err)
        
    return sampled_times, errors

def main():
    experiments = {
        "Baseline (No CANDY)": {"decay": 1.0, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 0.0, "T": 5.0},
        "CANDY (scale=0.5)":   {"decay": 1.0, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 0.5, "T": 5.0},
        "CANDY (scale=1.0)":   {"decay": 1.0, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 1.0, "T": 5.0},
        "CANDY (scale=2.0)":   {"decay": 1.0, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 2.0, "T": 5.0},
    }
    
    results = {}
    
    for name, params in experiments.items():
        t, err = run_single_experiment(params)
        results[name] = {"time": t, "error": err}
        
    with open("candy_ablation_results.json", "w") as f:
        json.dump(results, f)
    print("\n[+] Data saved to candy_ablation_results.json")

    plt.figure(figsize=(10, 6))
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    for idx, (name, data) in enumerate(results.items()):
        plt.plot(data["time"], data["error"], label=name, linewidth=2.5, color=colors[idx])
        
    plt.title("Convergence Analysis of CANDY Diffusion", fontsize=16, fontweight='bold')
    plt.xlabel("Simulation Time (t)", fontsize=14)
    plt.ylabel("Mean Distance to Target (Error)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig("candy_convergence.png", dpi=300)
    print("[+] Plot saved to candy_convergence.png")

if __name__ == "__main__":
    main()