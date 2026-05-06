"""
experiments/run_all_ablations.py

Automated Quantitative Ablation Suite for CANDY Diffusion.
Generates publication-ready high-resolution plots for 5 experimental groups.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

from manifold_flow.systems.neural import CANDYDiffusionSystem
from manifold_flow.solvers.rk4_solver import RK4Solver

def compute_convergence_error(system, state):
    """Mean Euclidean distance from each particle to its ground-truth target."""
    Y = state.reshape(system.num_particles, system.base_dim)
    distances = np.linalg.norm(Y - system.targets, axis=1)
    return np.mean(distances)

def run_single_experiment(params, dt=0.01):
    print(f"  -> Running config: {params}")
    
    system = CANDYDiffusionSystem(num_particles=500)
    system.update_parameters(params)

    np.random.seed(42)
    y0 = system.get_initial_conditions()

    t_span = (0.0, params["T"])
    solver = RK4Solver()
    result = solver.solve(system, y0, t_span, dt)
    
    sample_indices = range(0, len(result.times), 10)
    if len(result.times) - 1 not in sample_indices:
        sample_indices = list(sample_indices) + [len(result.times) - 1]
        
    sampled_times = [result.times[i] for i in sample_indices]
    errors = [compute_convergence_error(system, result.states[i]) for i in sample_indices]
        
    return sampled_times, errors

def plot_line_chart(results, title, ylabel, filename):
    """Renders an academic-style line chart."""
    plt.figure(figsize=(10, 6), dpi=300)
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    
    for idx, (name, data) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        plt.plot(data["time"], data["error"], label=name, linewidth=2.5, color=color, alpha=0.9)
        
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Simulation Time (t)", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[+] Saved: {filename}")

def plot_bar_chart(results, title, ylabel, filename):
    """Renders an academic-style bar chart comparing final errors across groups."""
    plt.figure(figsize=(8, 6), dpi=300)
    names = list(results.keys())
    final_errors = [data["error"][-1] for data in results.values()]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = plt.bar(names, final_errors, color=colors[:len(names)], alpha=0.8, width=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 3), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[+] Saved: {filename}")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("="*60)
    print("Starting CANDY Diffusion Ablation Suite")
    print("="*60)

    print("\n[Experiment 1] CANDY Scale Ablation")
    exp_candy = {
        "Group A: Baseline (scale=0.0)": {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 0.0, "T": 5.0},
        "Group B: Weak (scale=0.5)":     {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 0.5, "T": 5.0},
        "Group C: Standard (scale=1.0)": {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 1.0, "T": 5.0},
        "Group D: Strong (scale=2.0)":   {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 2.0, "T": 5.0},
    }
    res_candy = {}
    for name, p in exp_candy.items():
        t, err = run_single_experiment(p)
        res_candy[name] = {"time": t, "error": err}
    plot_line_chart(res_candy, "Ablation on CANDY Attractor Strength", "Mean Euclidean Distance", os.path.join(FIGURES_DIR, "exp1_candy_scale.png"))

    print("\n[Experiment 2] Diffusion Time Window (T)")
    exp_time = {
        "Short (T=2.0)":    {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 1.0, "T": 2.0},
        "Standard (T=5.0)": {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 1.0, "T": 5.0},
        "Long (T=10.0)":    {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 1.0, "T": 10.0},
    }
    res_time = {}
    for name, p in exp_time.items():
        t, err = run_single_experiment(p)
        res_time[name] = {"time": t, "error": err}
    plot_bar_chart(res_time, "Final Convergence Error vs. Time Window (T)", "Final Error at t=T", os.path.join(FIGURES_DIR, "exp2_time_window.png"))

    print("\n[Experiment 3] U-Net Reconstruction Weight")
    exp_unet = {
        "No U-Net (w=0.0)":   {"decay": 0.3, "unet_weight": 0.0, "origin_weight": 1.5, "candy_scale": 1.0, "T": 5.0},
        "Weak U-Net (w=0.5)": {"decay": 0.3, "unet_weight": 0.5, "origin_weight": 1.5, "candy_scale": 1.0, "T": 5.0},
        "Standard (w=1.0)":   {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 1.0, "T": 5.0},
        "Overdriven (w=2.5)": {"decay": 0.3, "unet_weight": 2.5, "origin_weight": 1.5, "candy_scale": 1.0, "T": 5.0},
    }
    res_unet = {}
    for name, p in exp_unet.items():
        t, err = run_single_experiment(p)
        res_unet[name] = {"time": t, "error": err}
    plot_line_chart(res_unet, "Impact of U-Net Reconstruction Driving Force", "Mean Euclidean Distance", os.path.join(FIGURES_DIR, "exp3_unet_weight.png"))

    print("\n[Experiment 4] Prior Conditioning Strength")
    exp_origin = {
        "Unconditional (w=0.0)": {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 0.0, "candy_scale": 1.0, "T": 5.0},
        "Weak Prior (w=0.5)":    {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 0.5, "candy_scale": 1.0, "T": 5.0},
        "Standard (w=1.5)":      {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 1.0, "T": 5.0},
        "Strong Prior (w=4.0)":  {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 4.0, "candy_scale": 1.0, "T": 5.0},
    }
    res_origin = {}
    for name, p in exp_origin.items():
        t, err = run_single_experiment(p)
        res_origin[name] = {"time": t, "error": err}
    plot_line_chart(res_origin, "Ablation on Target Prior Guidance (CFG)", "Mean Euclidean Distance", os.path.join(FIGURES_DIR, "exp4_origin_weight.png"))

    print("\n[Experiment 5] Phase Space Damping Coefficient")
    exp_decay = {
        "Underdamped (decay=0.1)":  {"decay": 0.1, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 1.0, "T": 5.0},
        "Optimal Damping (decay=1.0)":{"decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 1.0, "T": 5.0},
        "Overdamped (decay=5.0)":   {"decay": 5.0, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 1.0, "T": 5.0},
    }
    res_decay = {}
    for name, p in exp_decay.items():
        t, err = run_single_experiment(p)
        res_decay[name] = {"time": t, "error": err}
    plot_line_chart(res_decay, "Lyapunov Stability and Dissipation (Decay)", "Mean Euclidean Distance", os.path.join(FIGURES_DIR, "exp5_decay_damping.png"))

    print("\n" + "="*60)
    print("[OK] All 5 experiments completed!")
    print("="*60)

if __name__ == "__main__":
    main()