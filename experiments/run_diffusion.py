"""
experiments/run_diffusion_ablations.py

Quantitative Ablation Suite for Continuous-Time Diffusion Models (SDE & ODE).
Generates high-resolution plots for mode convergence, numerical singularity, and thermodynamics.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

from manifold_flow.systems.diffusion import ReverseDiffusionSDE, ProbabilityFlowODE
from manifold_flow.solvers.rk4_solver import RK4Solver

GMM_CENTERS = np.array([
    [ 3.0,  3.0,  3.0], 
    [-3.0, -3.0,  3.0],
    [-3.0,  3.0, -3.0], 
    [ 3.0, -3.0, -3.0]
])

def compute_nearest_mode_error(system, state):
    """Mean distance from each particle to its nearest GMM cluster center."""
    Y = state.reshape(system.num_particles, system.base_dim)[:, :3]
    diff = Y[:, np.newaxis, :] - GMM_CENTERS[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    min_dists = np.min(dists, axis=1)
    return np.mean(min_dists)

def solve_sde_euler_maruyama(system, y0, t_span, dt):
    """Euler-Maruyama SDE solver with Wiener process increments."""
    times = np.arange(t_span[0], t_span[1], dt)
    states = [y0]
    y = y0.copy()
    for t in times[:-1]:
        drift = system.drift(t, y)
        diffusion = system.diffusion(t, y)
        dW = np.random.normal(0, np.sqrt(dt), size=y.shape)
        y = y + drift * dt + diffusion * dW
        states.append(y.copy())
    return times, states

def run_diffusion_experiment(system_cls, params, is_sde=False, dt=0.005):
    print(f"  -> Running {system_cls.__name__} with config: {params}")
    
    system = system_cls(num_particles=500)
    system.update_parameters(params)
    
    np.random.seed(42)
    y0 = system.get_initial_conditions()
    
    t_span = (0.0, 1.0)

    if is_sde:
        times, states = solve_sde_euler_maruyama(system, y0, t_span, dt)
    else:
        solver = RK4Solver()
        result = solver.solve(system, y0, t_span, dt)
        times = result.times
        states = result.states
        
    sample_indices = range(0, len(times), 5)
    sampled_times = [times[i] for i in sample_indices]
    errors = [compute_nearest_mode_error(system, states[i]) for i in sample_indices]
        
    return sampled_times, errors

def plot_diffusion_chart(results, title, filename, y_max=None):
    """Renders an academic-style line chart for diffusion experiment results."""
    plt.figure(figsize=(10, 6), dpi=300)
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
    
    for idx, (name, data) in enumerate(results.items()):
        plt.plot(data["time"], data["error"], label=name, linewidth=2.5, color=colors[idx], alpha=0.85)
        
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Reverse Simulation Time (t $\\to$ 1.0)", fontsize=14)
    plt.ylabel("Mean Distance to Nearest Mode", fontsize=14)
    if y_max:
        plt.ylim(-0.2, y_max)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[+] Saved: {filename}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("="*60)
    print("Starting Diffusion Models Ablation Suite")
    print("="*60)

    print("\n[Experiment 1] SDE vs. Probability Flow ODE")
    exp1_params = {"beta_min": 0.1, "beta_max": 20.0, "temperature": 1.0}
    res_sde_ode = {}
    
    t, err = run_diffusion_experiment(ProbabilityFlowODE, exp1_params, is_sde=False)
    res_sde_ode["Probability Flow ODE (Smooth)"] = {"time": t, "error": err}
    
    t, err = run_diffusion_experiment(ReverseDiffusionSDE, exp1_params, is_sde=True)
    res_sde_ode["Reverse SDE (Langevin Dynamics)"] = {"time": t, "error": err}
    
    plot_diffusion_chart(res_sde_ode, "SDE vs. ODE Phase Space Convergence", os.path.join(FIGURES_DIR, "diff_exp1_sde_vs_ode.png"))

    print("\n[Experiment 2] Maximum Noise Boundary (beta_max)")
    exp_beta_max = {
        "Under-diffused (beta_max=5.0)":  {"beta_min": 0.1, "beta_max": 5.0,  "temperature": 1.0},
        "Optimal (beta_max=20.0)":        {"beta_min": 0.1, "beta_max": 20.0, "temperature": 1.0},
        "Over-diffused (beta_max=50.0)":  {"beta_min": 0.1, "beta_max": 50.0, "temperature": 1.0},
    }
    res_bmax = {}
    for name, p in exp_beta_max.items():
        t, err = run_diffusion_experiment(ProbabilityFlowODE, p, is_sde=False)
        res_bmax[name] = {"time": t, "error": err}
    plot_diffusion_chart(res_bmax, "Ablation on Forward Information Destruction (beta_max)", os.path.join(FIGURES_DIR, "diff_exp2_beta_max.png"), y_max=8.0)

    print("\n[Experiment 3] Thermodynamic Sampling Temperature")
    exp_temp = {
        "Cold Sampling (T=0.1)": {"beta_min": 0.1, "beta_max": 20.0, "temperature": 0.1},
        "Standard (T=1.0)":      {"beta_min": 0.1, "beta_max": 20.0, "temperature": 1.0},
        "Hot Sampling (T=2.0)":  {"beta_min": 0.1, "beta_max": 20.0, "temperature": 2.0},
    }
    res_temp = {}
    for name, p in exp_temp.items():
        t, err = run_diffusion_experiment(ProbabilityFlowODE, p, is_sde=False)
        res_temp[name] = {"time": t, "error": err}
    plot_diffusion_chart(res_temp, "Sampling Diversity and Mode Collapse via Temperature", os.path.join(FIGURES_DIR, "diff_exp3_temperature.png"), y_max=8.0)

    print("\n[Experiment 4] Terminal Singularity Regularization (beta_min)")
    exp_beta_min = {
        "Singularity Risk (beta_min=0.001)":  {"beta_min": 0.001, "beta_max": 20.0, "temperature": 1.0},
        "Optimal Regularized (beta_min=0.1)": {"beta_min": 0.1,   "beta_max": 20.0, "temperature": 1.0},
        "High Residual Noise (beta_min=2.0)": {"beta_min": 2.0,   "beta_max": 20.0, "temperature": 1.0},
    }
    res_bmin = {}
    for name, p in exp_beta_min.items():
        t, err = run_diffusion_experiment(ProbabilityFlowODE, p, is_sde=False)
        res_bmin[name] = {"time": t, "error": err}
    plot_diffusion_chart(res_bmin, "Phase Space Singularity & Terminal Residual Noise (beta_min)", os.path.join(FIGURES_DIR, "diff_exp4_beta_min.png"), y_max=8.0)

    print("\n" + "="*60)
    print("[OK] All Diffusion experiments completed!")
    print("="*60)

if __name__ == "__main__":
    main()