"""
experiments/supplementary_experiments.py

Two supplementary experiments for the CANDY ablation study:

  Exp S1 — Masking Isolation (origin_weight=0)
      Sweeps candy_scale s ∈ {0.0, 0.5, 1.0, 2.0} while setting
      origin_weight=0 and unet_weight=1.0.  Removes the dominant
      origin-guidance term so the masking branch can be isolated.

  Exp S2 — Architecture Value: CANDY vs Pure Gradient Flow
      Compares three systems that all receive the SAME oracle targets:
        (a) Pure gradient flow:  dY = -γY + w_o·g_t·(T−Y)
            [candy_scale=0, unet_weight=0]
        (b) U-Net only:          adds the U-Net fusion term, no masking
            [candy_scale=0, unet_weight=1.0]
        (c) Full CANDY:          masking + U-Net + origin-guidance
            [candy_scale=1.0, unet_weight=1.0]
      All three use origin_weight=1.5, decay=0.3, T=5.0.
      Shows whether the attractor architecture contributes beyond
      the oracle targets alone.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

from manifold_flow.systems.neural import CANDYDiffusionSystem
from manifold_flow.solvers.rk4_solver import RK4Solver


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def compute_error(system, state):
    Y = state.reshape(system.num_particles, system.base_dim)
    return float(np.mean(np.linalg.norm(Y - system.targets, axis=1)))


def run_experiment(params, dt=0.01, seed=42):
    system = CANDYDiffusionSystem(num_particles=500)
    system.update_parameters(params)
    np.random.seed(seed)
    y0 = system.get_initial_conditions()
    solver = RK4Solver()
    result = solver.solve(system, y0, (0.0, params["T"]), dt)

    idx = list(range(0, len(result.times), 10))
    if (len(result.times) - 1) not in idx:
        idx.append(len(result.times) - 1)

    times  = [result.times[i] for i in idx]
    errors = [compute_error(system, result.states[i]) for i in idx]
    return times, errors


# ---------------------------------------------------------------------------
# Exp S1 — Masking isolation  (origin_weight = 0)
# ---------------------------------------------------------------------------

def run_exp_s1():
    print("\n[Exp S1] Masking scale isolation  (origin_weight=0, unet_weight=1.0)")

    configs = {
        r"$s=0.0$ (no masking)": {
            "decay": 0.3, "unet_weight": 1.0, "origin_weight": 0.0,
            "candy_scale": 0.0, "T": 5.0},
        r"$s=0.5$ (weak)": {
            "decay": 0.3, "unet_weight": 1.0, "origin_weight": 0.0,
            "candy_scale": 0.5, "T": 5.0},
        r"$s=1.0$ (standard)": {
            "decay": 0.3, "unet_weight": 1.0, "origin_weight": 0.0,
            "candy_scale": 1.0, "T": 5.0},
        r"$s=2.0$ (strong)": {
            "decay": 0.3, "unet_weight": 1.0, "origin_weight": 0.0,
            "candy_scale": 2.0, "T": 5.0},
    }

    results = {}
    for name, p in configs.items():
        t, e = run_experiment(p)
        results[name] = (t, e)
        print(f"  {name:30s}  final error = {e[-1]:.4f}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    colors = ["#555555", "#1f77b4", "#2ca02c", "#d62728"]
    ls     = ["--", "-.", ":", "-"]
    for (name, (t, e)), c, l in zip(results.items(), colors, ls):
        ax.plot(t, e, label=name, color=c, linestyle=l, linewidth=2.2)

    ax.set_xlabel("Simulation time $t$", fontsize=12)
    ax.set_ylabel("Mean distance to targets", fontsize=12)
    ax.set_title("Exp S1 — Masking scale isolation ($w_o=0$)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp_s1_masking_isolation.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> Saved: {path}")
    return results


# ---------------------------------------------------------------------------
# Exp S2 — Architecture value: pure gradient flow vs CANDY
# ---------------------------------------------------------------------------

def run_exp_s2():
    print("\n[Exp S2] Architecture value: pure gradient flow vs CANDY")
    print("         (all three systems receive the same oracle targets)")

    configs = {
        "Pure gradient flow\n"
        r"($s{=}0,\,w_u{=}0$)": {
            "decay": 0.3, "unet_weight": 0.0, "origin_weight": 1.5,
            "candy_scale": 0.0, "T": 5.0},
        "Gradient + U-Net\n"
        r"($s{=}0,\,w_u{=}1$)": {
            "decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5,
            "candy_scale": 0.0, "T": 5.0},
        "Full CANDY\n"
        r"($s{=}1,\,w_u{=}1$)": {
            "decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5,
            "candy_scale": 1.0, "T": 5.0},
    }

    results = {}
    for name, p in configs.items():
        t, e = run_experiment(p)
        results[name] = (t, e)
        label = name.split("\n")[0]
        print(f"  {label:30s}  final error = {e[-1]:.4f}")

    # ---- plot: convergence curves + bar of final errors ----
    fig = plt.figure(figsize=(11, 4.5), dpi=300)
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    colors = ["#ff7f0e", "#1f77b4", "#d62728"]
    ls     = ["--", "-.", "-"]
    short_names = []
    final_errors = []

    for (name, (t, e)), c, l in zip(results.items(), colors, ls):
        label = name.replace("\n", " ")
        ax1.plot(t, e, label=label, color=c, linestyle=l, linewidth=2.2)
        short_names.append(name.split("\n")[0])
        final_errors.append(e[-1])

    ax1.set_xlabel("Simulation time $t$", fontsize=12)
    ax1.set_ylabel("Mean distance to targets", fontsize=12)
    ax1.set_title("Convergence curves", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    bars = ax2.bar(range(len(short_names)), final_errors,
                   color=colors, alpha=0.82, width=0.5, edgecolor="white")
    for bar, val in zip(bars, final_errors):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(len(short_names)))
    ax2.set_xticklabels(short_names, fontsize=9.5)
    ax2.set_ylabel("Final error at $t=T$", fontsize=12)
    ax2.set_title("Final error comparison", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Exp S2 — Architecture contribution (oracle targets held constant)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp_s2_architecture_value.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> Saved: {path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 60)
    print("Supplementary Experiments S1 & S2")
    print("=" * 60)

    res_s1 = run_exp_s1()
    res_s2 = run_exp_s2()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nExp S1 — Masking scale (w_o=0):")
    for name, (_, e) in res_s1.items():
        print(f"  {name:35s}  final={e[-1]:.4f}")

    print("\nExp S2 — Architecture value:")
    for name, (_, e) in res_s2.items():
        label = name.split("\n")[0]
        print(f"  {label:35s}  final={e[-1]:.4f}")

    print("\n[Done] Figures saved to:", FIGURES_DIR)
