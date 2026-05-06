"""
experiments/neural_experiment.py

Pure Neural Systems Experiment
Runs two neural dynamical systems and measures convergence behaviour:

  1. HopfieldNetwork          -- 1000 particles, 4 neurons, 3 stored attractors
  2. TransformerAttentionSystem -- 1000 tokens, 3D embeddings, continuous attention
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from manifold_flow.systems.neural import HopfieldNetwork, TransformerAttentionSystem
from manifold_flow.solvers.rk4_solver import RK4Solver


# Hopfield stored patterns (from neural.py)
HOPFIELD_PATTERNS = np.array([
    [ 1.0, -1.0,  1.0, -1.0],
    [-1.0,  1.0, -1.0,  1.0],
    [ 1.0,  1.0, -1.0, -1.0],
])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def hopfield_pattern_distance(states_flat, num_particles, base_dim=4):
    """Mean distance from each particle to its nearest stored pattern."""
    Y = states_flat.reshape(num_particles, base_dim)
    dists = np.linalg.norm(Y[:, np.newaxis] - HOPFIELD_PATTERNS[np.newaxis], axis=2)
    return float(np.mean(np.min(dists, axis=1)))


def hopfield_convergence_rate(states_flat, num_particles, base_dim=4, threshold=0.5):
    """Fraction of particles within threshold of any stored pattern."""
    Y = states_flat.reshape(num_particles, base_dim)
    dists = np.linalg.norm(Y[:, np.newaxis] - HOPFIELD_PATTERNS[np.newaxis], axis=2)
    return float(np.mean(np.min(dists, axis=1) < threshold))


def token_spread(states_flat, num_particles, embed_dim=3):
    """Mean pairwise distance between token embeddings (lower = more clustered)."""
    Y = states_flat.reshape(num_particles, embed_dim)
    # Approximate via std of coordinates
    return float(np.mean(np.std(Y, axis=0)))


# ---------------------------------------------------------------------------
# Per-system runner
# ---------------------------------------------------------------------------

class NeuralSystemRun:
    def __init__(self, system, name, num_particles, base_dim):
        self.system = system
        self.name = name
        self.num_particles = num_particles
        self.base_dim = base_dim
        self.result = None

    def run(self, dt=0.02):
        solver = RK4Solver()
        y0 = self.system.get_initial_conditions()
        print(f"  [{self.name}] Integrating (dt={dt}, steps={int(1.0/dt)})...")
        self.result = solver.solve(self.system, y0, (0.0, 1.0), dt)
        print(f"  [{self.name}] Done. Shape: {self.result.states.shape}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

class NeuralExperiment:
    def __init__(self, num_particles=500, experiment_name="neural_experiment"):
        self.num_particles = num_particles
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hopfield = None
        self.transformer = None
        self.metrics = {}

    def setup(self):
        print(f"[{self.experiment_name}] Setting up neural systems ({self.num_particles} particles)...")
        self.hopfield = NeuralSystemRun(
            system=HopfieldNetwork(num_neurons=4, num_particles=self.num_particles),
            name="HopfieldNetwork",
            num_particles=self.num_particles,
            base_dim=4,
        )
        self.transformer = NeuralSystemRun(
            system=TransformerAttentionSystem(embed_dim=3, num_particles=self.num_particles),
            name="TransformerAttention",
            num_particles=self.num_particles,
            base_dim=3,
        )
        print("[OK] Systems ready")

    def run(self):
        print(f"[{self.experiment_name}] Running neural systems...")
        self.hopfield.run(dt=0.02)
        self.transformer.run(dt=0.02)
        print("[OK] All integrations complete")

    def analyze(self):
        print(f"[{self.experiment_name}] Analysing results...")

        hop_states = self.hopfield.result.states
        tfm_states = self.transformer.result.states

        # Hopfield: pattern convergence
        hop_dist = [hopfield_pattern_distance(s, self.num_particles) for s in hop_states]
        hop_conv = [hopfield_convergence_rate(s, self.num_particles) for s in hop_states]

        # Transformer: token clustering
        tfm_spread = [token_spread(s, self.num_particles) for s in tfm_states]

        self.metrics.update({
            "hopfield_pattern_dist_initial": hop_dist[0],
            "hopfield_pattern_dist_final":   hop_dist[-1],
            "hopfield_convergence_final":    hop_conv[-1],
            "transformer_spread_initial": tfm_spread[0],
            "transformer_spread_final":   tfm_spread[-1],
            "transformer_spread_ratio":   tfm_spread[-1] / max(tfm_spread[0], 1e-9),
        })

        self._hop_times  = self.hopfield.result.times
        self._tfm_times  = self.transformer.result.times
        self._hop_dist   = hop_dist
        self._hop_conv   = hop_conv
        self._tfm_spread = tfm_spread

        print(f"\n  --- HopfieldNetwork ---")
        print(f"    Pattern distance : {hop_dist[0]:.4f} -> {hop_dist[-1]:.4f}")
        print(f"    Convergence rate : {hop_conv[-1]:.2%}")
        print(f"\n  --- TransformerAttention ---")
        print(f"    Token spread : {tfm_spread[0]:.4f} -> {tfm_spread[-1]:.4f}  "
              f"(x{self.metrics['transformer_spread_ratio']:.2f})")

    def visualize(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[{self.experiment_name}] Generating visualizations...")

        # --- Plot 1: metrics ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(self._hop_times, self._hop_dist, "b-", linewidth=2)
        axes[0].set_title("Hopfield: Mean Distance to Nearest Pattern")
        axes[0].set_xlabel("Time t"); axes[0].set_ylabel("Distance")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self._hop_times, self._hop_conv, "g-", linewidth=2)
        axes[1].set_title("Hopfield: Convergence Rate (< 0.5 of pattern)")
        axes[1].set_xlabel("Time t"); axes[1].set_ylabel("Fraction converged")
        axes[1].set_ylim(0, 1.05); axes[1].grid(True, alpha=0.3)

        axes[2].plot(self._tfm_times, self._tfm_spread, "r-", linewidth=2)
        axes[2].set_title("Transformer: Token Spread (std of positions)")
        axes[2].set_xlabel("Time t"); axes[2].set_ylabel("Mean coord std")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("Neural Systems -- Dynamics Metrics", fontsize=13)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/neural_metrics.png", dpi=150)
        plt.close()

        # --- Plot 2: 3D snapshots ---
        fig = plt.figure(figsize=(18, 8))

        snap_frac = [0, 0.5, 1.0]
        for row, (run, label, color) in enumerate([
            (self.hopfield,    "HopfieldNetwork (first 3 neurons)", "steelblue"),
            (self.transformer, "TransformerAttention (3D embedding)",  "tomato"),
        ]):
            states = run.result.states
            n = len(states)
            for col, frac in enumerate(snap_frac):
                idx = int(frac * (n - 1))
                ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
                pts = states[idx].reshape(run.num_particles, run.base_dim)[:, :3]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=color, alpha=0.2, s=4)
                if row == 0:
                    # Draw stored patterns as large markers
                    ax.scatter(HOPFIELD_PATTERNS[:, 0],
                               HOPFIELD_PATTERNS[:, 1],
                               HOPFIELD_PATTERNS[:, 2],
                               c="black", marker="*", s=200, zorder=5)
                t_val = run.result.times[idx]
                ax.set_title(f"{label.split('(')[0].strip()}\nt={t_val:.2f}", fontsize=8)
                ax.tick_params(labelsize=6)

        fig.suptitle("Neural Systems -- 3D Particle Snapshots", fontsize=13)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/neural_snapshots.png", dpi=150)
        plt.close()

        print(f"[OK] Visualizations saved to {save_dir}/")

    def save_results(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        out = {
            "experiment": self.experiment_name,
            "timestamp": self.timestamp,
            "config": {"num_particles": self.num_particles},
            "metrics": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in self.metrics.items()},
        }
        path = f"{save_dir}/{self.experiment_name}_{self.timestamp}_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] Results saved: {path}")


def main():
    print("=" * 60)
    print("Neural Systems Experiment")
    print("=" * 60)
    exp = NeuralExperiment(num_particles=500)
    exp.setup()
    exp.run()
    exp.analyze()
    FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    exp.visualize(save_dir=FIGURES_DIR)
    exp.save_results(save_dir=FIGURES_DIR)
    print("\n[OK] Done.")
    return exp

if __name__ == "__main__":
    exp = main()
