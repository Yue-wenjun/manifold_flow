"""
experiments/candy_experiment.py

CANDY Diffusion Experiment
Compares three variants of the CANDYDiffusionSystem to isolate the contribution
of each architectural component:

  1. Baseline (candy_scale=0)  — U-Net + prior conditioning only, no CANDY masking
  2. Standard CANDY            — full dynamics (CANDY masking + U-Net + prior)
  3. CANDY-only (unet_weight=0)— CANDY masking + prior, no U-Net reconstruction

All three are deterministic ODEs integrated with RK4 over t ∈ [0, T_param].

Metrics (tracked over time for each variant):
  - Convergence error  : mean Euclidean distance from each particle to its assigned target
  - Cluster purity     : fraction of particles within CAPTURE_RADIUS of their assigned target
  - Inter-class gap    : mean distance between the four class-cluster centroids in current state
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from manifold_flow.systems.neural import CANDYDiffusionSystem
from manifold_flow.solvers.rk4_solver import RK4Solver


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAPTURE_RADIUS = 1.5   # distance threshold to call a particle "converged"
NUM_CLASSES    = 4
CLASS_CENTERS  = np.array([
    [ 4.0,  4.0,  4.0],
    [-4.0, -4.0,  4.0],
    [-4.0,  4.0, -4.0],
    [ 4.0, -4.0, -4.0],
])

# Canonical parameter sets for the three variants
VARIANT_CONFIGS = {
    "baseline":     {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 0.0, "T": 5.0},
    "standard":     {"decay": 0.3, "unet_weight": 1.0, "origin_weight": 1.5, "candy_scale": 1.0, "T": 5.0},
    "candy_only":   {"decay": 0.3, "unet_weight": 0.0, "origin_weight": 1.5, "candy_scale": 1.0, "T": 5.0},
}

VARIANT_LABELS = {
    "baseline":   "Baseline (no CANDY)",
    "standard":   "Standard CANDY",
    "candy_only": "CANDY-only (no U-Net)",
}

VARIANT_COLORS = {
    "baseline":   "steelblue",
    "standard":   "tomato",
    "candy_only": "forestgreen",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def convergence_error(states_flat: np.ndarray, targets: np.ndarray) -> float:
    """Mean Euclidean distance from each particle to its assigned target."""
    Y = states_flat.reshape(targets.shape)
    return float(np.mean(np.linalg.norm(Y - targets, axis=1)))


def cluster_purity(states_flat: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of particles within CAPTURE_RADIUS of their assigned target."""
    Y = states_flat.reshape(targets.shape)
    dists = np.linalg.norm(Y - targets, axis=1)
    return float(np.mean(dists < CAPTURE_RADIUS))


def inter_class_gap(states_flat: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean distance between the four class-cluster centroids computed from
    the current particle positions.  Higher = better class separation.
    """
    Y = states_flat.reshape(targets.shape)
    centroids = np.array([
        Y[np.all(targets == CLASS_CENTERS[k], axis=1)].mean(axis=0)
        for k in range(NUM_CLASSES)
    ])
    # Mean of all pairwise distances between centroids
    n = NUM_CLASSES
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(centroids[i] - centroids[j])
            count += 1
    return float(total / max(count, 1))


# ---------------------------------------------------------------------------
# Per-variant runner
# ---------------------------------------------------------------------------

class CANDYVariantRun:
    def __init__(self, key: str, params: dict, num_particles: int, seed: int = 42):
        self.key          = key
        self.label        = VARIANT_LABELS[key]
        self.color        = VARIANT_COLORS[key]
        self.num_particles = num_particles
        self.seed         = seed

        system = CANDYDiffusionSystem(num_particles=num_particles)
        system.update_parameters(params)
        self.system  = system
        self.params  = params
        self.result  = None
        self.snapshot_indices = []

    def run(self, dt: float = 0.01):
        np.random.seed(self.seed)
        y0 = self.system.get_initial_conditions()
        t_span = (0.0, self.params["T"])

        solver = RK4Solver()
        print(f"  [{self.label}] Integrating (dt={dt}, T={self.params['T']})...")
        self.result = solver.solve(self.system, y0, t_span, dt)

        n = len(self.result.times)
        self.snapshot_indices = [int(i * (n - 1) / 4) for i in range(5)]
        print(f"  [{self.label}] Done. Trajectory shape: {self.result.states.shape}")

    # Metric time-series -------------------------------------------------------
    def error_series(self):
        return [convergence_error(s, self.system.targets) for s in self.result.states]

    def purity_series(self):
        return [cluster_purity(s, self.system.targets) for s in self.result.states]

    def gap_series(self):
        return [inter_class_gap(s, self.system.targets) for s in self.result.states]

    def get_snapshot(self, idx: int) -> np.ndarray:
        """Returns (num_particles, 3) array at trajectory index idx."""
        return self.result.states[idx].reshape(self.num_particles, 3)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

class CANDYExperiment:
    """
    Orchestrates the three CANDY variant experiments and produces
    publication-ready figures + a JSON result summary.
    """

    def __init__(
        self,
        num_particles: int = 500,
        experiment_name: str = "candy_comparison",
    ):
        self.num_particles   = num_particles
        self.experiment_name = experiment_name
        self.timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.runs: dict[str, CANDYVariantRun] = {}
        self.metrics: dict = {}

    # ------------------------------------------------------------------
    def setup(self):
        print(f"[{self.experiment_name}] Setting up CANDY variants ({self.num_particles} particles)...")
        for key, params in VARIANT_CONFIGS.items():
            self.runs[key] = CANDYVariantRun(key, params, self.num_particles)
        print("[OK] All variants ready")

    # ------------------------------------------------------------------
    def run_all(self, dt: float = 0.01):
        print(f"[{self.experiment_name}] Running all three variants...")
        for run in self.runs.values():
            run.run(dt=dt)
        print("[OK] All integrations complete")

    # ------------------------------------------------------------------
    def analyze(self):
        print(f"[{self.experiment_name}] Analysing results...")

        for key, run in self.runs.items():
            err    = run.error_series()
            purity = run.purity_series()
            gap    = run.gap_series()

            self.metrics[f"{key}_error_initial"]  = err[0]
            self.metrics[f"{key}_error_final"]    = err[-1]
            self.metrics[f"{key}_purity_final"]   = purity[-1]
            self.metrics[f"{key}_gap_final"]      = gap[-1]

            # Store time-series on the run object for plotting
            run._err    = err
            run._purity = purity
            run._gap    = gap

        self._print_summary()

    def _print_summary(self):
        for key in VARIANT_CONFIGS:
            m = self.metrics
            print(f"\n  --- {VARIANT_LABELS[key]} ---")
            print(f"    Convergence error : {m[f'{key}_error_initial']:.4f} -> {m[f'{key}_error_final']:.4f}")
            print(f"    Cluster purity    : {m[f'{key}_purity_final']:.2%}")
            print(f"    Inter-class gap   : {m[f'{key}_gap_final']:.4f}")

    # ------------------------------------------------------------------
    def visualize(self, save_dir: str = "./results"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[{self.experiment_name}] Generating visualizations...")

        self._plot_metrics(save_dir)
        self._plot_3d_snapshots(save_dir)
        self._plot_final_state(save_dir)

        print(f"[OK] Visualizations saved to {save_dir}/")

    def _plot_metrics(self, save_dir: str):
        """Three-panel metrics: convergence error, cluster purity, inter-class gap."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for key, run in self.runs.items():
            t = run.result.times
            c = run.color
            lbl = run.label

            axes[0].plot(t, run._err,    color=c, linewidth=2, label=lbl)
            axes[1].plot(t, run._purity, color=c, linewidth=2, label=lbl)
            axes[2].plot(t, run._gap,    color=c, linewidth=2, label=lbl)

        axes[0].set_title("Convergence Error (lower = better)")
        axes[0].set_xlabel("Time t")
        axes[0].set_ylabel("Mean dist. to assigned target")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].set_title(f"Cluster Purity (within r={CAPTURE_RADIUS})")
        axes[1].set_xlabel("Time t")
        axes[1].set_ylabel("Fraction converged")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].set_title("Inter-class Gap (higher = better separation)")
        axes[2].set_xlabel("Time t")
        axes[2].set_ylabel("Mean centroid-pair distance")
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        fig.suptitle("CANDY Diffusion -- Metrics Over Time", fontsize=14)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/candy_metrics.png", dpi=150)
        plt.close()

    def _plot_3d_snapshots(self, save_dir: str):
        """3-column grid (initial | mid | final) for each variant."""
        keys   = list(VARIANT_CONFIGS.keys())
        n_rows = len(keys)

        fig = plt.figure(figsize=(15, 4 * n_rows))
        snap_positions = [0, 2, 4]
        col_titles = ["t = 0 (initial)", "t = T/2 (mid)", "t = T (final)"]

        for row, key in enumerate(keys):
            run = self.runs[key]
            for col, snap_pos in enumerate(snap_positions):
                ax = fig.add_subplot(n_rows, 3, row * 3 + col + 1, projection="3d")
                snap_idx = run.snapshot_indices[snap_pos]
                pts = run.get_snapshot(snap_idx)

                # Color particles by assigned class (0-3)
                targets = run.system.targets
                class_ids = np.array([
                    np.argmin(np.linalg.norm(targets[i] - CLASS_CENTERS, axis=1))
                    for i in range(run.num_particles)
                ])
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=class_ids, cmap="tab10", alpha=0.3, s=4, vmin=0, vmax=9)

                # Draw target class centers as stars
                ax.scatter(CLASS_CENTERS[:, 0], CLASS_CENTERS[:, 1], CLASS_CENTERS[:, 2],
                           c="black", marker="*", s=150, zorder=5)

                if row == 0:
                    ax.set_title(col_titles[col], fontsize=9)
                ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
                ax.set_zlabel("Z", fontsize=7)
                ax.tick_params(labelsize=6)
                if col == 0:
                    ax.text2D(-0.25, 0.5, run.label, transform=ax.transAxes,
                              fontsize=8, rotation=90, va="center")

        fig.suptitle("CANDY Diffusion -- 3D Particle Snapshots (color = class)", fontsize=13)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/candy_snapshots_3d.png", dpi=150)
        plt.close()

    def _plot_final_state(self, save_dir: str):
        """Side-by-side 3D scatter of all three variants at t = T."""
        keys = list(VARIANT_CONFIGS.keys())
        fig = plt.figure(figsize=(6 * len(keys), 5))

        for col, key in enumerate(keys):
            run = self.runs[key]
            ax = fig.add_subplot(1, len(keys), col + 1, projection="3d")

            pts = run.get_snapshot(-1)
            targets = run.system.targets
            class_ids = np.array([
                np.argmin(np.linalg.norm(targets[i] - CLASS_CENTERS, axis=1))
                for i in range(run.num_particles)
            ])

            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=class_ids, cmap="tab10", alpha=0.3, s=5,
                       vmin=0, vmax=9, label="Particles")
            ax.scatter(CLASS_CENTERS[:, 0], CLASS_CENTERS[:, 1], CLASS_CENTERS[:, 2],
                       c="black", marker="*", s=200, zorder=5, label="Targets")

            ax.set_title(f"{run.label}\nFinal state (t=T)", fontsize=10)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            if col == 0:
                ax.legend(fontsize=7)

        fig.suptitle("CANDY Diffusion -- Final State Comparison", fontsize=13)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/candy_final_comparison.png", dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    def save_results(self, save_dir: str = "./results"):
        os.makedirs(save_dir, exist_ok=True)

        result_file = f"{save_dir}/{self.experiment_name}_{self.timestamp}_results.json"

        metrics_out = {}
        for k, v in self.metrics.items():
            if isinstance(v, np.ndarray):
                metrics_out[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                metrics_out[k] = v.item()
            else:
                metrics_out[k] = v

        output = {
            "experiment":  self.experiment_name,
            "timestamp":   self.timestamp,
            "config": {
                "num_particles": self.num_particles,
                "variants": VARIANT_CONFIGS,
            },
            "metrics": metrics_out,
        }
        with open(result_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[OK] Results saved: {result_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CANDY Diffusion Experiment")
    print("=" * 60)

    exp = CANDYExperiment(num_particles=500, experiment_name="candy_comparison")

    exp.setup()
    exp.run_all(dt=0.01)
    exp.analyze()
    FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    exp.visualize(save_dir=FIGURES_DIR)
    exp.save_results(save_dir=FIGURES_DIR)

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)

    return exp


if __name__ == "__main__":
    exp = main()
