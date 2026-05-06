"""
experiments/diffusion_experiment.py

Diffusion Systems Experiment
Runs three diffusion processes side-by-side and compares their behaviour:

  1. ForwardDiffusionSDE  — tight cluster → Gaussian noise  (stochastic, Euler-Maruyama)
  2. ReverseDiffusionSDE  — Gaussian noise → 4 GMM clusters (stochastic, Euler-Maruyama)
  3. ProbabilityFlowODE   — Gaussian noise → 4 GMM clusters (deterministic, RK4)

GMM cluster centers (from ReverseDiffusionSDE / ProbabilityFlowODE):
    [ 3,  3,  3]  [-3, -3,  3]  [-3,  3, -3]  [ 3, -3, -3]

Metrics:
  - Forward:  particle spread (std of positions) over time
  - Reverse / ODE: cluster-separation score and cluster purity over time
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from manifold_flow.systems.diffusion import (
    ForwardDiffusionSDE,
    ReverseDiffusionSDE,
    ProbabilityFlowODE,
)
from manifold_flow.solvers.euler_maruyama import EulerMaruyamaSolver
from manifold_flow.solvers.rk4_solver import RK4Solver


# ---------------------------------------------------------------------------
# GMM cluster centers (must match those hard-coded in diffusion.py)
# ---------------------------------------------------------------------------
GMM_CENTERS = np.array([
    [ 3.0,  3.0,  3.0],
    [-3.0, -3.0,  3.0],
    [-3.0,  3.0, -3.0],
    [ 3.0, -3.0, -3.0],
])

CAPTURE_RADIUS = 2.0  # distance threshold to count a particle as "in cluster"


# ---------------------------------------------------------------------------
# Helper metrics
# ---------------------------------------------------------------------------

def particle_spread(states_flat: np.ndarray, num_particles: int, base_dim: int = 3) -> float:
    """Mean standard deviation of particle coordinates — measures diffuseness."""
    Y = states_flat.reshape(num_particles, base_dim)
    return float(np.mean(np.std(Y, axis=0)))


def cluster_separation_score(states_flat: np.ndarray, num_particles: int, base_dim: int = 3) -> float:
    """Mean distance from each particle to its nearest GMM center (lower = better condensed)."""
    Y = states_flat.reshape(num_particles, base_dim)
    dists = np.linalg.norm(Y[:, np.newaxis, :] - GMM_CENTERS[np.newaxis, :, :], axis=2)  # (N, 4)
    return float(np.mean(np.min(dists, axis=1)))


def cluster_purity(states_flat: np.ndarray, num_particles: int, base_dim: int = 3) -> float:
    """Fraction of particles within CAPTURE_RADIUS of their nearest GMM center."""
    Y = states_flat.reshape(num_particles, base_dim)
    dists = np.linalg.norm(Y[:, np.newaxis, :] - GMM_CENTERS[np.newaxis, :, :], axis=2)
    return float(np.mean(np.min(dists, axis=1) < CAPTURE_RADIUS))


# ---------------------------------------------------------------------------
# Per-system experiment runner
# ---------------------------------------------------------------------------

class DiffusionSystemExperiment:
    """Runs and records one diffusion system's trajectory."""

    def __init__(self, system, solver, name: str, num_particles: int):
        self.system = system
        self.solver = solver
        self.name = name
        self.num_particles = num_particles
        self.result = None   # SolverResult
        self.snapshot_indices = []

    def run(self, t_span=(0.0, 1.0), dt: float = 0.005):
        y0 = self.system.get_initial_conditions()
        print(f"  [{self.name}] Integrating (dt={dt}, steps~{int((t_span[1]-t_span[0])/dt)})...")
        self.result = self.solver.solve(self.system, y0, t_span, dt)
        n = len(self.result.times)
        # Pick ~5 evenly-spaced snapshot indices for visualisation
        self.snapshot_indices = [int(i * (n - 1) / 4) for i in range(5)]
        print(f"  [{self.name}] Done. Trajectory shape: {self.result.states.shape}")

    def spread_over_time(self):
        return [particle_spread(s, self.num_particles) for s in self.result.states]

    def separation_over_time(self):
        return [cluster_separation_score(s, self.num_particles) for s in self.result.states]

    def purity_over_time(self):
        return [cluster_purity(s, self.num_particles) for s in self.result.states]

    def get_snapshot(self, idx: int) -> np.ndarray:
        """Returns (num_particles, 3) array at trajectory index idx."""
        return self.result.states[idx].reshape(self.num_particles, 3)


# ---------------------------------------------------------------------------
# Main experiment orchestrator
# ---------------------------------------------------------------------------

class DiffusionExperiment:
    """
    Orchestrates the three diffusion experiments:
      1. Forward SDE
      2. Reverse SDE
      3. Probability Flow ODE
    """

    def __init__(
        self,
        num_particles: int = 500,
        experiment_name: str = "diffusion_comparison",
    ):
        self.num_particles = num_particles
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.runs: dict[str, DiffusionSystemExperiment] = {}
        self.metrics: dict = {}

    # ------------------------------------------------------------------
    def setup(self):
        print(f"[{self.experiment_name}] Setting up systems ({self.num_particles} particles)...")

        em_solver = EulerMaruyamaSolver(dt=0.005)
        rk4_solver = RK4Solver()

        self.runs["forward_sde"] = DiffusionSystemExperiment(
            system=ForwardDiffusionSDE(state_dim=3, num_particles=self.num_particles),
            solver=em_solver,
            name="ForwardSDE",
            num_particles=self.num_particles,
        )
        self.runs["reverse_sde"] = DiffusionSystemExperiment(
            system=ReverseDiffusionSDE(state_dim=3, num_particles=self.num_particles),
            solver=em_solver,
            name="ReverseSDE",
            num_particles=self.num_particles,
        )
        self.runs["prob_flow_ode"] = DiffusionSystemExperiment(
            system=ProbabilityFlowODE(state_dim=3, num_particles=self.num_particles),
            solver=rk4_solver,
            name="ProbFlowODE",
            num_particles=self.num_particles,
        )
        print("[OK] Systems ready")

    # ------------------------------------------------------------------
    def run_all(self, dt: float = 0.005):
        print(f"[{self.experiment_name}] Running all three systems...")
        for exp in self.runs.values():
            exp.run(t_span=(0.0, 1.0), dt=dt)
        print("[OK] All integrations complete")

    # ------------------------------------------------------------------
    def analyze(self):
        print(f"[{self.experiment_name}] Analysing results...")

        # --- Forward SDE: particle spread ---
        fwd = self.runs["forward_sde"]
        fwd_spread = fwd.spread_over_time()
        self.metrics["forward_spread_initial"] = fwd_spread[0]
        self.metrics["forward_spread_final"]   = fwd_spread[-1]
        self.metrics["forward_spread_ratio"]   = fwd_spread[-1] / max(fwd_spread[0], 1e-9)

        # --- Reverse SDE: separation + purity ---
        rev = self.runs["reverse_sde"]
        rev_sep   = rev.separation_over_time()
        rev_purity = rev.purity_over_time()
        self.metrics["reverse_sde_separation_initial"] = rev_sep[0]
        self.metrics["reverse_sde_separation_final"]   = rev_sep[-1]
        self.metrics["reverse_sde_purity_final"]       = rev_purity[-1]

        # --- Probability Flow ODE: same metrics ---
        ode = self.runs["prob_flow_ode"]
        ode_sep    = ode.separation_over_time()
        ode_purity = ode.purity_over_time()
        self.metrics["ode_separation_initial"] = ode_sep[0]
        self.metrics["ode_separation_final"]   = ode_sep[-1]
        self.metrics["ode_purity_final"]       = ode_purity[-1]

        # Store time-series for plotting
        self._fwd_spread  = fwd_spread
        self._rev_sep     = rev_sep
        self._rev_purity  = rev_purity
        self._ode_sep     = ode_sep
        self._ode_purity  = ode_purity

        self._print_summary()

    def _print_summary(self):
        m = self.metrics
        print("\n  --- Forward SDE ---")
        print(f"    Particle spread   initial : {m['forward_spread_initial']:.4f}")
        print(f"    Particle spread   final   : {m['forward_spread_final']:.4f}  "
              f"(x{m['forward_spread_ratio']:.1f} increase)")
        print("\n  --- Reverse SDE ---")
        print(f"    Cluster separation initial: {m['reverse_sde_separation_initial']:.4f}")
        print(f"    Cluster separation final  : {m['reverse_sde_separation_final']:.4f}")
        print(f"    Cluster purity     final  : {m['reverse_sde_purity_final']:.2%}")
        print("\n  --- Probability Flow ODE ---")
        print(f"    Cluster separation initial: {m['ode_separation_initial']:.4f}")
        print(f"    Cluster separation final  : {m['ode_separation_final']:.4f}")
        print(f"    Cluster purity     final  : {m['ode_purity_final']:.2%}")

    # ------------------------------------------------------------------
    def visualize(self, save_dir: str = "./results"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[{self.experiment_name}] Generating visualizations...")

        self._plot_metrics(save_dir)
        self._plot_3d_snapshots(save_dir)
        self._plot_comparison_final(save_dir)

        print(f"[OK] Visualizations saved to {save_dir}/")

    def _plot_metrics(self, save_dir: str):
        """Two-panel metrics: spread (forward) and separation+purity (reverse/ODE)."""
        times_fwd = self.runs["forward_sde"].result.times
        times_rev = self.runs["reverse_sde"].result.times
        times_ode = self.runs["prob_flow_ode"].result.times

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: Forward spread
        axes[0].plot(times_fwd, self._fwd_spread, "b-", linewidth=2)
        axes[0].set_title("Forward SDE: Particle Spread")
        axes[0].set_xlabel("Time t")
        axes[0].set_ylabel("Mean std of particle positions")
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Cluster separation (lower = better condensed)
        axes[1].plot(times_rev, self._rev_sep, "r-",  linewidth=2, label="Reverse SDE")
        axes[1].plot(times_ode, self._ode_sep, "g--", linewidth=2, label="Prob. Flow ODE")
        axes[1].set_title("Cluster Separation (lower = more condensed)")
        axes[1].set_xlabel("Time t")
        axes[1].set_ylabel("Mean dist. to nearest cluster center")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Cluster purity
        axes[2].plot(times_rev, self._rev_purity, "r-",  linewidth=2, label="Reverse SDE")
        axes[2].plot(times_ode, self._ode_purity, "g--", linewidth=2, label="Prob. Flow ODE")
        axes[2].set_title(f"Cluster Purity (within r={CAPTURE_RADIUS})")
        axes[2].set_xlabel("Time t")
        axes[2].set_ylabel(f"Fraction of particles near a center")
        axes[2].set_ylim(0, 1.05)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("Diffusion Systems -- Metrics Over Time", fontsize=14)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/diffusion_metrics.png", dpi=150)
        plt.close()

    def _plot_3d_snapshots(self, save_dir: str):
        """3-column grid (initial | mid | final) for each system."""
        systems_order = ["forward_sde", "reverse_sde", "prob_flow_ode"]
        labels = ["Forward SDE", "Reverse SDE", "Prob. Flow ODE"]
        colors = ["steelblue", "tomato", "forestgreen"]

        fig = plt.figure(figsize=(15, 12))
        snap_positions = [0, 2, 4]   # indices into snapshot_indices (initial, mid, final)
        col_titles = ["t = 0 (initial)", "t = 0.5 (mid)", "t = 1.0 (final)"]

        for row, (key, label, color) in enumerate(zip(systems_order, labels, colors)):
            exp = self.runs[key]
            for col, snap_pos in enumerate(snap_positions):
                ax = fig.add_subplot(3, 3, row * 3 + col + 1, projection="3d")
                snap_idx = exp.snapshot_indices[snap_pos]
                pts = exp.get_snapshot(snap_idx)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=color, alpha=0.3, s=4)
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=9)
                ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
                ax.set_zlabel("Z", fontsize=7)
                ax.tick_params(labelsize=6)
                if col == 0:
                    ax.text2D(-0.25, 0.5, label, transform=ax.transAxes,
                              fontsize=9, rotation=90, va="center")

        fig.suptitle("Diffusion Systems -- 3D Particle Snapshots", fontsize=13)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/diffusion_snapshots_3d.png", dpi=150)
        plt.close()

    def _plot_comparison_final(self, save_dir: str):
        """Side-by-side 3D scatter of reverse SDE vs ODE final states with GMM centers."""
        fig = plt.figure(figsize=(12, 5))

        for col, (key, label, color) in enumerate([
            ("reverse_sde",   "Reverse SDE",       "tomato"),
            ("prob_flow_ode", "Prob. Flow ODE",     "forestgreen"),
        ]):
            ax = fig.add_subplot(1, 2, col + 1, projection="3d")
            exp = self.runs[key]
            pts = exp.get_snapshot(-1)   # last frame
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=color, alpha=0.3, s=5, label="Particles")
            # Draw GMM cluster centers as large black stars
            ax.scatter(GMM_CENTERS[:, 0], GMM_CENTERS[:, 1], GMM_CENTERS[:, 2],
                       c="black", marker="*", s=200, zorder=5, label="Cluster centers")
            ax.set_title(f"{label} -- Final State (t=1)", fontsize=10)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.legend(fontsize=8)

        fig.suptitle("Reverse Diffusion: SDE vs. ODE Final Distributions", fontsize=13)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/diffusion_comparison_final.png", dpi=150)
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
            "experiment": self.experiment_name,
            "timestamp": self.timestamp,
            "config": {"num_particles": self.num_particles},
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
    print("Diffusion Systems Experiment")
    print("=" * 60)

    exp = DiffusionExperiment(num_particles=500, experiment_name="diffusion_comparison")

    exp.setup()
    exp.run_all(dt=0.005)
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
