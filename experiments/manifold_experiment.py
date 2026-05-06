"""
experiments/manifold_experiment.py

Pure Manifold Learning Experiment
Runs two manifold-learning dynamical systems and measures their quality:

  1. TSNEDynamicsSystem  -- 100 points from a 10D Swiss Roll, reduce to 3D
  2. UMAPDynamicsSystem  -- 100 points in two 10D clusters, reduce to 3D

Both systems are deterministic ODEs; integrated with RK4.

Metrics:
  - t-SNE  : approximate KL divergence KL(P || Q) over time (lower = better layout)
  - UMAP   : inter-cluster distance vs intra-cluster std (higher ratio = better)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from manifold_flow.systems.manifold import TSNEDynamicsSystem, UMAPDynamicsSystem
from manifold_flow.solvers.rk4_solver import RK4Solver


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def kl_divergence_tsne(system: TSNEDynamicsSystem, states_flat: np.ndarray) -> float:
    """
    Approximate KL(P || Q) from the current 3D embedding.
    Uses the same P matrix stored in the system and recomputes Q.
    """
    n = system.n_samples
    Y = states_flat.reshape(n, 3)

    sum_Y = np.sum(np.square(Y), axis=1)
    dist_sq = np.add(np.add(-2 * Y @ Y.T, sum_Y).T, sum_Y)
    np.fill_diagonal(dist_sq, 0.0)

    Q_unnorm = 1.0 / (1.0 + dist_sq)
    np.fill_diagonal(Q_unnorm, 0.0)
    Q = Q_unnorm / np.sum(Q_unnorm)
    Q = np.maximum(Q, 1e-12)

    P = system.P
    kl = float(np.sum(P * np.log(P / Q + 1e-12)))
    return kl


def umap_cluster_quality(states_flat: np.ndarray, n_samples: int) -> dict:
    """
    Computes cluster separation for the two-cluster UMAP dataset.
    Assumes first half of points belong to cluster 0, second half to cluster 1.
    """
    Y = states_flat.reshape(n_samples, 3)
    half = n_samples // 2
    c0, c1 = Y[:half], Y[half:]

    inter = float(np.linalg.norm(c0.mean(axis=0) - c1.mean(axis=0)))
    intra = float(0.5 * (np.std(c0) + np.std(c1)))
    ratio = inter / max(intra, 1e-8)
    return {"inter": inter, "intra": intra, "ratio": ratio}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class ManifoldExperiment:
    def __init__(self, experiment_name="manifold_experiment"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tsne_system  = None
        self.umap_system  = None
        self.tsne_result  = None
        self.umap_result  = None
        self.metrics = {}

    def setup(self):
        print(f"[{self.experiment_name}] Setting up manifold systems...")
        self.tsne_system = TSNEDynamicsSystem(perplexity=15.0, learning_rate=150.0)
        self.umap_system = UMAPDynamicsSystem()
        print(f"  t-SNE: {self.tsne_system.n_samples} points, "
              f"{self.tsne_system.high_dim_data.shape[1]}D -> 3D")
        print(f"  UMAP : {self.umap_system.n_samples} points, "
              f"{self.umap_system.high_dim_data.shape[1]}D -> 3D")
        print("[OK] Setup complete")

    def run(self, dt=0.005):
        solver = RK4Solver()
        print(f"[{self.experiment_name}] Integrating t-SNE (dt={dt})...")
        y0_tsne = self.tsne_system.get_initial_conditions()
        self.tsne_result = solver.solve(self.tsne_system, y0_tsne, (0.0, 3.0), dt)
        print(f"  t-SNE trajectory: {self.tsne_result.states.shape}")

        print(f"[{self.experiment_name}] Integrating UMAP (dt={dt})...")
        y0_umap = self.umap_system.get_initial_conditions()
        self.umap_result = solver.solve(self.umap_system, y0_umap, (0.0, 3.0), dt)
        print(f"  UMAP  trajectory: {self.umap_result.states.shape}")
        print("[OK] Integration complete")

    def analyze(self):
        print(f"[{self.experiment_name}] Analysing results...")

        # t-SNE: KL divergence over time
        kl_series = [kl_divergence_tsne(self.tsne_system, s)
                     for s in self.tsne_result.states]

        # UMAP: cluster quality over time
        n = self.umap_system.n_samples
        umap_ratio = [umap_cluster_quality(s, n)["ratio"]
                      for s in self.umap_result.states]

        self.metrics.update({
            "tsne_kl_initial": kl_series[0],
            "tsne_kl_final":   kl_series[-1],
            "tsne_kl_reduction": kl_series[0] - kl_series[-1],
            "umap_cluster_ratio_initial": umap_ratio[0],
            "umap_cluster_ratio_final":   umap_ratio[-1],
        })

        self._tsne_times = self.tsne_result.times
        self._umap_times = self.umap_result.times
        self._kl_series  = kl_series
        self._umap_ratio = umap_ratio

        print(f"\n  --- t-SNE ---")
        print(f"    KL divergence  : {kl_series[0]:.4f} -> {kl_series[-1]:.4f}  "
              f"(reduction: {self.metrics['tsne_kl_reduction']:.4f})")
        print(f"\n  --- UMAP ---")
        print(f"    Cluster ratio  : {umap_ratio[0]:.4f} -> {umap_ratio[-1]:.4f}")

    def visualize(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[{self.experiment_name}] Generating visualizations...")

        # --- Plot 1: metrics ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(self._tsne_times, self._kl_series, "b-", linewidth=2)
        ax1.set_title("t-SNE: KL Divergence KL(P||Q)")
        ax1.set_xlabel("Time t"); ax1.set_ylabel("KL divergence (lower = better)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(self._umap_times, self._umap_ratio, "r-", linewidth=2)
        ax2.set_title("UMAP: Inter/Intra Cluster Ratio (higher = better)")
        ax2.set_xlabel("Time t"); ax2.set_ylabel("Separation ratio")
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Manifold Learning Systems -- Metrics", fontsize=13)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/manifold_metrics.png", dpi=150)
        plt.close()

        # --- Plot 2: 3D embedding evolution ---
        fig = plt.figure(figsize=(18, 8))
        snap_frac = [0.0, 0.5, 1.0]

        for row, (result, system, label, color) in enumerate([
            (self.tsne_result, self.tsne_system, "t-SNE (Swiss Roll)", "steelblue"),
            (self.umap_result, self.umap_system, "UMAP (Two Clusters)", "tomato"),
        ]):
            states = result.states
            n_pts  = system.n_samples
            n_steps = len(states)

            # Color by original index (shows manifold structure)
            original_color = np.linspace(0, 1, n_pts)

            for col, frac in enumerate(snap_frac):
                idx = int(frac * (n_steps - 1))
                ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
                pts = states[idx].reshape(n_pts, 3)
                t_val = result.times[idx]

                if row == 0:
                    # t-SNE: color by Swiss Roll parameter (original ordering)
                    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                                    c=original_color, cmap="plasma", alpha=0.7, s=20)
                else:
                    # UMAP: color by cluster membership
                    half = n_pts // 2
                    c = np.zeros(n_pts)
                    c[half:] = 1.0
                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                               c=c, cmap="bwr", alpha=0.7, s=20)

                ax.set_title(f"{label}\nt={t_val:.2f}", fontsize=8)
                ax.tick_params(labelsize=6)

        fig.suptitle("Manifold Learning -- 3D Embedding Evolution\n"
                     "(t-SNE: color=Swiss Roll position, UMAP: color=cluster)", fontsize=11)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/manifold_snapshots.png", dpi=150)
        plt.close()

        print(f"[OK] Visualizations saved to {save_dir}/")

    def save_results(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        out = {
            "experiment": self.experiment_name,
            "timestamp": self.timestamp,
            "config": {
                "tsne_n_samples": self.tsne_system.n_samples,
                "umap_n_samples": self.umap_system.n_samples,
            },
            "metrics": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in self.metrics.items()},
        }
        path = f"{save_dir}/{self.experiment_name}_{self.timestamp}_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] Results saved: {path}")


def main():
    print("=" * 60)
    print("Manifold Learning Experiment")
    print("=" * 60)
    exp = ManifoldExperiment()
    exp.setup()
    exp.run(dt=0.005)
    exp.analyze()
    FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    exp.visualize(save_dir=FIGURES_DIR)
    exp.save_results(save_dir=FIGURES_DIR)
    print("\n[OK] Done.")
    return exp

if __name__ == "__main__":
    exp = main()
