"""
experiments/manifold_multimodal_experiment.py

Multimodal Manifold Learning Experiment -- Stanford Bunny Dataset
=================================================================
Two genuine modalities
  Modality A (geometric) : 3D t-SNE embedding — evolves under TSNEDynamicsSystem,
                           preserving the Bunny's neighbourhood structure.
  Modality B (semantic)  : Bunny part labels — body=0, left_ear=1, right_ear=2.
                           Fixed at generation; completely independent of coords.

Alignment score (Fisher scatter ratio)
  Measures how well the latent encoding of the *current* embedding clusters by
  part label.  Three curves tracked:

    spatial_score(t)  = semantic_alignment_score(embedding_t,        part_labels)
    tracking_score(t) = semantic_alignment_score(latent(embedding_t),part_labels)
    static_score      = semantic_alignment_score(latent(embedding_0), part_labels)
                        (constant reference line)

Scientific hypothesis
  t-SNE preserves local neighbourhood structure, so body / ear points that are
  spatially distinct in Bunny-space should remain clustered in the embedding.
  spatial_score should be maintained or improved, and tracking_score should
  follow it closely if the latent encoder tracks the geometric clustering.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from manifold_flow.systems.manifold import TSNEDynamicsSystem
from manifold_flow.solvers.rk4_solver import RK4Solver
from multimodal_data import (MultimodalAlignment, StanfordBunnyDataset,
                             generate_part_labels, semantic_alignment_score)


# ---------------------------------------------------------------------------
# Bunny data loader (same pattern as all multimodal experiments)
# ---------------------------------------------------------------------------

def load_bunny_data(num_points: int, latent_dim: int, noise_dims: int = 7):
    """
    Generate synthetic Bunny 3D coords and both modalities.

    Returns:
        coords_3d   : (num_points, 3)          — clean Bunny shape
        high_dim    : (num_points, 3+noise_dims)— t-SNE high-dim input
        part_labels : (num_points,) int        — Modality B: 0=body,1=left,2=right
        aligner     : MultimodalAlignment
        static_lat  : (num_points, latent_dim) — latent(coords_t=0), baseline
    """
    body_count   = num_points * 2 // 3
    ear_count    = num_points - body_count
    golden_ratio = (1 + np.sqrt(5)) / 2

    body = []
    for i in range(body_count):
        theta = 2 * np.pi * i / golden_ratio
        phi   = np.arccos(1 - 2 * (i + 0.5) / body_count)
        body.append([np.cos(theta)*np.sin(phi),
                     np.sin(theta)*np.sin(phi),
                     np.cos(phi)])

    ears = []
    per_ear = ear_count // 2
    for x_off in [-0.25, 0.25]:
        for i in range(per_ear):
            t = 2 * np.pi * i / per_ear
            h = 1.0 + 0.6 * (i / per_ear)
            ears.append([x_off + 0.1*np.cos(t), 0.1*np.sin(t), h])

    coords_3d   = np.array((body + ears)[:num_points])
    part_labels = generate_part_labels(num_points)         # Modality B (fixed)

    noise    = np.random.normal(0, 0.1, (num_points, noise_dims))
    high_dim = np.hstack([coords_3d, noise])

    aligner    = MultimodalAlignment(latent_dim=latent_dim, alignment_method="procrustes")
    static_lat = aligner.encode_to_latent(coords_3d.copy())  # baseline

    return coords_3d, high_dim, part_labels, aligner, static_lat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kl_divergence(system: TSNEDynamicsSystem, states_flat: np.ndarray) -> float:
    n = system.n_samples
    Y = states_flat.reshape(n, 3)
    sum_Y = np.sum(np.square(Y), axis=1)
    dist_sq = np.add(np.add(-2 * Y @ Y.T, sum_Y).T, sum_Y)
    np.fill_diagonal(dist_sq, 0.0)
    Q_unnorm = 1.0 / (1.0 + dist_sq)
    np.fill_diagonal(Q_unnorm, 0.0)
    Q = Q_unnorm / np.sum(Q_unnorm)
    Q = np.maximum(Q, 1e-12)
    return float(np.sum(system.P * np.log(system.P / Q + 1e-12)))


def multimodal_scores(trajectory_states, aligner, static_latent,
                      part_labels, n_samples):
    """
    Three alignment curves over the t-SNE trajectory:
      static_scores   : semantic_alignment(latent(embedding_0), labels) — constant
      tracking_scores : semantic_alignment(latent(embedding_t), labels) — evolves
      spatial_scores  : semantic_alignment(embedding_t,         labels) — evolves
    """
    static_val = semantic_alignment_score(static_latent, part_labels)
    tracking_scores, spatial_scores = [], []

    for state in trajectory_states:
        coords    = state.reshape(n_samples, 3)
        track_lat = aligner.encode_to_latent(coords.copy())
        tracking_scores.append(semantic_alignment_score(track_lat, part_labels))
        spatial_scores.append(semantic_alignment_score(coords,     part_labels))

    static_scores = [static_val] * len(trajectory_states)
    return static_scores, tracking_scores, spatial_scores


def pca_project(latent: np.ndarray, n_components: int = 2) -> np.ndarray:
    centered = latent - latent.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:n_components].T


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class ManifoldMultimodalExperiment:
    def __init__(self, num_points=100, latent_dim=16,
                 experiment_name="manifold_multimodal"):
        self.num_points  = num_points
        self.latent_dim  = latent_dim
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result  = None
        self.metrics = {}

    def setup(self):
        print(f"[{self.experiment_name}] Loading Stanford Bunny data "
              f"({self.num_points} pts, latent_dim={self.latent_dim})...")
        self.bunny_coords, high_dim, self.part_labels, self.aligner, self.static_latent = \
            load_bunny_data(self.num_points, self.latent_dim)
        n_body = int((self.part_labels == 0).sum())
        n_lear = int((self.part_labels == 1).sum())
        n_rear = int((self.part_labels == 2).sum())
        print(f"  Bunny 3D range   : [{self.bunny_coords.min():.3f}, {self.bunny_coords.max():.3f}]")
        print(f"  High-dim shape   : {high_dim.shape}  (3D + 7D noise)")
        print(f"  Part labels      : body={n_body}, left_ear={n_lear}, right_ear={n_rear}")

        # Pass Bunny high-dim data to t-SNE so P encodes Bunny neighbourhoods
        self.system = TSNEDynamicsSystem(
            high_dim_data=high_dim, perplexity=15.0, learning_rate=150.0
        )
        self.solver = RK4Solver()
        print("[OK] Setup complete")

    def run(self, dt=0.005):
        print(f"[{self.experiment_name}] Integrating t-SNE ODE (Bunny high-dim input)...")
        y0 = self.system.get_initial_conditions()   # random 3D scatter (standard t-SNE start)
        self.result = self.solver.solve(self.system, y0, (0.0, 3.0), dt)
        print(f"[OK] Trajectory: {self.result.states.shape}")

    def analyze(self):
        print(f"[{self.experiment_name}] Analysing multimodal alignment...")
        states = self.result.states
        times  = self.result.times
        n      = self.system.n_samples

        kl_series = [kl_divergence(self.system, s) for s in states]
        static_scores, tracking_scores, spatial_scores = multimodal_scores(
            states, self.aligner, self.static_latent, self.part_labels, n
        )

        self.metrics.update({
            "kl_initial": kl_series[0],
            "kl_final":   kl_series[-1],
            "kl_reduction": kl_series[0] - kl_series[-1],
            "spatial_alignment_initial":  spatial_scores[0],
            "spatial_alignment_final":    spatial_scores[-1],
            "tracking_alignment_initial": tracking_scores[0],
            "tracking_alignment_final":   tracking_scores[-1],
            "static_alignment_baseline":  static_scores[0],
            "latent_amplification_final": tracking_scores[-1] - spatial_scores[-1],
        })
        self._times   = times
        self._kl      = kl_series
        self._static  = static_scores
        self._track   = tracking_scores
        self._spatial = spatial_scores

        print(f"  KL divergence    : {kl_series[0]:.4f} -> {kl_series[-1]:.4f}")
        print(f"  Spatial  part alignment  : {spatial_scores[0]:.4f} -> {spatial_scores[-1]:.4f}")
        print(f"  Tracking latent alignment: {tracking_scores[0]:.4f} -> {tracking_scores[-1]:.4f}")
        print(f"  Static latent (baseline) : {static_scores[0]:.4f}  (constant)")
        print(f"  Latent amplification     : {self.metrics['latent_amplification_final']:+.4f}")

    def visualize(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[{self.experiment_name}] Generating visualizations...")

        # Color by part label: 0=body, 1=left_ear, 2=right_ear
        part_colors = self.part_labels.astype(float)

        # --- Metrics ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(self._times, self._kl, "b-", linewidth=2)
        ax1.set_title("t-SNE KL Divergence (Bunny neighbourhood structure)")
        ax1.set_xlabel("Time t"); ax1.set_ylabel("KL (lower = better Bunny layout)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(self._times, self._spatial, "b-",  linewidth=2, label="Spatial (3D embedding)")
        ax2.plot(self._times, self._track,   "g-",  linewidth=2, label="Tracking latent")
        ax2.axhline(self._static[0], color="r", linestyle="--", linewidth=1.5,
                    label=f"Static latent baseline ({self._static[0]:.3f})")
        ax2.set_title("Part-label Alignment: Spatial vs Latent\n"
                      "(Fisher scatter ratio — higher = parts better separated)")
        ax2.set_xlabel("Time t"); ax2.set_ylabel("Semantic alignment score [0,1]")
        ax2.set_ylim(0, 1.05); ax2.legend(); ax2.grid(True, alpha=0.3)

        fig.suptitle("Manifold Multimodal -- Stanford Bunny Dataset\n"
                     "Modality A: t-SNE embedding  |  Modality B: part labels (body/ear)",
                     fontsize=11)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/manifold_multimodal_metrics.png", dpi=150)
        plt.close()

        # --- Bunny 3D reference + t-SNE embedding + latent PCA ---
        states = self.result.states
        n = self.system.n_samples
        fig = plt.figure(figsize=(18, 10))

        # Row 1: t-SNE embedding at initial, mid, and final
        for col, (frac, label) in enumerate([
            (0.0,  "t-SNE t=0 (scatter)"),
            (0.5,  "t-SNE t=1.5 (forming)"),
            (1.0,  "t-SNE t=3.0 (layout)"),
        ]):
            idx   = int(frac * (len(states) - 1))
            pts   = states[idx].reshape(n, 3)
            score = semantic_alignment_score(pts, self.part_labels)
            ax    = fig.add_subplot(2, 3, col + 1, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=part_colors, cmap="tab10", vmin=0, vmax=2, alpha=0.8, s=20)
            ax.set_title(f"{label}\nspatial_align={score:.3f}", fontsize=9)
            ax.tick_params(labelsize=6)

        # Row 2: Bunny 3D reference, latent PCA (initial), latent PCA (final)
        ref_score = semantic_alignment_score(self.bunny_coords, self.part_labels)
        ax4 = fig.add_subplot(2, 3, 4, projection="3d")
        ax4.scatter(self.bunny_coords[:, 0], self.bunny_coords[:, 1],
                    self.bunny_coords[:, 2],
                    c=part_colors, cmap="tab10", vmin=0, vmax=2, alpha=0.8, s=20)
        ax4.set_title(f"Bunny 3D Reference\nspatial_align={ref_score:.3f}", fontsize=9)
        ax4.tick_params(labelsize=6)

        for col, (idx, label) in enumerate([(0, "Latent PCA t=0"),
                                             (-1, "Latent PCA t=3")]):
            coords    = states[idx].reshape(n, 3)
            lat       = self.aligner.encode_to_latent(coords.copy())
            pca       = pca_project(lat, 2)
            lat_score = semantic_alignment_score(lat, self.part_labels)
            ax        = fig.add_subplot(2, 3, col + 5)
            ax.scatter(pca[:, 0], pca[:, 1],
                       c=part_colors, cmap="tab10", vmin=0, vmax=2, alpha=0.8, s=20)
            ax.set_title(f"{label}\nlatent_align={lat_score:.3f}", fontsize=9)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.grid(True, alpha=0.3)

        fig.suptitle("Bunny t-SNE Layout + Latent Space\n"
                     "(color = part label: blue=body, orange=left_ear, green=right_ear)",
                     fontsize=11)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/manifold_multimodal_layout.png", dpi=150)
        plt.close()

        print(f"[OK] Visualizations saved to {save_dir}/")

    def save_results(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        out = {
            "experiment": self.experiment_name,
            "timestamp":  self.timestamp,
            "config": {"num_points": self.num_points, "latent_dim": self.latent_dim,
                       "dataset": "StanfordBunny (synthetic, 3D+7D noise)",
                       "modality_a": "3D t-SNE embedding",
                       "modality_b": "part labels: body=0, left_ear=1, right_ear=2",
                       "alignment_metric": "Fisher between-class scatter ratio"},
            "metrics": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in self.metrics.items()},
        }
        path = f"{save_dir}/{self.experiment_name}_{self.timestamp}_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] Results saved: {path}")


def main():
    print("=" * 60)
    print("Manifold Multimodal Experiment -- Stanford Bunny Dataset")
    print("=" * 60)
    exp = ManifoldMultimodalExperiment(num_points=100, latent_dim=16)
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
