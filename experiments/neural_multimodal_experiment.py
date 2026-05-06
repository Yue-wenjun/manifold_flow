"""
experiments/neural_multimodal_experiment.py

Multimodal Neural Experiment -- Stanford Bunny Dataset
=======================================================
Two genuine modalities
  Modality A (geometric) : 3D neural state — evolves under HopfieldNetwork ODE
                           toward 3 stored attractor patterns that are set to
                           the centroid of each Bunny part (body, left_ear,
                           right_ear).
  Modality B (semantic)  : Bunny part labels — body=0, left_ear=1, right_ear=2.
                           Fixed at generation; completely independent of coords.

Alignment score (Fisher scatter ratio)
  Three curves tracked over time:
    spatial_score(t)  = semantic_alignment_score(coords_t[:,:3], part_labels)
    tracking_score(t) = semantic_alignment_score(latent(coords_t[:,:3]), part_labels)
    static_score      = semantic_alignment_score(latent(coords_0[:,:3]), part_labels)
                        (constant reference line)

Scientific hypothesis
  Since the Hopfield attractors are set to part centroids, points should
  converge *by part* → spatial_score should rise toward 1.  The tracking
  latent should follow; the gap reveals whether the latent amplifies the
  emerging part structure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from manifold_flow.systems.neural import HopfieldNetwork
from manifold_flow.solvers.rk4_solver import RK4Solver
from multimodal_data import (MultimodalAlignment, StanfordBunnyDataset,
                             generate_part_labels, semantic_alignment_score)

# Hopfield patterns will be computed from Bunny part centroids in setup().
# Placeholder — overwritten before use.
HOPFIELD_PATTERNS = None
CONVERGENCE_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Bunny data loader (same logic as all multimodal experiments)
# ---------------------------------------------------------------------------

def load_bunny_data(num_points: int, latent_dim: int):
    """
    Generate synthetic Bunny coords and both modalities.

    Returns:
        coords      : (num_points, 3)          — normalised to [-1,1]
        part_labels : (num_points,) int        — Modality B: 0=body,1=left,2=right
        static_lat  : (num_points, latent_dim) — latent(coords_t=0), baseline
        aligner     : MultimodalAlignment
        patterns    : (3, 4)                   — Hopfield attractors = part centroids
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

    coords      = np.array((body + ears)[:num_points])
    coords      = coords / (np.abs(coords).max() + 1e-8)   # normalise to [-1,1]
    part_labels = generate_part_labels(num_points)          # Modality B (fixed)

    # Build Hopfield attractors from part centroids (4D: 3D centroid + 0 for dim 4)
    patterns = np.zeros((3, 4))
    for c in range(3):
        mask = part_labels == c
        patterns[c, :3] = coords[mask].mean(axis=0)

    aligner    = MultimodalAlignment(latent_dim=latent_dim, alignment_method="procrustes")
    static_lat = aligner.encode_to_latent(coords.copy())

    return coords, part_labels, static_lat, aligner, patterns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_3d(states_flat, num_particles):
    """First 3 of the 4 Hopfield neuron dimensions."""
    return states_flat.reshape(num_particles, 4)[:, :3]


def pattern_distance(states_flat, num_particles, patterns):
    Y     = states_flat.reshape(num_particles, 4)
    dists = np.linalg.norm(Y[:, np.newaxis] - patterns[np.newaxis], axis=2)
    return float(np.mean(np.min(dists, axis=1)))


def convergence_rate(states_flat, num_particles, patterns):
    Y     = states_flat.reshape(num_particles, 4)
    dists = np.linalg.norm(Y[:, np.newaxis] - patterns[np.newaxis], axis=2)
    return float(np.mean(np.min(dists, axis=1) < CONVERGENCE_THRESHOLD))


def multimodal_scores(trajectory_states, aligner, static_latent,
                      part_labels, num_particles):
    """
    Three alignment curves over the Hopfield trajectory:
      static_scores   : semantic_alignment(latent(coords_0), labels) — constant
      tracking_scores : semantic_alignment(latent(coords_t), labels) — evolves
      spatial_scores  : semantic_alignment(coords_t[:,:3],   labels) — evolves
    """
    static_val = semantic_alignment_score(static_latent, part_labels)
    tracking_scores, spatial_scores = [], []

    for state in trajectory_states:
        coords    = extract_3d(state, num_particles)
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

class NeuralMultimodalExperiment:
    def __init__(self, num_particles=100, latent_dim=16,
                 experiment_name="neural_multimodal"):
        self.num_particles = num_particles
        self.latent_dim    = latent_dim
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result = None
        self.metrics = {}

    def setup(self):
        print(f"[{self.experiment_name}] Loading Stanford Bunny data "
              f"({self.num_particles} pts, latent_dim={self.latent_dim})...")
        (self.bunny_coords, self.part_labels,
         self.static_latent, self.aligner, self.patterns) = load_bunny_data(
            self.num_particles, self.latent_dim
        )
        n_body = int((self.part_labels == 0).sum())
        n_lear = int((self.part_labels == 1).sum())
        n_rear = int((self.part_labels == 2).sum())
        print(f"  Bunny coords range: "
              f"[{self.bunny_coords.min():.3f}, {self.bunny_coords.max():.3f}]")
        print(f"  Part labels: body={n_body}, left_ear={n_lear}, right_ear={n_rear}")
        print(f"  Hopfield attractors set to part centroids:")
        for i, name in enumerate(["body", "left_ear", "right_ear"]):
            print(f"    pattern[{i}] ({name}): {self.patterns[i, :3]}")

        self.system = HopfieldNetwork(num_neurons=4, num_particles=self.num_particles)
        # Override Hebbian weights with part-centroid patterns so that the
        # dynamics actually target the Bunny part centroids, not the default
        # hardcoded [1,-1,1,-1] orthogonal patterns.
        self.system.weights = (self.patterns.T @ self.patterns) / 3.0
        np.fill_diagonal(self.system.weights, 0.0)
        self.solver = RK4Solver()
        print("[OK] Setup complete")

    def run(self, dt=0.02):
        print(f"[{self.experiment_name}] Integrating Hopfield from Bunny initial state...")
        # Build 4D initial state: Bunny coords for dims 0-2, zero for dim 3
        init_4d = np.concatenate(
            [self.bunny_coords, np.zeros((self.num_particles, 1))], axis=1
        )
        y0 = init_4d.flatten()
        self.result = self.solver.solve(self.system, y0, (0.0, 1.0), dt)
        print(f"[OK] Trajectory: {self.result.states.shape}")

    def analyze(self):
        print(f"[{self.experiment_name}] Analysing multimodal alignment...")
        states = self.result.states
        times  = self.result.times

        pat_dist = [pattern_distance(s, self.num_particles, self.patterns) for s in states]
        conv     = [convergence_rate(s,  self.num_particles, self.patterns) for s in states]
        static_scores, tracking_scores, spatial_scores = multimodal_scores(
            states, self.aligner, self.static_latent,
            self.part_labels, self.num_particles
        )

        self.metrics.update({
            "pattern_dist_initial": pat_dist[0],
            "pattern_dist_final":   pat_dist[-1],
            "convergence_rate_final": conv[-1],
            "spatial_alignment_initial":  spatial_scores[0],
            "spatial_alignment_final":    spatial_scores[-1],
            "tracking_alignment_initial": tracking_scores[0],
            "tracking_alignment_final":   tracking_scores[-1],
            "static_alignment_baseline":  static_scores[0],
            "latent_amplification_final": tracking_scores[-1] - spatial_scores[-1],
        })
        self._times   = times
        self._pat     = pat_dist
        self._conv    = conv
        self._static  = static_scores
        self._track   = tracking_scores
        self._spatial = spatial_scores

        print(f"  Pattern distance         : {pat_dist[0]:.4f} -> {pat_dist[-1]:.4f}")
        print(f"  Convergence rate         : {conv[-1]:.2%}")
        print(f"  Spatial  part alignment  : {spatial_scores[0]:.4f} -> {spatial_scores[-1]:.4f}")
        print(f"  Tracking latent alignment: {tracking_scores[0]:.4f} -> {tracking_scores[-1]:.4f}")
        print(f"  Static latent (baseline) : {static_scores[0]:.4f}  (constant)")
        print(f"  Latent amplification     : {self.metrics['latent_amplification_final']:+.4f}")

    def visualize(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[{self.experiment_name}] Generating visualizations...")

        # --- Metrics ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(self._times, self._pat, "b-", linewidth=2)
        axes[0].set_title("Hopfield: Pattern Distance")
        axes[0].set_xlabel("Time t"); axes[0].set_ylabel("Mean dist to nearest pattern")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self._times, self._conv, "g-", linewidth=2)
        axes[1].set_title("Hopfield: Convergence Rate")
        axes[1].set_xlabel("Time t"); axes[1].set_ylabel("Fraction converged")
        axes[1].set_ylim(0, 1.05); axes[1].grid(True, alpha=0.3)

        axes[2].plot(self._times, self._spatial, "b-",  linewidth=2, label="Spatial (3D coords)")
        axes[2].plot(self._times, self._track,   "g-",  linewidth=2, label="Tracking latent")
        axes[2].axhline(self._static[0], color="r", linestyle="--", linewidth=1.5,
                        label=f"Static latent baseline ({self._static[0]:.3f})")
        axes[2].set_title("Part-label Alignment: Spatial vs Latent\n"
                          "(Fisher scatter ratio — higher = parts better separated)")
        axes[2].set_xlabel("Time t"); axes[2].set_ylabel("Semantic alignment score [0,1]")
        axes[2].set_ylim(0, 1.05); axes[2].legend(); axes[2].grid(True, alpha=0.3)

        fig.suptitle("Neural Multimodal -- Stanford Bunny Dataset\n"
                     "Modality A: 3D neural state  |  Modality B: part labels (body/ear)",
                     fontsize=11)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/neural_multimodal_metrics.png", dpi=150)
        plt.close()

        # --- 3D state + latent PCA (Bunny vs final) ---
        states = self.result.states
        fig = plt.figure(figsize=(16, 8))
        part_colors = self.part_labels.astype(float)  # fixed: body/left_ear/right_ear

        for col, (idx, label) in enumerate([(0, "Initial: Bunny (t=0)"),
                                             (-1, "Final: Attractors (t=1)")]):
            state  = states[idx]
            pts_3d = extract_3d(state, self.num_particles)
            colors = part_colors

            sp_score = semantic_alignment_score(pts_3d, self.part_labels)
            ax1 = fig.add_subplot(2, 2, col + 1, projection="3d")
            ax1.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
                        c=colors, cmap="tab10", vmin=0, vmax=2, alpha=0.4, s=12)
            ax1.scatter(self.patterns[:, 0], self.patterns[:, 1],
                        self.patterns[:, 2],
                        c="black", marker="*", s=200, zorder=5)
            ax1.set_title(f"3D Neural State -- {label}\nspatial_align={sp_score:.3f}",
                          fontsize=9)
            ax1.tick_params(labelsize=6)

            lat       = self.aligner.encode_to_latent(pts_3d.copy())
            pca       = pca_project(lat, 2)
            lat_score = semantic_alignment_score(lat, self.part_labels)
            ax2 = fig.add_subplot(2, 2, col + 3)
            ax2.scatter(pca[:, 0], pca[:, 1], c=colors, cmap="tab10",
                        vmin=0, vmax=2, alpha=0.4, s=12)
            ax2.set_title(f"Latent PCA (2D) -- {label}\nlatent_align={lat_score:.3f}",
                          fontsize=9)
            ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
            ax2.grid(True, alpha=0.3)

        fig.suptitle("Bunny -> Hopfield Attractors (part centroids): 3D State + Latent Space\n"
                     "(color = part label: blue=body, orange=left_ear, green=right_ear)",
                     fontsize=11)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/neural_multimodal_latent.png", dpi=150)
        plt.close()

        print(f"[OK] Visualizations saved to {save_dir}/")

    def save_results(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        out = {
            "experiment": self.experiment_name,
            "timestamp":  self.timestamp,
            "config": {"num_particles": self.num_particles, "latent_dim": self.latent_dim,
                       "dataset": "StanfordBunny (synthetic)",
                       "modality_a": "3D Hopfield neural state",
                       "modality_b": "part labels: body=0, left_ear=1, right_ear=2",
                       "alignment_metric": "Fisher between-class scatter ratio",
                       "attractors": "set to part centroids (body/left_ear/right_ear)"},
            "metrics": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in self.metrics.items()},
        }
        path = f"{save_dir}/{self.experiment_name}_{self.timestamp}_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] Results saved: {path}")


def main():
    print("=" * 60)
    print("Neural Multimodal Experiment -- Stanford Bunny Dataset")
    print("=" * 60)
    exp = NeuralMultimodalExperiment(num_particles=100, latent_dim=16)
    exp.setup()
    exp.run(dt=0.02)
    exp.analyze()
    FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    exp.visualize(save_dir=FIGURES_DIR)
    exp.save_results(save_dir=FIGURES_DIR)
    print("\n[OK] Done.")
    return exp

if __name__ == "__main__":
    exp = main()
