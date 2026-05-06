"""
experiments/candy_multimodal_experiment.py

CANDY Diffusion Multimodal Experiment -- Stanford Bunny Dataset
===============================================================
Two genuine modalities
  Modality A (geometric) : 3D particle positions — evolve under CANDYDiffusionSystem,
                           which combines:
                             · CANDY masking  (feature extraction via Wp)
                             · U-Net fusion   (encoder W_fuse + decoder W_unet)
                             · Graph schedule g(t) that transitions from noisy
                               feature-driven to ground-truth-driven guidance
  Modality B (semantic)  : Bunny part labels — body=0, left_ear=1, right_ear=2.
                           Fixed at generation; completely independent of coords.

CANDY target design
  Unlike the base CANDYDiffusionSystem (which uses 4 random targets), here the
  3 CANDY target classes are set to the **centroid of each Bunny part**:
    target class 0 (body)      → mean position of body points
    target class 1 (left_ear)  → mean position of left-ear points
    target class 2 (right_ear) → mean position of right-ear points
  Each particle's target = centroid of its own part.

  This makes CANDY semantically meaningful: the U-Net + graph-schedule should
  guide body points toward the body cluster, ear points toward their ear cluster.

Alignment score (Fisher scatter ratio)
    spatial_score(t)  = semantic_alignment_score(coords_t,         part_labels)
    tracking_score(t) = semantic_alignment_score(latent(coords_t), part_labels)
    static_score      = semantic_alignment_score(latent(coords_0), part_labels)
                        (constant reference line)

Scientific hypothesis
  As CANDY diffusion converges (g(t) shifts weight toward ground truth),
  spatial_score should rise toward 1 (body/ear points converge to distinct
  regions).  The tracking score reveals whether the latent encoder amplifies
  or suppresses this emerging semantic structure.
  Compare with diffusion_multimodal_experiment where GMM centers ≠ part centroids
  (spatial_score stays flat) — here the targets are aligned, so we expect
  spatial_score to clearly rise.
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
from multimodal_data import (MultimodalAlignment, StanfordBunnyDataset,
                             generate_part_labels, semantic_alignment_score)


# ---------------------------------------------------------------------------
# Bunny data loader
# ---------------------------------------------------------------------------

def load_bunny_data(num_points: int, latent_dim: int, scale: float = 3.5):
    """
    Generate synthetic Stanford Bunny point cloud scaled into the CANDY target
    range (targets sit at ±4 by default, so scale ≈ 3.5 gives a reasonable
    initial spread around the targets).

    Returns:
        coords      : (num_points, 3)          — Modality A initial state
        part_labels : (num_points,) int        — Modality B: 0=body,1=left,2=right
        static_lat  : (num_points, latent_dim) — latent(coords_0), baseline
        aligner     : MultimodalAlignment
        part_centroids : (3, 3)                — centroid per part label
    """
    body_count   = num_points * 2 // 3
    ear_count    = num_points - body_count
    golden_ratio = (1 + np.sqrt(5)) / 2

    body = []
    for i in range(body_count):
        theta = 2 * np.pi * i / golden_ratio
        phi   = np.arccos(1 - 2 * (i + 0.5) / body_count)
        body.append([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)])

    ears = []
    per_ear = ear_count // 2
    for x_off in [-0.25, 0.25]:
        for i in range(per_ear):
            t = 2 * np.pi * i / per_ear
            h = 1.0 + 0.6 * (i / per_ear)
            ears.append([x_off + 0.1 * np.cos(t), 0.1 * np.sin(t), h])

    coords      = np.array((body + ears)[:num_points]) * scale
    part_labels = generate_part_labels(num_points)

    # Compute part centroids — used as CANDY targets
    part_centroids = np.zeros((3, 3))
    for c in range(3):
        mask = part_labels == c
        part_centroids[c] = coords[mask].mean(axis=0)

    aligner    = MultimodalAlignment(latent_dim=latent_dim, alignment_method="procrustes")
    static_lat = aligner.encode_to_latent(coords.copy())

    return coords, part_labels, static_lat, aligner, part_centroids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def convergence_error(coords: np.ndarray, targets: np.ndarray) -> float:
    """Mean distance of each particle to its assigned CANDY target."""
    return float(np.mean(np.linalg.norm(coords - targets, axis=1)))


def multimodal_scores(trajectory_states, aligner, static_latent,
                      part_labels, num_particles):
    """
    Three alignment curves:
      static_scores   : semantic_alignment(latent(coords_0), labels) — constant
      tracking_scores : semantic_alignment(latent(coords_t), labels) — evolves
      spatial_scores  : semantic_alignment(coords_t,         labels) — evolves
    """
    static_val = semantic_alignment_score(static_latent, part_labels)
    tracking_scores, spatial_scores = [], []

    for state in trajectory_states:
        coords    = state.reshape(num_particles, 3)
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

class CANDYMultimodalExperiment:
    def __init__(self, num_particles: int = 100, latent_dim: int = 16,
                 experiment_name: str = "candy_multimodal"):
        self.num_particles   = num_particles
        self.latent_dim      = latent_dim
        self.experiment_name = experiment_name
        self.timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result          = None
        self.metrics         = {}

    # ------------------------------------------------------------------
    def setup(self, candy_scale: float = 1.0, unet_weight: float = 1.0,
              origin_weight: float = 1.5, decay: float = 0.3):
        print(f"[{self.experiment_name}] Loading Stanford Bunny data "
              f"({self.num_particles} pts, latent_dim={self.latent_dim})...")
        (self.bunny_coords, self.part_labels,
         self.static_latent, self.aligner, self.part_centroids) = load_bunny_data(
            self.num_particles, self.latent_dim
        )
        n_body = int((self.part_labels == 0).sum())
        n_lear = int((self.part_labels == 1).sum())
        n_rear = int((self.part_labels == 2).sum())
        print(f"  Bunny coords range: "
              f"[{self.bunny_coords.min():.2f}, {self.bunny_coords.max():.2f}]")
        print(f"  Part labels: body={n_body}, left_ear={n_lear}, right_ear={n_rear}")
        print(f"  CANDY targets (= part centroids):")
        for i, name in enumerate(["body", "left_ear", "right_ear"]):
            print(f"    target[{i}] ({name}): {self.part_centroids[i]}")

        # Build CANDY system with 3 classes
        self.system = CANDYDiffusionSystem(
            state_dim=3, num_particles=self.num_particles, num_classes=3
        )
        self.system.update_parameters({
            "decay":        decay,
            "unet_weight":  unet_weight,
            "origin_weight": origin_weight,
            "candy_scale":  candy_scale,
            "T":            5.0,
        })

        # Override targets: each particle → centroid of its part
        self.particle_targets = self.part_centroids[self.part_labels]  # (N, 3)
        self.system.targets   = self.particle_targets

        self.solver = RK4Solver()
        print("[OK] Setup complete")

    # ------------------------------------------------------------------
    def run(self, t_span: tuple = (0.0, 5.0), dt: float = 0.05):
        print(f"[{self.experiment_name}] Integrating CANDY from Bunny initial state...")
        y0 = self.bunny_coords.flatten().copy()   # Bunny as starting point
        self.result = self.solver.solve(self.system, y0, t_span, dt)
        print(f"[OK] Trajectory: {self.result.states.shape}")

    # ------------------------------------------------------------------
    def analyze(self):
        print(f"[{self.experiment_name}] Analysing multimodal alignment...")
        states = self.result.states
        times  = self.result.times

        # CANDY-specific: convergence error toward part-centroid targets
        conv_errors = [
            convergence_error(s.reshape(self.num_particles, 3), self.particle_targets)
            for s in states
        ]

        static_scores, tracking_scores, spatial_scores = multimodal_scores(
            states, self.aligner, self.static_latent,
            self.part_labels, self.num_particles
        )

        self.metrics.update({
            "conv_error_initial": conv_errors[0],
            "conv_error_final":   conv_errors[-1],
            "conv_error_reduction": conv_errors[0] - conv_errors[-1],
            "spatial_alignment_initial":  spatial_scores[0],
            "spatial_alignment_final":    spatial_scores[-1],
            "tracking_alignment_initial": tracking_scores[0],
            "tracking_alignment_final":   tracking_scores[-1],
            "static_alignment_baseline":  static_scores[0],
            "latent_amplification_final": tracking_scores[-1] - spatial_scores[-1],
        })
        self._times        = times
        self._conv_errors  = conv_errors
        self._static       = static_scores
        self._track        = tracking_scores
        self._spatial      = spatial_scores

        print(f"  CANDY conv error         : {conv_errors[0]:.4f} -> {conv_errors[-1]:.4f}")
        print(f"  Spatial  part alignment  : {spatial_scores[0]:.4f} -> {spatial_scores[-1]:.4f}")
        print(f"  Tracking latent alignment: {tracking_scores[0]:.4f} -> {tracking_scores[-1]:.4f}")
        print(f"  Static latent (baseline) : {static_scores[0]:.4f}  (constant)")
        print(f"  Latent amplification     : {self.metrics['latent_amplification_final']:+.4f}")

    # ------------------------------------------------------------------
    def visualize(self, save_dir: str = "./results"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[{self.experiment_name}] Generating visualizations...")
        part_colors = self.part_labels.astype(float)

        # --- Panel 1: metrics over time ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(self._times, self._conv_errors, "b-", linewidth=2)
        axes[0].set_title("CANDY Convergence Error\n(mean dist to part-centroid target)")
        axes[0].set_xlabel("Time t")
        axes[0].set_ylabel("Mean Euclidean distance")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self._times, self._spatial, "b-",  linewidth=2,
                     label="Spatial (3D coords)")
        axes[1].plot(self._times, self._track,   "g-",  linewidth=2,
                     label="Tracking latent")
        axes[1].axhline(self._static[0], color="r", linestyle="--", linewidth=1.5,
                        label=f"Static latent baseline ({self._static[0]:.3f})")
        axes[1].set_title("Part-label Alignment: Spatial vs Latent\n"
                          "(Fisher scatter ratio — higher = parts better separated)")
        axes[1].set_xlabel("Time t")
        axes[1].set_ylabel("Semantic alignment score [0,1]")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Latent amplification = tracking - spatial
        amp = [t - s for t, s in zip(self._track, self._spatial)]
        axes[2].plot(self._times, amp, "purple", linewidth=2)
        axes[2].axhline(0, color="gray", linestyle="--", linewidth=1)
        axes[2].set_title("Latent Amplification over Time\n"
                          "(tracking − spatial: >0 = latent amplifies part structure)")
        axes[2].set_xlabel("Time t")
        axes[2].set_ylabel("Alignment gap")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("CANDY Diffusion Multimodal -- Stanford Bunny Dataset\n"
                     "Modality A: 3D CANDY state  |  Modality B: part labels (body/ear)",
                     fontsize=11)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/candy_multimodal_metrics.png", dpi=150)
        plt.close()

        # --- Panel 2: 3D snapshots (t=0, t=mid, t=final) ---
        states = self.result.states
        n      = len(states)
        fig    = plt.figure(figsize=(18, 10))

        snap_specs = [
            (0,       "t=0 (Bunny)"),
            (n // 4,  "t=T/4"),
            (n // 2,  "t=T/2"),
            (3*n//4,  "t=3T/4"),
            (n - 1,   "t=T (final)"),
        ]

        for col, (idx, label) in enumerate(snap_specs):
            pts   = states[idx].reshape(self.num_particles, 3)
            score = semantic_alignment_score(pts, self.part_labels)

            ax = fig.add_subplot(2, 5, col + 1, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=part_colors, cmap="tab10", vmin=0, vmax=2, alpha=0.6, s=14)
            # Draw part-centroid targets as stars
            ax.scatter(self.part_centroids[:, 0],
                       self.part_centroids[:, 1],
                       self.part_centroids[:, 2],
                       c="black", marker="*", s=200, zorder=5)
            ax.set_title(f"{label}\nspatial_align={score:.3f}", fontsize=8)
            ax.tick_params(labelsize=5)

        # Row 2: latent PCA snapshots
        for col, (idx, label) in enumerate(snap_specs):
            pts       = states[idx].reshape(self.num_particles, 3)
            lat       = self.aligner.encode_to_latent(pts.copy())
            pca       = pca_project(lat, 2)
            lat_score = semantic_alignment_score(lat, self.part_labels)

            ax = fig.add_subplot(2, 5, col + 6)
            ax.scatter(pca[:, 0], pca[:, 1],
                       c=part_colors, cmap="tab10", vmin=0, vmax=2, alpha=0.6, s=14)
            ax.set_title(f"Latent PCA {label}\nlatent_align={lat_score:.3f}", fontsize=8)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.grid(True, alpha=0.3)

        fig.suptitle("CANDY Diffusion: Bunny → Part-centroid Targets\n"
                     "(color = part label: blue=body, orange=left_ear, green=right_ear\n"
                     " ★ = CANDY target centroid per part)",
                     fontsize=10)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/candy_multimodal_snapshots.png", dpi=150)
        plt.close()

        print(f"[OK] Visualizations saved to {save_dir}/")

    # ------------------------------------------------------------------
    def save_results(self, save_dir: str = "./results"):
        os.makedirs(save_dir, exist_ok=True)
        out = {
            "experiment": self.experiment_name,
            "timestamp":  self.timestamp,
            "config": {
                "num_particles": self.num_particles,
                "latent_dim":    self.latent_dim,
                "dataset":       "StanfordBunny (synthetic)",
                "modality_a":    "3D CANDY diffusion state",
                "modality_b":    "part labels: body=0, left_ear=1, right_ear=2",
                "alignment_metric": "Fisher between-class scatter ratio",
                "candy_targets": "set to part centroids (body/left_ear/right_ear)",
                "candy_params":  {k: v for k, v in self.system.parameters.items()},
            },
            "metrics": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in self.metrics.items()},
        }
        path = f"{save_dir}/{self.experiment_name}_{self.timestamp}_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] Results saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CANDY Diffusion Multimodal Experiment -- Stanford Bunny Dataset")
    print("=" * 60)

    exp = CANDYMultimodalExperiment(num_particles=100, latent_dim=16)
    exp.setup(
        candy_scale=1.0,
        unet_weight=1.0,
        origin_weight=1.5,
        decay=0.3,
    )
    exp.run(t_span=(0.0, 5.0), dt=0.05)
    exp.analyze()
    FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    exp.visualize(save_dir=FIGURES_DIR)
    exp.save_results(save_dir=FIGURES_DIR)

    print("\n[OK] Done.")
    return exp


if __name__ == "__main__":
    exp = main()
