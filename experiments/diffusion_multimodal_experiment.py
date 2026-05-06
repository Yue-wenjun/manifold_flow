"""
experiments/diffusion_multimodal_experiment.py

Multimodal Diffusion Experiment -- Stanford Bunny Dataset
==========================================================
Two genuine modalities
  Modality A (geometric) : 3D particle positions — evolve under ProbabilityFlowODE
                           toward 4 GMM cluster centres.
  Modality B (semantic)  : Bunny part labels — body=0, left_ear=1, right_ear=2.
                           Fixed at generation; completely independent of coords.

Alignment score (Fisher scatter ratio)
  Measures how well the latent encoding of the *current* 3D state clusters by
  part label.  Three curves are tracked over time:

    spatial_score(t)  = semantic_alignment_score(coords_t,          part_labels)
                        → does the raw 3D geometry cluster by part?
    tracking_score(t) = semantic_alignment_score(latent(coords_t),  part_labels)
                        → does the latent space capture part structure?
    static_score      = semantic_alignment_score(latent(coords_0),  part_labels)
                        → baseline: initial latent (constant reference line)

Scientific hypothesis
  As diffusion pulls body / ear points toward different GMM centres, spatial
  part separation should increase.  If the latent encoder tracks this, the
  tracking score rises together.  The gap (tracking − spatial) shows whether
  the latent *amplifies* or *suppresses* geometric part structure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from manifold_flow.systems.diffusion import ProbabilityFlowODE
from manifold_flow.solvers.rk4_solver import RK4Solver
from multimodal_data import (MultimodalAlignment, StanfordBunnyDataset,
                             generate_part_labels, semantic_alignment_score)

GMM_CENTERS = np.array([
    [ 3.0,  3.0,  3.0],
    [-3.0, -3.0,  3.0],
    [-3.0,  3.0, -3.0],
    [ 3.0, -3.0, -3.0],
])
CAPTURE_RADIUS = 2.0


# ---------------------------------------------------------------------------
# Bunny data loader (shared pattern across all multimodal experiments)
# ---------------------------------------------------------------------------

def load_bunny_data(num_points: int, latent_dim: int, scale: float = 2.5):
    """
    Generate synthetic Stanford Bunny point cloud.

    Returns:
        coords     : (num_points, 3)          — Modality A: 3D positions (scaled)
        part_labels: (num_points,) int        — Modality B: 0=body,1=left_ear,2=right_ear
        static_lat : (num_points, latent_dim) — latent(coords_t=0), used as baseline
        aligner    : MultimodalAlignment      — encoder for Modality A → latent
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
    part_labels = generate_part_labels(num_points)            # Modality B (fixed)

    aligner    = MultimodalAlignment(latent_dim=latent_dim, alignment_method="procrustes")
    static_lat = aligner.encode_to_latent(coords.copy())      # baseline latent at t=0

    return coords, part_labels, static_lat, aligner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cluster_purity(coords: np.ndarray) -> float:
    dists = np.linalg.norm(coords[:, np.newaxis] - GMM_CENTERS[np.newaxis], axis=2)
    return float(np.mean(np.min(dists, axis=1) < CAPTURE_RADIUS))


def multimodal_scores(trajectory_states, aligner, static_latent,
                      part_labels, num_particles):
    """
    Compute three alignment curves over the trajectory:
      static_scores   : semantic_alignment_score(latent(coords_0), part_labels) — constant
      tracking_scores : semantic_alignment_score(latent(coords_t), part_labels) — evolves
      spatial_scores  : semantic_alignment_score(coords_t,         part_labels) — evolves
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


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class DiffusionMultimodalExperiment:
    def __init__(self, num_particles=100, latent_dim=16,
                 experiment_name="diffusion_multimodal"):
        self.num_particles = num_particles
        self.latent_dim    = latent_dim
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result = None
        self.metrics = {}

    def setup(self):
        print(f"[{self.experiment_name}] Loading Stanford Bunny data "
              f"({self.num_particles} pts, latent_dim={self.latent_dim})...")
        self.bunny_coords, self.part_labels, self.static_latent, self.aligner = \
            load_bunny_data(self.num_particles, self.latent_dim)
        n_body = int((self.part_labels == 0).sum())
        n_lear = int((self.part_labels == 1).sum())
        n_rear = int((self.part_labels == 2).sum())
        print(f"  Bunny coords range: "
              f"[{self.bunny_coords.min():.2f}, {self.bunny_coords.max():.2f}]")
        print(f"  Part labels: body={n_body}, left_ear={n_lear}, right_ear={n_rear}")

        self.system = ProbabilityFlowODE(state_dim=3, num_particles=self.num_particles)
        self.solver = RK4Solver()
        print("[OK] Setup complete")

    def run(self, dt=0.005):
        print(f"[{self.experiment_name}] Integrating ODE from Bunny initial state...")
        # Use Bunny coords as initial positions instead of system default (random noise)
        y0 = self.bunny_coords.flatten().copy()
        self.result = self.solver.solve(self.system, y0, (0.0, 1.0), dt)
        print(f"[OK] Trajectory: {self.result.states.shape}")

    def analyze(self):
        print(f"[{self.experiment_name}] Analysing multimodal alignment...")
        states = self.result.states
        times  = self.result.times

        purity = [cluster_purity(s.reshape(self.num_particles, 3)) for s in states]
        static_scores, tracking_scores, spatial_scores = multimodal_scores(
            states, self.aligner, self.static_latent,
            self.part_labels, self.num_particles
        )

        self.metrics.update({
            "purity_initial": purity[0],
            "purity_final":   purity[-1],
            "spatial_alignment_initial":  spatial_scores[0],
            "spatial_alignment_final":    spatial_scores[-1],
            "tracking_alignment_initial": tracking_scores[0],
            "tracking_alignment_final":   tracking_scores[-1],
            "static_alignment_baseline":  static_scores[0],
            "latent_amplification_final": tracking_scores[-1] - spatial_scores[-1],
        })
        self._times   = times
        self._purity  = purity
        self._static  = static_scores
        self._track   = tracking_scores
        self._spatial = spatial_scores

        print(f"  3D cluster purity        : {purity[0]:.2%} -> {purity[-1]:.2%}")
        print(f"  Spatial  part alignment  : {spatial_scores[0]:.4f} -> {spatial_scores[-1]:.4f}")
        print(f"  Tracking latent alignment: {tracking_scores[0]:.4f} -> {tracking_scores[-1]:.4f}")
        print(f"  Static latent (baseline) : {static_scores[0]:.4f}  (constant)")
        print(f"  Latent amplification     : {self.metrics['latent_amplification_final']:+.4f}")

    def visualize(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[{self.experiment_name}] Generating visualizations...")

        # --- Metrics ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(self._times, self._purity, "b-", linewidth=2)
        ax1.set_title("3D Cluster Purity (Bunny -> GMM Clusters)")
        ax1.set_xlabel("Time t"); ax1.set_ylabel("Fraction within cluster radius")
        ax1.set_ylim(0, 1.05); ax1.grid(True, alpha=0.3)

        ax2.plot(self._times, self._spatial, "b-",  linewidth=2, label="Spatial (3D coords)")
        ax2.plot(self._times, self._track,   "g-",  linewidth=2, label="Tracking latent")
        ax2.axhline(self._static[0], color="r", linestyle="--", linewidth=1.5,
                    label=f"Static latent baseline ({self._static[0]:.3f})")
        ax2.set_title("Part-label Alignment: Spatial vs Latent\n"
                      "(Fisher scatter ratio — higher = parts better separated)")
        ax2.set_xlabel("Time t"); ax2.set_ylabel("Semantic alignment score [0,1]")
        ax2.set_ylim(0, 1.05); ax2.legend(); ax2.grid(True, alpha=0.3)

        fig.suptitle("Diffusion Multimodal -- Stanford Bunny Dataset\n"
                     "Modality A: 3D positions  |  Modality B: part labels (body/ear)",
                     fontsize=11)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/diffusion_multimodal_metrics.png", dpi=150)
        plt.close()

        # --- 3D snapshots colored by part label ---
        states = self.result.states
        n = len(states)
        snap_idx    = [0, n//4, n//2, 3*n//4, n-1]
        snap_labels = ["t=0\n(Bunny)", "t=0.25", "t=0.5", "t=0.75", "t=1.0\n(Clusters)"]
        part_colors = self.part_labels.astype(float)  # 0/1/2 → body/left/right

        fig = plt.figure(figsize=(20, 4))
        for i, (idx, lbl) in enumerate(zip(snap_idx, snap_labels)):
            ax  = fig.add_subplot(1, 5, i + 1, projection="3d")
            pts = states[idx].reshape(self.num_particles, 3)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=part_colors, cmap="tab10", vmin=0, vmax=2,
                       alpha=0.6, s=12)
            ax.scatter(GMM_CENTERS[:, 0], GMM_CENTERS[:, 1], GMM_CENTERS[:, 2],
                       c="black", marker="*", s=120, zorder=5)
            score = semantic_alignment_score(pts, self.part_labels)
            ax.set_title(f"{lbl}\nalign={score:.2f}", fontsize=8)
            ax.tick_params(labelsize=6)
        fig.suptitle("Bunny Diffusion: particle evolution\n"
                     "(color = part label: blue=body, orange=left_ear, green=right_ear)",
                     fontsize=10)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/diffusion_multimodal_snapshots.png", dpi=150)
        plt.close()

        print(f"[OK] Visualizations saved to {save_dir}/")

    def save_results(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        out = {
            "experiment": self.experiment_name,
            "timestamp":  self.timestamp,
            "config": {"num_particles": self.num_particles, "latent_dim": self.latent_dim,
                       "dataset": "StanfordBunny (synthetic)",
                       "modality_a": "3D particle positions",
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
    print("Diffusion Multimodal Experiment -- Stanford Bunny Dataset")
    print("=" * 60)
    exp = DiffusionMultimodalExperiment(num_particles=100, latent_dim=16)
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
