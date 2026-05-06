"""
experiments/ode_unet_multimodal_experiment.py

NeuralODE & U-Net Multimodal Experiment -- Stanford Bunny Dataset
==================================================================
Two modalities, two systems:

  Modality A (geometric) : 3D point cloud — Stanford Bunny synthetic dataset.
                           Evolved under two systems in separate runs:
                             · StandardNeuralODE  (neutral-orbit attractor)
                             · UNetDynamicsSystem  (hierarchical skip-coupled attractor)
  Modality B (semantic)  : Bunny part labels — body=0, left_ear=1, right_ear=2.
                           Fixed at generation; independent of coords.

Initial conditions
  Both systems start from the Bunny point cloud (3D coords normalised to [-1,1]).
  NeuralODE: particles start at Bunny coords (first 3 of 10 dims, rest zero).
  U-Net    : encoder starts at Bunny coords; decoder starts at zeros.
             project_to_3d returns the decoder state — so we track whether
             the decoder output retains semantic part structure.

Alignment score (Fisher scatter ratio)
  spatial_score(t)   = semantic_alignment(coords_t, part_labels)
  tracking_score(t)  = semantic_alignment(latent(coords_t), part_labels)
  static_score       = semantic_alignment(latent(coords_0), part_labels)

Scientific hypotheses
  H1 (NeuralODE):
    Orthogonal W + no decay → neutral orbits.  Particles rotate around the
    fixed point without collapsing.  spatial_score should remain roughly
    constant or decay slightly — the system neither clusters nor separates
    parts.  Demonstrates lack of attractor guidance.

  H2 (U-Net):
    The encoder contracts toward its own attractor pattern while the skip
    connection continuously injects encoder features into the decoder.
    If the encoder's attractor RETAINS spatial neighbourhood structure,
    decoder output will preserve (and may amplify) part separation.
    spatial_score should remain comparable to initial or increase slowly —
    demonstrating that hierarchical skip coupling is a *passive memory*
    mechanism, not an active separator.

  Comparison:
    Both systems lack explicit semantic targets (unlike CANDY Diffusion /
    Hopfield with part-centroid attractors).  This negative-control
    experiment anchors the discussion: attractor structure alone (without
    semantic guidance) cannot produce strong part-label alignment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from manifold_flow.systems.neural import StandardNeuralODE, UNetDynamicsSystem
from manifold_flow.solvers.rk4_solver import RK4Solver
from multimodal_data import (MultimodalAlignment, generate_part_labels,
                             semantic_alignment_score)


# ---------------------------------------------------------------------------
# Bunny data loader  (same geometry as other multimodal experiments)
# ---------------------------------------------------------------------------

def load_bunny_data(num_points: int, latent_dim: int):
    """
    Returns:
        coords      : (num_points, 3)   normalised to [-1, 1]
        part_labels : (num_points,) int
        static_lat  : (num_points, latent_dim)
        aligner     : MultimodalAlignment
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

    coords      = np.array((body + ears)[:num_points], dtype=float)
    coords      = coords / (np.abs(coords).max() + 1e-8)
    part_labels = generate_part_labels(num_points)

    aligner    = MultimodalAlignment(latent_dim=latent_dim, alignment_method="procrustes")
    static_lat = aligner.encode_to_latent(coords.copy())

    return coords, part_labels, static_lat, aligner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pca_project(latent: np.ndarray, n_components: int = 2) -> np.ndarray:
    centered = latent - latent.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:n_components].T


def multimodal_scores(trajectory_states, extract_fn, aligner, static_latent,
                      part_labels):
    """
    Compute three alignment curves over a trajectory.

    extract_fn : callable(state_flat) -> (N, 3) array of 3D coords to score
    """
    static_val = semantic_alignment_score(static_latent, part_labels)
    spatial_scores, tracking_scores = [], []

    for state in trajectory_states:
        coords    = extract_fn(state)
        lat       = aligner.encode_to_latent(coords.copy())
        spatial_scores.append(semantic_alignment_score(coords, part_labels))
        tracking_scores.append(semantic_alignment_score(lat,   part_labels))

    static_scores = [static_val] * len(trajectory_states)
    return static_scores, tracking_scores, spatial_scores


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class OdeUnetMultimodalExperiment:
    def __init__(self, num_points=150, latent_dim=16,
                 experiment_name="ode_unet_multimodal"):
        self.num_points      = num_points
        self.latent_dim      = latent_dim
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {}

    def setup(self):
        print(f"[{self.experiment_name}] Loading Stanford Bunny data "
              f"({self.num_points} pts, latent_dim={self.latent_dim})...")
        (self.bunny_coords, self.part_labels,
         self.static_latent, self.aligner) = load_bunny_data(
            self.num_points, self.latent_dim
        )
        print(f"  Coords range: [{self.bunny_coords.min():.3f}, "
              f"{self.bunny_coords.max():.3f}]")
        counts = [(self.part_labels == c).sum() for c in range(3)]
        print(f"  Part labels: body={counts[0]}, left_ear={counts[1]}, "
              f"right_ear={counts[2]}")

        # --- NeuralODE (state_dim=10): first 3 dims = Bunny coords, rest 0 ---
        self.ode_system = StandardNeuralODE(state_dim=10,
                                            num_particles=self.num_points)
        ode_init = np.zeros((self.num_points, 10))
        ode_init[:, :3] = self.bunny_coords
        self.ode_y0 = ode_init.flatten()

        # --- UNet (state_dim=3): encoder = Bunny coords, decoder = 0 ---
        self.unet_system = UNetDynamicsSystem(state_dim=3,
                                              num_particles=self.num_points)
        unet_init = np.concatenate(
            [self.bunny_coords, np.zeros((self.num_points, 3))], axis=1
        )
        self.unet_y0 = unet_init.flatten()

        self.solver = RK4Solver()
        print("[OK] Setup complete")

    def run(self, t_end=1.5, dt=0.02):
        print(f"[{self.experiment_name}] Integrating (t_end={t_end}, dt={dt})...")
        self.ode_result  = self.solver.solve(self.ode_system,  self.ode_y0,
                                             (0.0, t_end), dt)
        self.unet_result = self.solver.solve(self.unet_system, self.unet_y0,
                                             (0.0, t_end), dt)
        print(f"  NeuralODE  trajectory: {self.ode_result.states.shape}")
        print(f"  U-Net      trajectory: {self.unet_result.states.shape}")
        print("[OK] Done")

    def analyze(self):
        print(f"[{self.experiment_name}] Analysing multimodal alignment...")

        # NeuralODE: extract first 3 dims of 10-dim state
        def ode_extract(s):
            return s.reshape(self.num_points, 10)[:, :3]

        # U-Net: extract decoder state (dims 3-5 of 6-dim state)
        def unet_extract(s):
            return s.reshape(self.num_points, 6)[:, 3:]

        ode_static,  ode_track,  ode_spatial = multimodal_scores(
            self.ode_result.states, ode_extract,
            self.aligner, self.static_latent, self.part_labels
        )
        unet_static, unet_track, unet_spatial = multimodal_scores(
            self.unet_result.states, unet_extract,
            self.aligner, self.static_latent, self.part_labels
        )

        self.metrics.update({
            # NeuralODE
            "ode_spatial_initial":  ode_spatial[0],
            "ode_spatial_final":    ode_spatial[-1],
            "ode_tracking_initial": ode_track[0],
            "ode_tracking_final":   ode_track[-1],
            "ode_static_baseline":  ode_static[0],
            # U-Net
            "unet_spatial_initial":  unet_spatial[0],
            "unet_spatial_final":    unet_spatial[-1],
            "unet_tracking_initial": unet_track[0],
            "unet_tracking_final":   unet_track[-1],
            "unet_static_baseline":  unet_static[0],
        })

        self._ode_times   = self.ode_result.times
        self._unet_times  = self.unet_result.times
        self._ode_static  = ode_static
        self._ode_track   = ode_track
        self._ode_spatial = ode_spatial
        self._unet_static  = unet_static
        self._unet_track   = unet_track
        self._unet_spatial = unet_spatial

        print(f"\n  --- NeuralODE (neutral orbits) ---")
        print(f"    spatial_score   : {ode_spatial[0]:.4f} -> {ode_spatial[-1]:.4f}")
        print(f"    tracking_score  : {ode_track[0]:.4f} -> {ode_track[-1]:.4f}")
        print(f"    static baseline : {ode_static[0]:.4f}")
        print(f"\n  --- U-Net (hierarchical attractor) ---")
        print(f"    spatial_score   : {unet_spatial[0]:.4f} -> {unet_spatial[-1]:.4f}")
        print(f"    tracking_score  : {unet_track[0]:.4f} -> {unet_track[-1]:.4f}")
        print(f"    static baseline : {unet_static[0]:.4f}")

    def visualize(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[{self.experiment_name}] Generating visualizations...")

        # ── Plot 1: alignment score curves (side-by-side) ──────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = {"spatial": "royalblue", "tracking": "seagreen", "static": "crimson"}

        for ax, (times, spatial, track, static, title) in zip(axes, [
            (self._ode_times,  self._ode_spatial,  self._ode_track,
             self._ode_static,  "StandardNeuralODE (neutral orbits)"),
            (self._unet_times, self._unet_spatial, self._unet_track,
             self._unet_static, "UNetDynamicsSystem (hierarchical attractor)"),
        ]):
            ax.plot(times, spatial, color=colors["spatial"],  linewidth=2,
                    label="Spatial (3D state)")
            ax.plot(times, track,   color=colors["tracking"], linewidth=2,
                    label="Tracking latent")
            ax.axhline(static[0], color=colors["static"], linestyle="--",
                       linewidth=1.5,
                       label=f"Static baseline ({static[0]:.3f})")
            ax.set_title(f"Part-label Alignment\n{title}", fontsize=10)
            ax.set_xlabel("Time t")
            ax.set_ylabel("Semantic alignment score (Fisher ratio)")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("ODE vs U-Net Multimodal — Stanford Bunny\n"
                     "Negative control: no semantic targets → alignment comparison",
                     fontsize=11)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/ode_unet_multimodal_metrics.png", dpi=150)
        plt.close()

        # ── Plot 2: 3D state snapshots + latent PCA (initial vs final) ────
        fig = plt.figure(figsize=(20, 10))
        part_colors = self.part_labels.astype(float)
        snap_pairs = [
            # (result,         extract_fn,                           label_prefix)
            (self.ode_result,
             lambda s: s.reshape(self.num_points, 10)[:, :3],
             "NeuralODE"),
            (self.unet_result,
             lambda s: s.reshape(self.num_points, 6)[:, 3:],
             "U-Net Decoder"),
        ]

        for row, (result, extract_fn, prefix) in enumerate(snap_pairs):
            for col, (idx, lbl) in enumerate([(0, "t=0 (Bunny init)"),
                                              (-1, "t=end")]):
                pts      = extract_fn(result.states[idx])
                sp_score = semantic_alignment_score(pts, self.part_labels)

                ax3d = fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d")
                ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                             c=part_colors, cmap="tab10", vmin=0, vmax=2,
                             alpha=0.5, s=18)
                ax3d.set_title(f"{prefix}\n{lbl}  align={sp_score:.3f}", fontsize=8)
                ax3d.tick_params(labelsize=6)

                lat      = self.aligner.encode_to_latent(pts.copy())
                pca      = pca_project(lat, 2)
                lt_score = semantic_alignment_score(lat, self.part_labels)

                ax2d = fig.add_subplot(2, 4, row * 4 + col + 3)
                ax2d.scatter(pca[:, 0], pca[:, 1],
                             c=part_colors, cmap="tab10", vmin=0, vmax=2,
                             alpha=0.5, s=18)
                ax2d.set_title(f"Latent PCA — {prefix}\n{lbl}  lat={lt_score:.3f}",
                               fontsize=8)
                ax2d.set_xlabel("PC1"); ax2d.set_ylabel("PC2")
                ax2d.grid(True, alpha=0.3)

        fig.suptitle("ODE vs U-Net Multimodal — 3D State & Latent PCA\n"
                     "(color = part label: body / left_ear / right_ear)",
                     fontsize=11)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/ode_unet_multimodal_snapshots.png", dpi=150)
        plt.close()

        print(f"[OK] Visualizations saved to {save_dir}/")

    def save_results(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        out = {
            "experiment": self.experiment_name,
            "timestamp":  self.timestamp,
            "config": {
                "num_points":  self.num_points,
                "latent_dim":  self.latent_dim,
                "dataset":     "StanfordBunny (synthetic)",
                "modality_a":  "3D neural state (NeuralODE first-3-dims / UNet decoder)",
                "modality_b":  "part labels: body=0, left_ear=1, right_ear=2",
                "metric":      "Fisher between-class scatter ratio",
                "hypothesis":  "Negative control: no semantic targets",
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
    print("NeuralODE & U-Net Multimodal Experiment -- Stanford Bunny")
    print("=" * 60)
    exp = OdeUnetMultimodalExperiment(num_points=150, latent_dim=16)
    exp.setup()
    exp.run(t_end=1.5, dt=0.02)
    exp.analyze()
    FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    exp.visualize(save_dir=FIGURES_DIR)
    exp.save_results(save_dir=FIGURES_DIR)
    print("\n[OK] Done.")
    return exp


if __name__ == "__main__":
    exp = main()
