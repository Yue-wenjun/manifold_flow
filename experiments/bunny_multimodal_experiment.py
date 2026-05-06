"""
experiments/bunny_multimodal_experiment.py

Stanford Bunny Multimodal Experiment Framework
Demonstrates joint evolution of 3D coordinates and latent space
within the manifold_flow system.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from datetime import datetime

from multimodal_data import MultimodalAlignment, StanfordBunnyDataset, MultimodalBatch
from manifold_flow.systems.bunny_modal import (
    MultimodalBunnySystem,
    MultimodalLatentFlow,
)
from manifold_flow.solvers.euler_maruyama import EulerMaruyamaUncorrected


class BunnyMultimodalExperiment:
    """
    Multimodal Bunny experiment manager.

    Experiment pipeline:
    1. Load/generate Bunny data (3D coordinates + latent-space features)
    2. Initialize the multimodal dynamical system
    3. Evolve on the manifold and observe synchronization between modalities
    4. Analyse alignment quality and reconstruction error
    5. Visualize results
    """

    def __init__(
        self,
        num_points: int = 100,
        latent_dim: int = 16,
        experiment_name: str = "bunny_multimodal",
    ):
        """
        Initialize the experiment.

        Args:
            num_points: Number of Bunny point-cloud vertices.
            latent_dim: Dimensionality of the latent space.
            experiment_name: Human-readable experiment identifier.
        """
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Data containers
        self.coords_3d = None
        self.latent_features = None
        self.alignment = None

        # Dynamical system and solver
        self.system = None
        self.solver = None

        # Results
        self.trajectories = []
        self.alignment_scores = []
        self.reconstruction_errors = []
        self.metrics = {}

    def setup_dataset(self, mode: str = "synthetic"):
        """
        Set up the dataset.

        Args:
            mode: "synthetic" — generate synthetic data,
                  "real_ply"  — load from a real PLY file.
        """
        print(f"[{self.experiment_name}] Initializing dataset (mode={mode})...")

        if mode == "synthetic":
            self._setup_synthetic_dataset()
        elif mode == "real_ply":
            self._setup_real_ply_dataset()
        else:
            raise ValueError(f"Unknown data mode: {mode}")

        # Initialize the multimodal alignment object
        self.alignment = MultimodalAlignment(
            latent_dim=self.latent_dim, alignment_method="procrustes"
        )

        print(f"[OK] Dataset ready: {len(self.coords_3d)} points")

    def _setup_synthetic_dataset(self):
        """Generate synthetic Bunny-like point-cloud data."""
        print("  Generating synthetic data...")

        # Bunny approximation: spherical body + two elongated ear cylinders
        # Body: distribute points evenly over a unit sphere via spherical coordinates
        body_count = self.num_points * 2 // 3  # ~67 % of points
        ear_count = self.num_points - body_count  # ~33 % split between two ears

        # --- Body (sphere) ---
        body_points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(body_count):
            theta = 2 * np.pi * i / golden_ratio          # azimuth (golden-angle spiral)
            phi = np.arccos(1 - 2 * (i + 0.5) / body_count)  # polar
            x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(phi)
            body_points.append([x, y, z])

        # --- Ears (two thin cylinders rising above the body) ---
        ear_points = []
        per_ear = ear_count // 2
        for ear_idx, x_offset in enumerate([-0.25, 0.25]):
            for i in range(per_ear):
                t = 2 * np.pi * i / per_ear
                r = 0.1                       # thin radius
                h = 1.0 + 0.6 * (i / per_ear)  # height [1.0, 1.6]
                x = x_offset + r * np.cos(t)
                y = r * np.sin(t)
                z = h
                ear_points.append([x, y, z])

        all_points = body_points + ear_points
        self.coords_3d = np.array(all_points[: self.num_points])

        # Generate synthetic latent features, weakly correlated with coordinates
        self.latent_features = np.random.normal(
            0, 0.5, (self.num_points, self.latent_dim)
        )
        for i in range(min(3, self.latent_dim)):
            self.latent_features[:, i] += self.coords_3d[:, min(i, 2)] * 0.3

    def _setup_real_ply_dataset(self):
        """Load the real Stanford Bunny PLY file."""
        print("  Loading PLY file...")

        bunny_dataset = StanfordBunnyDataset()
        dataset_info = bunny_dataset.download_link()

        print("\n  Stanford Bunny dataset links:")
        for source, url in dataset_info.items():
            print(f"    {source:12s}: {url}")

        ply_path = "./bunny_data/bun_zipper.ply"

        if os.path.exists(ply_path):
            print(f"  Loading from local file: {ply_path}")
            self.coords_3d = bunny_dataset.load_from_ply(ply_path)
        else:
            print(f"  [WARN] File not found: {ply_path}")
            print(f"  Please download the Bunny data from:")
            print(f"    {dataset_info['standard']}")
            print(f"  and place it in ./bunny_data/")
            print(f"  Falling back to synthetic data...")
            self._setup_synthetic_dataset()
            return

        # Load or generate latent features
        latent_cache = "./bunny_data/latent_features.npz"
        if os.path.exists(latent_cache):
            print("  Loading pre-computed latent features...")
            data = np.load(latent_cache)
            self.latent_features = data["latent"]
        else:
            print("  Generating synthetic latent features...")
            self.latent_features = bunny_dataset.generate_synthetic_latent(
                self.latent_dim
            )

    def setup_system(self, system_type: str = "full_multimodal"):
        """
        Initialize the dynamical system.

        Args:
            system_type: "full_multimodal" — coupled 3D + latent system,
                         "latent_only"     — lightweight latent-flow only.
        """
        print(f"[{self.experiment_name}] Initializing {system_type} system...")

        if system_type == "full_multimodal":
            self.system = MultimodalBunnySystem(
                num_points=self.num_points,
                latent_dim=self.latent_dim,
                modality_weight=0.5,
                alignment_strength=1.0,
            )
        elif system_type == "latent_only":
            self.system = MultimodalLatentFlow(
                num_points=self.num_points, latent_dim=self.latent_dim
            )
        else:
            raise ValueError(f"Unknown system type: {system_type}")

        self.solver = EulerMaruyamaUncorrected(dt=0.001)

        print(f"[OK] System ready -- state dimension: {self.system.state_dim}")

    def construct_initial_state(
        self, use_data_coords: bool = True, use_data_latent: bool = True
    ) -> np.ndarray:
        """
        Build the initial state vector.

        For MultimodalBunnySystem the layout is:
            y = [coords (3N) | velocities (3N) | latent (N * latent_dim)]
        """
        if use_data_coords and self.coords_3d is not None:
            coords = self.coords_3d.copy()
        else:
            coords = np.random.normal(0, 1, (self.num_points, 3))

        velocities = np.zeros((self.num_points, 3))  # start from rest

        if use_data_latent and self.latent_features is not None:
            latent = self.latent_features.copy()
        else:
            latent = np.random.normal(0, 1, (self.num_points, self.latent_dim))

        state = np.concatenate(
            [coords.flatten(), velocities.flatten(), latent.flatten()]
        )
        return state

    def run_evolution(
        self,
        t_span: Tuple[float, float] = (0.0, 1.0),
        num_steps: int = 100,
        save_interval: int = 10,
    ):
        """
        Integrate the multimodal system forward in time.

        Args:
            t_span: (t_start, t_end) integration interval.
            num_steps: Total number of integration steps.
            save_interval: How often (in steps) to snapshot the state.
        """
        if self.system is None:
            raise ValueError("System not initialized -- call setup_system() first.")

        print(f"[{self.experiment_name}] Running evolution ({num_steps} steps)...")

        y = self.construct_initial_state()
        t = t_span[0]
        dt = (t_span[1] - t_span[0]) / num_steps

        trajectory = [y.copy()]
        time_points = [t]

        for step in range(num_steps):
            y = self.solver.solve_step(self.system, t, y, dt)
            t += dt

            if (step + 1) % save_interval == 0:
                trajectory.append(y.copy())
                time_points.append(t)

                if hasattr(self.system, "compute_alignment_score"):
                    score = self.system.compute_alignment_score(y)
                    self.alignment_scores.append(score)

            if (step + 1) % max(1, num_steps // 10) == 0:
                progress = (step + 1) / num_steps * 100
                print(f"  Progress: {progress:.1f}% (t={t:.3f})")

        self.trajectories.append(np.array(trajectory))
        self.metrics["time_points"] = np.array(time_points)
        self.metrics["trajectory_length"] = len(trajectory)

        print("[OK] Evolution complete")

    def analyze_results(self):
        """Analyse the evolution trajectory."""
        if not self.trajectories:
            print("[WARN] No trajectory data to analyse.")
            return

        if not hasattr(self.system, "alignment_matrix"):
            print("[WARN] System has no alignment_matrix -- skipping reconstruction analysis.")
            return

        print(f"[{self.experiment_name}] Analysing results...")

        trajectory = self.trajectories[0]

        reconstruction_errors = []
        alignment_scores_computed = []

        for state in trajectory:
            coords, latent = self._decompose_state(state)

            # Decode latent features back to coordinate space via alignment matrix
            # alignment_matrix: (3, latent_dim)  →  latent @ A.T : (N, 3)
            coords_decoded = latent @ self.system.alignment_matrix.T
            error = np.linalg.norm(coords - coords_decoded) / len(coords)
            reconstruction_errors.append(error)

            score = np.exp(-error)
            alignment_scores_computed.append(score)

        self.reconstruction_errors = reconstruction_errors
        self.alignment_scores = alignment_scores_computed

        self.metrics.update(
            {
                "mean_reconstruction_error": np.mean(reconstruction_errors),
                "min_reconstruction_error": np.min(reconstruction_errors),
                "max_reconstruction_error": np.max(reconstruction_errors),
                "mean_alignment_score": np.mean(alignment_scores_computed),
                "final_alignment_score": alignment_scores_computed[-1],
            }
        )

        print("[OK] Analysis complete")
        print(f"  Mean reconstruction error : {self.metrics['mean_reconstruction_error']:.4f}")
        print(f"  Final alignment score     : {self.metrics['final_alignment_score']:.4f}")

    def _decompose_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose the flat state vector into (coords, latent) arrays."""
        coords = state[: 3 * self.num_points].reshape(self.num_points, 3)
        latent = state[6 * self.num_points :].reshape(self.num_points, self.latent_dim)
        return coords, latent

    def visualize_results(self, save_dir: str = "./results"):
        """Generate and save result visualizations."""
        os.makedirs(save_dir, exist_ok=True)

        print(f"[{self.experiment_name}] Generating visualizations...")

        # --- 1. Alignment score and reconstruction error over time ---
        if self.alignment_scores:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(self.alignment_scores, "b-", linewidth=2, label="Alignment Score")
            ax1.set_xlabel("Evolution step (snapshot index)")
            ax1.set_ylabel("Alignment score [0, 1]")
            ax1.set_title("Multimodal alignment quality over evolution")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(
                self.reconstruction_errors,
                "r-",
                linewidth=2,
                label="Reconstruction Error",
            )
            ax2.set_xlabel("Evolution step (snapshot index)")
            ax2.set_ylabel("Reconstruction error")
            ax2.set_title("Coordinate–latent reconstruction error")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            fig.tight_layout()
            plt.savefig(f"{save_dir}/alignment_evolution.png", dpi=150)
            plt.close()

        # --- 2. 3-D coordinate evolution ---
        if self.trajectories:
            trajectory = self.trajectories[0]

            fig = plt.figure(figsize=(15, 5))

            # Initial state
            ax1 = fig.add_subplot(1, 3, 1, projection="3d")
            coords_init, _ = self._decompose_state(trajectory[0])
            ax1.scatter(
                coords_init[:, 0],
                coords_init[:, 1],
                coords_init[:, 2],
                c="blue",
                alpha=0.5,
                s=10,
            )
            ax1.set_title("Initial coordinates (t=0)")
            ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

            # Mid state
            ax2 = fig.add_subplot(1, 3, 2, projection="3d")
            if len(trajectory) > 1:
                mid_idx = len(trajectory) // 2
                coords_mid, _ = self._decompose_state(trajectory[mid_idx])
                ax2.scatter(
                    coords_mid[:, 0],
                    coords_mid[:, 1],
                    coords_mid[:, 2],
                    c="green",
                    alpha=0.5,
                    s=10,
                )
                ax2.set_title(f"Mid state ({mid_idx}/{len(trajectory)-1} snapshots)")
            else:
                ax2.set_title("Mid state (unavailable)")
            ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

            # Final state
            ax3 = fig.add_subplot(1, 3, 3, projection="3d")
            coords_final, _ = self._decompose_state(trajectory[-1])
            ax3.scatter(
                coords_final[:, 0],
                coords_final[:, 1],
                coords_final[:, 2],
                c="red",
                alpha=0.5,
                s=10,
            )
            ax3.set_title("Final coordinates (t=1)")
            ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")

            fig.tight_layout()
            plt.savefig(f"{save_dir}/coords_evolution_3d.png", dpi=150)
            plt.close()

        print(f"[OK] Visualizations saved to {save_dir}/")

    def save_results(self, save_dir: str = "./results"):
        """Save experiment results and metadata as JSON."""
        os.makedirs(save_dir, exist_ok=True)

        result_file = (
            f"{save_dir}/{self.experiment_name}_{self.timestamp}_results.json"
        )

        metrics_serializable = {}
        for k, v in self.metrics.items():
            if isinstance(v, np.ndarray):
                metrics_serializable[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                metrics_serializable[k] = v.item()
            else:
                metrics_serializable[k] = v

        results = {
            "experiment": self.experiment_name,
            "timestamp": self.timestamp,
            "config": {"num_points": self.num_points, "latent_dim": self.latent_dim},
            "metrics": metrics_serializable,
        }

        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[OK] Results saved: {result_file}")


def main():
    """Main experiment pipeline."""

    print("=" * 60)
    print("Stanford Bunny Multimodal Experiment Framework")
    print("=" * 60)

    experiment = BunnyMultimodalExperiment(
        num_points=100, latent_dim=16, experiment_name="bunny_modal_v1"
    )

    # 1. Load data
    experiment.setup_dataset(mode="synthetic")

    # 2. Initialize system
    experiment.setup_system(system_type="full_multimodal")

    # 3. Run evolution
    experiment.run_evolution(t_span=(0.0, 1.0), num_steps=100, save_interval=10)

    # 4. Analyse results
    experiment.analyze_results()

    # 5. Visualize
    experiment.visualize_results(save_dir="./results")

    # 6. Save
    experiment.save_results(save_dir="./results")

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)

    print("\nAcquiring Stanford Bunny Dataset:")
    dataset = StanfordBunnyDataset()
    links = dataset.download_link()
    print(f"  Standard version : {links['standard']}")
    print(f"  High resolution  : {links['high_res']}")

    return experiment


if __name__ == "__main__":
    experiment = main()
