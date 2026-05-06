"""
quick_start_bunny.py

Stanford Bunny multimodal experiment quick-start script.
Run this directly to see a demo using synthetic data — no download required.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from multimodal_data import StanfordBunnyDataset, MultimodalAlignment
from manifold_flow.systems.bunny_modal import (
    MultimodalBunnySystem,
    MultimodalLatentFlow,
)
from manifold_flow.solvers.euler_maruyama import EulerMaruyamaUncorrected


def examples_basic_alignment():
    """Example 1: Basic multimodal alignment."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Multimodal Alignment")
    print("=" * 60)

    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points)
    coords_3d = np.column_stack([np.cos(theta), np.sin(theta), 0.1 * theta / np.pi])

    print(f"Generated {n_points} 3D points, shape: {coords_3d.shape}")

    aligner = MultimodalAlignment(latent_dim=16)

    latent = aligner.encode_to_latent(coords_3d)
    print(f"Encoded to latent space: {latent.shape}, range [{latent.min():.3f}, {latent.max():.3f}]")

    coords_reconstructed = aligner.decode_from_latent(latent)
    print(f"Decoded back to coords: {coords_reconstructed.shape}")

    error = np.linalg.norm(coords_3d - coords_reconstructed) / n_points
    score = aligner.compute_alignment_score(latent, coords_3d)

    print(f"Reconstruction error: {error:.4f}")
    print(f"Alignment score [0,1]: {score:.4f}")


def example_bunny_evolution():
    """Example 2: Bunny multimodal system evolution."""
    print("\n" + "=" * 60)
    print("Example 2: Bunny Multimodal System Evolution")
    print("=" * 60)

    num_points = 50
    latent_dim = 8

    system = MultimodalBunnySystem(
        num_points=num_points, latent_dim=latent_dim, coupling_strength=0.8
    )

    print(f"MultimodalBunnySystem initialized: state_dim={system.state_dim}, "
          f"points={num_points}, latent_dim={latent_dim}")

    solver = EulerMaruyamaUncorrected(dt=0.005)

    y = system.get_initial_conditions()
    t = 0.0
    dt = 0.01
    num_steps = 50

    trajectory = [y.copy()]
    alignment_scores = []

    print(f"Running {num_steps} evolution steps...")

    for step in range(num_steps):
        y = solver.solve_step(system, t, y, dt)
        t += dt
        trajectory.append(y.copy())

        score = system.compute_alignment_score(y)
        alignment_scores.append(score)

        if (step + 1) % 10 == 0:
            print(f"  step {step+1:3d}/{num_steps}, t={t:.3f}, alignment={score:.4f}")

    print(f"Evolution complete — initial: {alignment_scores[0]:.4f}, "
          f"final: {alignment_scores[-1]:.4f}, mean: {np.mean(alignment_scores):.4f}")

    return trajectory, alignment_scores


def example_latent_flow():
    """Example 3: Lightweight latent flow system."""
    print("\n" + "=" * 60)
    print("Example 3: Latent Flow System (Lightweight)")
    print("=" * 60)

    num_points = 50
    latent_dim = 8

    system = MultimodalLatentFlow(num_points=num_points, latent_dim=latent_dim)

    print(f"MultimodalLatentFlow initialized: state_dim={system.state_dim} (latent only)")

    solver = EulerMaruyamaUncorrected(dt=0.01)

    y = system.get_initial_conditions()
    t = 0.0
    dt = 0.02
    num_steps = 50

    print(f"Running {num_steps} evolution steps...")

    trajectory_latent = []
    for step in range(num_steps):
        y = solver.solve_step(system, t, y, dt)
        t += dt
        trajectory_latent.append(y.copy())

        if (step + 1) % 10 == 0:
            norm = np.linalg.norm(y)
            print(f"  step {step+1:3d}/{num_steps}, t={t:.3f}, state_norm={norm:.4f}")

    print("Evolution complete.")
    return trajectory_latent


def example_stanford_bunny_dataset():
    """Example 4: Stanford Bunny dataset info."""
    print("\n" + "=" * 60)
    print("Example 4: Stanford Bunny Dataset Links")
    print("=" * 60)

    dataset = StanfordBunnyDataset()
    links = dataset.download_link()

    print("\nOfficial download links:")
    for source, url in links.items():
        print(f"  {source:15s}: {url}")

    print("\nQuick download commands:")
    print("  mkdir -p bunny_data")
    print("  cd bunny_data")
    print("  wget https://graphics.stanford.edu/data/3Dscanrep/bunny/reconstruction/bun_zipper.ply")
    print("  cd ..")

    print("\nUsage in code:")
    print("  from multimodal_data import StanfordBunnyDataset")
    print("  dataset = StanfordBunnyDataset()")
    print("  coords = dataset.load_from_ply('./bunny_data/bun_zipper.ply')")


def visualize_results(
    trajectory, alignment_scores, save_path="./quick_start_results.png"
):
    """Visualizes evolution results."""
    print(f"\nGenerating visualization...")

    trajectory = np.array(trajectory)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(alignment_scores, "b-", linewidth=2)
    axes[0].set_xlabel("Evolution Step")
    axes[0].set_ylabel("Alignment Score")
    axes[0].set_title("Multimodal Alignment Score over Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    state_norms = [np.linalg.norm(s) for s in trajectory]
    axes[1].plot(state_norms, "r-", linewidth=2)
    axes[1].set_xlabel("Evolution Step")
    axes[1].set_ylabel("State Norm")
    axes[1].set_title("System State Norm over Time")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    try:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    except Exception:
        print("Could not save figure (possibly no display environment).")

    plt.close()


def main():
    """Runs all examples."""
    print("\n")
    print("=" * 60)
    print("  Stanford Bunny Multimodal Experiments - Quick Start")
    print("=" * 60)

    try:
        examples_basic_alignment()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        trajectory, alignment_scores = example_bunny_evolution()
        visualize_results(trajectory, alignment_scores)
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_latent_flow()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_stanford_bunny_dataset()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)

    print("\nNext steps:\n")
    print("1. Download Stanford Bunny dataset (optional):")
    print("   cd bunny_data && wget https://graphics.stanford.edu/data/3Dscanrep/bunny/reconstruction/bun_zipper.ply\n")
    print("2. Run the full experiment suite:")
    print("   cd experiments && python bunny_multimodal_experiment.py\n")
    print("3. Read the docs:")
    print("   cat MULTIMODAL_BUNNY_README.md\n")
    print("Key files:")
    print("   multimodal_data.py                              — data utilities")
    print("   manifold_flow/systems/bunny_modal.py            — dynamical system")
    print("   experiments/bunny_multimodal_experiment.py      — full experiment")
    print("   MULTIMODAL_BUNNY_README.md                      — full docs")


if __name__ == "__main__":
    main()
