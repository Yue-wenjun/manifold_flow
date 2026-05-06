"""
systems/bunny_modal.py

Stanford Bunny multimodal dynamical system.
Handles joint evolution of 3D coordinates and latent-space features.

Integration points:
- Modality A: 3D coordinate space (state_dim = 3N, N = num_points)
- Modality B: Latent-space representation (state_dim = N * latent_dim)
- Joint dynamics: synchronized evolution of both modalities on the manifold
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..core.types import StateVector, Projection3D, ParameterSet, Time
from ..core.base_system import DynamicalSystem, StochasticSystem, DeterministicSystem


@dataclass
class ModalityConfig:
    """Multimodal configuration."""

    num_points: int
    latent_dim: int
    modality_weight: float = 0.5
    alignment_strength: float = 1.0


class MultimodalBunnySystem(StochasticSystem):
    """
    Multimodal Bunny dynamical system.

    State layout:
    - y[0:3N]                    = 3D coordinates (xyz for N points)
    - y[3N:3N+N*latent_dim]      = latent-space features

    Dynamics:
    - 3D coordinates driven by geometric Laplacian and latent-space guidance
    - Latent dimensions constrained by coordinate encoding and internal dynamics
    """

    def __init__(
        self,
        num_points: int = 100,
        latent_dim: int = 16,
        modality_weight: float = 0.5,
        alignment_strength: float = 1.0,
    ):
        """
        Args:
            num_points: number of mesh points
            latent_dim: latent space dimensionality
            modality_weight: cross-modal coupling weight in [0, 1]
            alignment_strength: alignment force magnitude
        """
        self.config = ModalityConfig(
            num_points=num_points,
            latent_dim=latent_dim,
            modality_weight=modality_weight,
            alignment_strength=alignment_strength,
        )

        total_dim = 6 * num_points + num_points * latent_dim

        super().__init__(
            state_dim=total_dim,
            parameters={
                "coord_decay": 0.5,
                "latent_decay": 0.3,
                "coupling_strength": 0.8,
                "noise_scale": 0.01,
                "flow_speed": 1.0,
            },
        )

        self._compute_mesh_laplacian()
        self.alignment_matrix = np.random.randn(3, latent_dim) * 0.1

    def _compute_mesh_laplacian(self):
        """Approximate mesh Laplacian via k-NN adjacency."""
        N = self.config.num_points
        k_neighbors = min(6, N - 1)

        self.laplacian = np.zeros((N, N))
        for i in range(N):
            neighbors = []
            for offset in range(1, k_neighbors + 1):
                neighbors.append((i + offset) % N)
                neighbors.append((i - offset) % N)

            neighbor_list = list(set(neighbors))[:k_neighbors]
            self.laplacian[i, neighbor_list] = -1.0
            self.laplacian[i, i] = len(neighbor_list)

    def get_initial_conditions(self) -> StateVector:
        """Random point cloud initial conditions."""
        n_points = self.config.num_points
        latent_dim = self.config.latent_dim

        coords = np.random.uniform(-1, 1, size=(n_points, 3))
        coords = coords / np.linalg.norm(coords, axis=1, keepdims=True)

        velocities = np.random.normal(0, 0.01, size=(n_points, 3))
        latent = np.random.normal(0, 1, size=(n_points, latent_dim))

        state = np.concatenate(
            [
                coords.flatten(),
                velocities.flatten(),
                latent.flatten(),
            ]
        )

        return state

    def drift(self, t: Time, y: StateVector) -> StateVector:
        """
        Multimodal drift term.

        d/dt [coords; latent] = f_coords + f_latent + f_coupling
        """
        n_points = self.config.num_points
        latent_dim = self.config.latent_dim

        coords = y[0 : 3 * n_points].reshape(n_points, 3)
        velocities = y[3 * n_points : 6 * n_points].reshape(n_points, 3)
        latent = y[6 * n_points : 6 * n_points + n_points * latent_dim].reshape(
            n_points, latent_dim
        )

        coord_decay = self.parameters["coord_decay"]
        coupling_strength = self.parameters["coupling_strength"]

        laplacian_force = (self.laplacian @ coords) * (-0.1)
        latent_guidance = latent @ self.alignment_matrix.T
        latent_force = latent_guidance * coupling_strength

        dcoords = velocities
        dvelocities = laplacian_force + latent_force - coord_decay * velocities

        latent_decay = self.parameters["latent_decay"]

        coords_encoded = (
            coords @ np.linalg.pinv(self.alignment_matrix).T
        )

        reconstruction_force = (coords_encoded - latent) * 0.5
        latent_drift = reconstruction_force - latent_decay * latent

        drifts = np.concatenate(
            [dcoords.flatten(), dvelocities.flatten(), latent_drift.flatten()]
        )

        return drifts

    def diffusion(self, t: Time, y: StateVector) -> np.ndarray:
        """
        Multimodal stochastic diffusion matrix.

        Returns diagonal (state_dim, state_dim) diffusion coefficients.
        """
        noise_scale = self.parameters["noise_scale"]

        n_points = self.config.num_points
        total_dim = self.state_dim

        diffusion = np.eye(total_dim) * noise_scale
        diffusion[0 : 6 * n_points, 0 : 6 * n_points] *= 1.5
        diffusion[6 * n_points :, 6 * n_points :] *= 0.5

        return diffusion

    def project_to_3d(self, state: StateVector) -> Projection3D:
        """Projects state to 3D centroid for visualization."""
        n_points = self.config.num_points
        coords = state[0 : 3 * n_points].reshape(n_points, 3)
        return coords.mean(axis=0)

    def get_modal_decomposition(
        self, state: StateVector
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decomposes state into two modalities.

        Returns:
            coords: (num_points, 3)
            latent: (num_points, latent_dim)
        """
        n_points = self.config.num_points
        latent_dim = self.config.latent_dim

        coords = state[0 : 3 * n_points].reshape(n_points, 3)
        latent = state[6 * n_points : 6 * n_points + n_points * latent_dim].reshape(
            n_points, latent_dim
        )

        return coords, latent

    def compute_alignment_score(self, state: StateVector) -> float:
        """
        Computes cross-modal alignment quality.

        Measures consistency between 3D coordinates and latent features.
        """
        coords, latent = self.get_modal_decomposition(state)

        coords_decoded = latent @ self.alignment_matrix.T
        error = np.linalg.norm(coords - coords_decoded) / len(coords)
        score = np.exp(-error)

        return score

    def update_parameters(self, new_params: Dict[str, Any]):
        self.parameters.update(new_params)


class MultimodalLatentFlow(DeterministicSystem):
    """
    Lightweight system focused on latent-space evolution.
    Used for rapid prototyping and latent dynamics research.
    """

    def __init__(self, num_points: int = 100, latent_dim: int = 16):
        """
        Args:
            num_points: number of points
            latent_dim: latent space dimensionality
        """
        self.num_points = num_points
        self.latent_dim = latent_dim

        super().__init__(
            state_dim=num_points * latent_dim,
            parameters={
                "drift_strength": 1.0,
                "coupling_strength": 0.5,
                "noise_scale": 0.01,
            },
        )

        self.latent_dynamics = np.random.randn(latent_dim, latent_dim) * 0.1

    def get_initial_conditions(self) -> StateVector:
        return np.random.normal(0, 1, size=self.state_dim)

    def drift(self, t: Time, y: StateVector) -> StateVector:
        """Latent-space drift dynamics."""
        latent = y.reshape(self.num_points, self.latent_dim)

        drift_strength = self.parameters["drift_strength"]

        linear_drift = (latent @ self.latent_dynamics.T) * drift_strength
        nonlinear_drift = np.tanh(latent @ self.latent_dynamics.T) * 0.1

        return (linear_drift + nonlinear_drift).flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        """Projects latent state to 3D using the first three components."""
        latent = state.reshape(self.num_points, self.latent_dim)

        if self.latent_dim >= 3:
            return latent[0, :3]
        else:
            proj = np.zeros(3)
            proj[: self.latent_dim] = latent[0, :]
            return proj
