"""
multimodal_data.py

Multimodal data processing for manifold-flow experiments.

Two genuine modalities:
  Modality A — 3D coordinates (evolves under ODE dynamics)
  Modality B — part labels: body=0, left_ear=1, right_ear=2
               (fixed at generation time, fully independent of coords)

Alignment score = Fisher-like between-class scatter ratio in latent space,
measuring how well the latent representation of the current 3D state
preserves the semantic part structure.

Stanford Bunny dataset:
  https://graphics.stanford.edu/data/3Dscanrep/
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import json
import os


def generate_part_labels(num_points: int) -> np.ndarray:
    """
    Return part label for each Bunny point: 0=body, 1=left_ear, 2=right_ear.

    The labels match the generation order used in load_bunny_data across all
    multimodal experiments:
      first  2/3  of points → body (golden-angle sphere)
      next   1/6  of points → left ear
      last   1/6  of points → right ear
    """
    body_count = num_points * 2 // 3
    ear_count  = num_points - body_count
    per_ear    = ear_count // 2
    labels = np.zeros(num_points, dtype=int)
    labels[body_count : body_count + per_ear] = 1
    labels[body_count + per_ear :]             = 2
    return labels


def semantic_alignment_score(vectors: np.ndarray, part_labels: np.ndarray) -> float:
    """
    Fisher-like between-class scatter ratio.

    Measures how well part labels cluster in the given vector space
    (works for both 3D coordinates and latent features).

      score = between_class_scatter / total_scatter  ∈ [0, 1]

    0 → parts are completely mixed (no clustering)
    1 → each part forms a perfectly separated cluster
    """
    classes = np.unique(part_labels)
    global_centroid = vectors.mean(axis=0)
    total_scatter = float(np.sum((vectors - global_centroid) ** 2))
    if total_scatter < 1e-12:
        return 0.0
    between_scatter = 0.0
    for c in classes:
        mask = part_labels == c
        class_centroid = vectors[mask].mean(axis=0)
        between_scatter += mask.sum() * float(
            np.sum((class_centroid - global_centroid) ** 2)
        )
    return float(np.clip(between_scatter / total_scatter, 0.0, 1.0))


class MultimodalAlignment:
    """Cross-modal alignment: 3D coordinates <-> latent-space representation."""

    def __init__(self, latent_dim: int = 16, alignment_method: str = "procrustes"):
        """
        Args:
            latent_dim: latent space dimensionality
            alignment_method: alignment strategy ("procrustes", "icp", "spectral")
        """
        self.latent_dim = latent_dim
        self.alignment_method = alignment_method
        self.alignment_matrix = None
        self.mean_coord = None
        self.coord_scale = 1.0
        self.mean_latent = None

    def encode_to_latent(self, coords_3d: np.ndarray) -> np.ndarray:
        """
        Encodes 3D coordinates into latent space (Modality A -> Modality B).
        Uses PCA-style projection followed by a nonlinear mapping.

        Args:
            coords_3d: shape (N, 3)

        Returns:
            shape (N, latent_dim)
        """
        coords_normalized = self._normalize_coords(coords_3d)

        if self.alignment_matrix is None:
            self.alignment_matrix = np.random.randn(3, self.latent_dim) * 0.1

        latent = coords_normalized @ self.alignment_matrix
        latent = np.tanh(latent) * 2.0

        return latent

    def decode_from_latent(self, latent: np.ndarray) -> np.ndarray:
        """
        Decodes latent features back to 3D coordinates (Modality B -> Modality A).

        Args:
            latent: shape (N, latent_dim)

        Returns:
            shape (N, 3)
        """
        if self.alignment_matrix is None:
            raise ValueError("Must encode first to initialize the alignment matrix.")

        latent_unscaled = np.arctanh(np.clip(latent / 2.0, -0.999, 0.999))

        inv_matrix = np.linalg.pinv(self.alignment_matrix)
        coords = latent_unscaled @ inv_matrix

        return self._denormalize_coords(coords)

    def align_procrustes(
        self, coords_source: np.ndarray, coords_target: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Aligns two 3D point sets via Procrustes analysis.

        Args:
            coords_source: shape (N, 3)
            coords_target: shape (N, 3)

        Returns:
            (aligned_coords, residual)
        """
        source_c = coords_source - coords_source.mean(axis=0)
        target_c = coords_target - coords_target.mean(axis=0)

        U, _, Vt = np.linalg.svd(source_c.T @ target_c)
        R = U @ Vt

        aligned = source_c @ R.T
        residual = np.linalg.norm(aligned - target_c) / len(coords_source)

        return aligned + target_c.mean(axis=0), residual

    def align_latent_to_coords(
        self, latent: np.ndarray, coords_target: np.ndarray
    ) -> np.ndarray:
        """
        Aligns latent representation to a target 3D coordinate set.

        Args:
            latent: shape (N, latent_dim)
            coords_target: shape (N, 3)

        Returns:
            shape (N, 3) aligned coordinates
        """
        coords_decoded = self.decode_from_latent(latent)
        aligned_coords, _ = self.align_procrustes(coords_decoded, coords_target)
        return aligned_coords

    def compute_alignment_score(self, latent: np.ndarray,
                                part_labels: np.ndarray) -> float:
        """
        Semantic alignment score [0, 1].

        Measures how well the latent representation clusters by Bunny part label
        (body / left_ear / right_ear) using a Fisher-like between-class scatter
        ratio.  Higher = the latent space has learned the part structure.

        Args:
            latent      : Shape (N, latent_dim) — latent features at time t
            part_labels : Shape (N,) int        — fixed part labels (0/1/2)

        Returns:
            float in [0, 1]
        """
        return semantic_alignment_score(latent, part_labels)

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Normalizes 3D coordinates to zero mean and unit scale."""
        self.mean_coord = coords.mean(axis=0)
        coords_c = coords - self.mean_coord
        self.coord_scale = np.linalg.norm(coords_c) / np.sqrt(len(coords))
        if self.coord_scale < 1e-12:
            self.coord_scale = 1.0
        return coords_c / self.coord_scale

    def _denormalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Reverses coordinate normalization."""
        if self.mean_coord is None:
            return coords
        scale = getattr(self, "coord_scale", 1.0)
        return coords * scale + self.mean_coord


class StanfordBunnyDataset:
    """
    Stanford Bunny multimodal dataset loader.

    Source: https://graphics.stanford.edu/data/3Dscanrep/bunny/
    """

    def __init__(self, data_path: str = "./bunny_data"):
        """
        Args:
            data_path: path to data files
        """
        self.data_path = data_path
        self.coords_3d = None
        self.latent_features = None
        self.metadata = {}

    @staticmethod
    def download_link() -> str:
        """Returns official Stanford Bunny download links."""
        return {
            "main": "https://graphics.stanford.edu/data/3Dscanrep/bunny/",
            "standard": "https://graphics.stanford.edu/data/3Dscanrep/bunny/bunny.tar.gz",
            "high_res": "https://graphics.stanford.edu/data/3Dscanrep/bunny/reconstruction/bun_zipper.ply",
            "manifesto": "https://graphics.stanford.edu/data/3Dscanrep/",
        }

    def load_from_ply(self, ply_path: str) -> np.ndarray:
        """
        Loads 3D coordinates from a PLY file.

        Args:
            ply_path: path to PLY file

        Returns:
            shape (N, 3)
        """
        try:
            import open3d as o3d

            mesh = o3d.io.read_triangle_mesh(ply_path)
            coords = np.asarray(mesh.vertices)
        except ImportError:
            coords = self._parse_ply_simple(ply_path)

        self.coords_3d = coords
        self.metadata["num_points"] = len(coords)
        return coords

    def _parse_ply_simple(self, ply_path: str) -> np.ndarray:
        """Minimal dependency-free PLY parser."""
        coords = []
        header_end = False
        vertex_count = 0

        with open(ply_path, "r") as f:
            for line in f:
                line = line.strip()

                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])

                if line == "end_header":
                    header_end = True
                    break

            if header_end:
                for _ in range(vertex_count):
                    parts = f.readline().strip().split()
                    coords.append([float(parts[0]), float(parts[1]), float(parts[2])])

        return np.array(coords)

    def generate_synthetic_latent(self, latent_dim: int = 16) -> np.ndarray:
        """
        Generates synthetic latent features for demonstration.
        In production, these should come from a pretrained autoencoder or diffusion model.

        Args:
            latent_dim: latent space dimensionality

        Returns:
            shape (N, latent_dim)
        """
        if self.coords_3d is None:
            raise ValueError("Must load 3D coordinates first.")

        aligner = MultimodalAlignment(latent_dim=latent_dim)
        self.latent_features = aligner.encode_to_latent(self.coords_3d)

        noise = np.random.normal(0, 0.05, self.latent_features.shape)
        self.latent_features += noise

        return self.latent_features

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the aligned multimodal data pair (coords, latent)."""
        if self.coords_3d is None or self.latent_features is None:
            raise ValueError("Data not fully loaded.")
        return self.coords_3d, self.latent_features

    def save_multimodal(self, save_path: str):
        """Saves multimodal data as NPZ."""
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.savez(
            save_path,
            coords_3d=self.coords_3d,
            latent_features=self.latent_features,
            metadata=self.metadata,
        )

    def load_multimodal(self, save_path: str):
        """Loads multimodal data from NPZ."""
        data = np.load(save_path, allow_pickle=True)
        self.coords_3d = data["coords_3d"]
        self.latent_features = data["latent_features"]
        self.metadata = dict(data["metadata"].item())


class MultimodalBatch:
    """Batch iterator for multimodal data."""

    def __init__(
        self, coords_3d: np.ndarray, latent_features: np.ndarray, batch_size: int = 32
    ):
        """
        Args:
            coords_3d: shape (N, 3)
            latent_features: shape (N, latent_dim)
            batch_size: samples per batch
        """
        self.coords_3d = coords_3d
        self.latent_features = latent_features
        self.batch_size = batch_size
        self.n_samples = len(coords_3d)
        self.current_idx = 0

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.current_idx >= self.n_samples:
            raise StopIteration

        end_idx = min(self.current_idx + self.batch_size, self.n_samples)
        batch_coords = self.coords_3d[self.current_idx : end_idx]
        batch_latent = self.latent_features[self.current_idx : end_idx]

        self.current_idx = end_idx
        return batch_coords, batch_latent

    def shuffle(self):
        """Randomly shuffles the dataset."""
        idx = np.random.permutation(self.n_samples)
        self.coords_3d = self.coords_3d[idx]
        self.latent_features = self.latent_features[idx]
