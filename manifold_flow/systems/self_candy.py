"""
manifold_flow/systems/self_candy.py

Self-Supervised CANDY (CANDY_SSL)
==================================
Unsupervised variant of the CANDY module that learns representations via
cluster-completion pretraining: one K-means cluster of tokens is zeroed out,
the encoder reconstructs it from the remaining clusters.  This mirrors the
LT-PCA-kmean framework (see LT-PCA-kmean 2.0.docx), where K-means defines
the P (allow set) / Z (forbid set) boundary, and the LT projection is trained
to reconstruct the Z set from the P set.

This module is PyTorch-based (not an ODE system) and is used independently
of the Manifold Flow ODE framework.  It provides the experimental basis for
Exp S3: Self-Supervised CANDY (Supplementary).

Architecture:
  Encoder  : CANDY sparse masking (Wp lower-triangular, custom activation φ)
             followed by a cross-feature transform (Wzp)
  Decoder  : two-layer MLP that reconstructs the original token values
  SSL task : Cluster-completion (mask one K-means cluster, reconstruct from
             the remaining clusters — faithful to the LT-PCA-kmean framework)

Contrast with random MAE:
  Random masked autoencoding (MAE-style) has no structural prior: it optimises
  for local surface smoothness rather than cluster-level discrimination.
  Cluster-completion pretraining directly optimises for inter-cluster
  distinguishability, so the learned representations reflect the K-means
  partition — enabling the encoder to surpass PCA on semantic alignment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Soft-threshold activation  φ(x) = (|x+1| - |x-1|) / 2  ∈ [-1, 1]
# ---------------------------------------------------------------------------

class _SoftThreshold(nn.Module):
    """φ(x) = (|x+1| − |x−1|) / 2  —  piece-wise linear clamp to [−1, 1]."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.abs(x + 1.0) - torch.abs(x - 1.0)) / 2.0


# ---------------------------------------------------------------------------
# CANDY_SSL
# ---------------------------------------------------------------------------

class CANDY_SSL(nn.Module):
    """
    Self-supervised CANDY module with cluster-completion pretraining.

    Parameters
    ----------
    seq_len      : int     Number of tokens (particles / points).
    feat_dim     : int     Feature dimension per token.
    num_clusters : int     Number of K-means clusters for the P/Z split.
    mask_ratio   : float   Fallback random-mask ratio (used only when no
                           cluster labels are supplied).
    """

    def __init__(self, seq_len: int, feat_dim: int,
                 num_clusters: int = 3, mask_ratio: float = 0.30):
        super().__init__()
        self.seq_len      = seq_len
        self.feat_dim     = feat_dim
        self.num_clusters = num_clusters
        self.mask_ratio   = mask_ratio

        self.phi = _SoftThreshold()

        # ── Encoder ────────────────────────────────────────────────────────
        # Per-token feature-selection weights (learnable, init via K-means)
        self.p_mask = nn.Parameter(torch.zeros(seq_len, feat_dim))

        # Sparse lower-triangular weight matrix (unit diagonal enforced)
        Wp_init = torch.tril(torch.randn(seq_len, seq_len) * 0.05)
        Wp_init.fill_diagonal_(1.0)
        self.Wp = nn.Parameter(Wp_init)

        # Post-masking projection
        self.enc_proj1 = nn.Sequential(
            _SoftThreshold(),
            nn.Linear(feat_dim, feat_dim),
            _SoftThreshold(),
        )

        # Cross-token transform
        self.Wzp = nn.Parameter(torch.randn(seq_len, seq_len) * 0.1)
        self.enc_proj2 = nn.Sequential(
            _SoftThreshold(),
            nn.Linear(feat_dim, feat_dim),
            _SoftThreshold(),
        )

        # ── Decoder ────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Linear(feat_dim * 4, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    # ── K-means initialisation ──────────────────────────────────────────────

    def init_from_kmeans(self, cluster_labels: np.ndarray):
        """
        Initialise p_mask from K-means cluster labels (no gradient).

        For the token at position i:
          * p_mask[i, :] = +1.0  if point i is the largest cluster (body)
          * p_mask[i, :] = -1.0  otherwise  (smaller clusters = ears)

        This biases the learnable mask toward the dominant P set (body),
        matching the LT-PCA-kmean initialisation strategy.

        Parameters
        ----------
        cluster_labels : (S,) int array  output of KMeans.labels_
        """
        S = self.seq_len
        assert len(cluster_labels) == S

        # Identify the largest cluster
        counts = np.bincount(cluster_labels)
        dominant = int(np.argmax(counts))

        init = np.where(cluster_labels == dominant, 1.0, -1.0)   # (S,)
        init = np.stack([init] * self.feat_dim, axis=1)           # (S, D)

        with torch.no_grad():
            self.p_mask.copy_(torch.from_numpy(init.astype(np.float32)))

    def init_wp_from_kmeans(self, cluster_labels: np.ndarray):
        """
        Permute Wp so that within-cluster connections come first.

        Re-ordering the token indices so each cluster forms a contiguous block
        exploits the lower-triangular structure: earlier (larger) clusters
        propagate their representation to later (smaller) clusters via Wp.

        Parameters
        ----------
        cluster_labels : (S,) int array
        """
        # Sort indices: largest cluster first, then by cluster id
        order = np.argsort(cluster_labels, kind='stable')
        # Reorder Wp rows and columns
        with torch.no_grad():
            W = self.Wp.data
            W = W[order][:, order]
            self.Wp.copy_(W)

    # ── masking helpers ─────────────────────────────────────────────────────

    def cluster_mask(self, x: torch.Tensor, cluster_labels: np.ndarray):
        """
        Mask one randomly chosen K-means cluster (the Z set for that step).

        The remaining clusters form the P (allow) set used for reconstruction.
        This implements the LT-PCA-kmean cluster-completion objective:
        given P, reconstruct Z.

        Parameters
        ----------
        x              : (B, S, D)
        cluster_labels : (S,) int array

        Returns
        -------
        masked_x : (B, S, D)   Z-cluster tokens set to 0
        mask     : (B, S) bool  True at masked (Z) positions
        """
        B, S, _ = x.shape
        k = np.random.randint(0, self.num_clusters)
        z_idx = np.where(cluster_labels == k)[0]

        mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)
        mask[:, z_idx] = True

        masked_x = x.clone()
        masked_x[mask] = 0.0
        return masked_x, mask

    def random_mask(self, x: torch.Tensor):
        """
        Fallback: randomly zero out mask_ratio of tokens (MAE-style).
        Used only when cluster labels are not available.
        """
        B, S, _ = x.shape
        num_masked = max(1, int(S * self.mask_ratio))

        noise       = torch.rand(B, S, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask_ids    = ids_shuffle[:, :num_masked]

        mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)
        mask.scatter_(1, mask_ids, True)

        masked_x       = x.clone()
        masked_x[mask] = 0.0
        return masked_x, mask

    # ── encoder ─────────────────────────────────────────────────────────────

    def _enforce_wp(self) -> torch.Tensor:
        """Return lower-triangular Wp with unit diagonal (no in-place)."""
        W = torch.tril(self.Wp)
        diag_ones = torch.ones(self.seq_len, device=self.Wp.device)
        W = W - torch.diag(W.diagonal()) + torch.diag(diag_ones)
        return W

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        CANDY sparse-masking encoder.

        Parameters
        ----------
        x : (B, S, D)

        Returns
        -------
        z : (B, S, D)
        """
        p_mask = self.phi(self.p_mask)                           # (S, D)
        p_set  = x * p_mask.unsqueeze(0)                         # (B, S, D)

        Wp  = self._enforce_wp()
        Wzp = self.Wzp

        p_out = self.enc_proj1(
            torch.bmm(Wp.unsqueeze(0).expand(x.size(0), -1, -1), p_set)
        )
        z_out = self.enc_proj2(
            torch.bmm(Wzp.unsqueeze(0).expand(x.size(0), -1, -1), p_out)
        )
        return p_out + z_out

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, return_loss: bool = True,
                cluster_labels: np.ndarray = None):
        """
        Parameters
        ----------
        x               : (B, C, S, D)
        return_loss     : bool
        cluster_labels  : (S,) int array or None
            If provided, uses cluster-completion masking (LT-PCA-kmean style).
            If None, falls back to random masking (MAE style).

        Returns
        -------
        return_loss=True  : (loss scalar, reconstructed (B,C,S,D))
        return_loss=False : encoded representation (B,C,S,D)
        """
        B, C, S, D = x.shape
        x_flat = x.view(B * C, S, D)

        if return_loss:
            if cluster_labels is not None:
                masked_x, mask = self.cluster_mask(x_flat, cluster_labels)
            else:
                masked_x, mask = self.random_mask(x_flat)

            encoded       = self.encode(masked_x)
            reconstructed = self.decoder(encoded)
            loss = F.mse_loss(reconstructed[mask], x_flat[mask])
            return loss, reconstructed.view(B, C, S, D)
        else:
            encoded = self.encode(x_flat)
            return encoded.view(B, C, S, D)
