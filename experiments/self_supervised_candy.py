"""
experiments/self_supervised_candy.py

Exp S3 — Self-Supervised CANDY on Stanford Bunny
=================================================
Trains CANDY_SSL via cluster-completion pretraining on the Bunny point cloud
using only 3-D geometry (no part labels).  Evaluates the Fisher-scatter
semantic alignment score of the learned representations at each epoch.

Scientific question
    Can the CANDY sparse-masking encoder learn representations that respect
    the semantic part structure without oracle supervision?

Method (LT-PCA-kmean framework)
    Following the LT-PCA-kmean document, K-means (k=3) defines the P/Z split:
      P set = the two clusters not masked in this step
      Z set = the one masked cluster (randomly chosen per training step)
    The encoder is trained to reconstruct the Z set from the P set.
    This cluster-completion objective directly rewards inter-cluster
    discriminability, enabling the encoder to surpass PCA on Fisher score.

    The p_mask parameter is initialised from the K-means labels (dominant
    cluster → +1, others → −1), and Wp is re-ordered so that the largest
    cluster (body) occupies the leading rows of the lower-triangular matrix.

Baselines (all evaluated without labels)
    Random CANDY init  — Fisher score before any training
    PCA (3-component)  — classical unsupervised representation
    Supervised CANDY   — oracle upper bound (score ≈ 1.000)

Output figures
    exp_s3_training_curve.png   — loss + Fisher score vs. epoch
    exp_s3_representations.png  — PCA of learned repr, coloured by part label
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
import torch.optim as optim

from manifold_flow.systems.self_candy import CANDY_SSL
from multimodal_data import generate_part_labels, semantic_alignment_score

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data generation (same bunny as supervised experiments)
# ---------------------------------------------------------------------------

def make_bunny(num_points: int = 100, scale: float = 1.0) -> np.ndarray:
    """Return (num_points, 3) bunny point cloud normalised to [-1, 1]^3."""
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

    ears, per_ear = [], ear_count // 2
    for x_off in [-0.25, 0.25]:
        for i in range(per_ear):
            t = 2 * np.pi * i / per_ear
            h = 1.0 + 0.6 * (i / per_ear)
            ears.append([x_off + 0.1 * np.cos(t), 0.1 * np.sin(t), h])

    coords = np.array((body + ears)[:num_points], dtype=np.float32)
    coords -= coords.min(axis=0)
    coords /= (coords.max(axis=0) + 1e-8)
    coords = (coords - 0.5) * 2.0 * scale
    return coords


def random_rotate(coords: np.ndarray) -> np.ndarray:
    """Apply a random 3-D rotation (augmentation)."""
    angles = np.random.uniform(0, 2 * np.pi, 3).astype(np.float32)
    cx, sx = np.cos(angles[0]), np.sin(angles[0])
    cy, sy = np.cos(angles[1]), np.sin(angles[1])
    cz, sz = np.cos(angles[2]), np.sin(angles[2])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float32)
    return coords @ (Rx @ Ry @ Rz).T


def to_tensor(coords: np.ndarray) -> torch.Tensor:
    """(N, 3) → (1, 1, N, 3) float32 tensor."""
    return torch.from_numpy(coords).unsqueeze(0).unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------------------------
# Fisher-scatter score from CANDY_SSL encoder output
# ---------------------------------------------------------------------------

def fisher_score(model: CANDY_SSL, coords: np.ndarray,
                 labels: np.ndarray) -> float:
    """Encode coords (no mask) and compute Fisher-scatter score."""
    model.eval()
    with torch.no_grad():
        x   = to_tensor(coords)
        rep = model(x, return_loss=False)          # (1, 1, N, 3)
        rep_np = rep.squeeze().cpu().numpy()       # (N, 3)
    return semantic_alignment_score(rep_np, labels)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(num_points: int = 100, epochs: int = 500, lr: float = 3e-3,
          num_clusters: int = 3, log_every: int = 10, seed: int = 0):

    torch.manual_seed(seed)
    np.random.seed(seed)

    coords_np = make_bunny(num_points)
    labels    = generate_part_labels(num_points)

    # ── K-means clustering (unsupervised, no labels used) ──────────────────
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    kmeans.fit(coords_np)
    km_labels = kmeans.labels_           # (N,) int, values in {0,1,2}

    # Measure how well K-means alone aligns with part structure
    score_kmeans = semantic_alignment_score(
        kmeans.cluster_centers_[km_labels], labels   # centroid repr
    )
    print(f"K-means cluster alignment (unsupervised): {score_kmeans:.4f}")

    # ── Build model, initialise from K-means ───────────────────────────────
    model = CANDY_SSL(seq_len=num_points, feat_dim=3,
                      num_clusters=num_clusters).to(DEVICE)
    model.init_from_kmeans(km_labels)
    model.init_wp_from_kmeans(km_labels)

    opt = optim.Adam(model.parameters(), lr=lr)

    # ── Baselines before training ──────────────────────────────────────────
    score_random = fisher_score(model, coords_np, labels)
    pca          = PCA(n_components=3).fit_transform(coords_np)
    score_pca    = semantic_alignment_score(pca, labels)
    score_oracle = 1.000   # supervised CANDY

    print(f"Baselines — K-means init CANDY: {score_random:.4f}  "
          f"PCA: {score_pca:.4f}  oracle supervised: {score_oracle:.4f}")

    # ── Training ───────────────────────────────────────────────────────────
    loss_hist, score_hist, epoch_hist = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        coords_aug = random_rotate(coords_np)
        x = to_tensor(coords_aug)

        # Cluster-completion: mask one K-means cluster, reconstruct from others
        loss, _ = model(x, return_loss=True, cluster_labels=km_labels)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % log_every == 0 or epoch == 1:
            s = fisher_score(model, coords_np, labels)
            loss_hist.append(loss.item())
            score_hist.append(s)
            epoch_hist.append(epoch)
            print(f"  epoch {epoch:4d}  loss={loss.item():.5f}  "
                  f"Fisher={s:.4f}")

    final_score = fisher_score(model, coords_np, labels)
    print(f"\nFinal Fisher score (no rotation): {final_score:.4f}")

    return {
        "model"        : model,
        "coords"       : coords_np,
        "labels"       : labels,
        "km_labels"    : km_labels,
        "epoch_hist"   : epoch_hist,
        "loss_hist"    : loss_hist,
        "score_hist"   : score_hist,
        "score_random" : score_random,
        "score_pca"    : score_pca,
        "score_kmeans" : score_kmeans,
        "score_oracle" : score_oracle,
        "final_score"  : final_score,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_results(res: dict):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Figure 1: training curve (loss + Fisher score) ────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    ax1.plot(res["epoch_hist"], res["loss_hist"],
             color="#1f77b4", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Reconstruction loss (MSE)", fontsize=12)
    ax1.set_title("SSL Training Loss", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(res["epoch_hist"], res["score_hist"],
             color="#d62728", linewidth=2, label="CANDY-SSL (K-means init)")
    ax2.axhline(res["score_random"], color="gray",   linestyle=":",
                linewidth=1.5, label=f"K-means init, before training ({res['score_random']:.3f})")
    ax2.axhline(res["score_pca"],    color="#ff7f0e", linestyle="--",
                linewidth=1.5, label=f"PCA ({res['score_pca']:.3f})")
    ax2.axhline(res["score_kmeans"], color="#9467bd", linestyle="-.",
                linewidth=1.5, label=f"K-means centroids ({res['score_kmeans']:.3f})")
    ax2.axhline(res["score_oracle"], color="#2ca02c", linestyle="-.",
                linewidth=1.5, label=f"Oracle supervised ({res['score_oracle']:.3f})")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Fisher-scatter score", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Semantic Alignment Score", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Exp S3 — Self-Supervised CANDY (K-means cluster-completion)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path1 = os.path.join(FIGURES_DIR, "exp_s3_training_curve.png")
    fig.savefig(path1)
    plt.close(fig)
    print(f"Saved: {path1}")

    # ── Figure 2: PCA of learned representations ──────────────────────────
    res["model"].eval()
    with torch.no_grad():
        x   = to_tensor(res["coords"])
        rep = res["model"](x, return_loss=False).squeeze().cpu().numpy()

    pca2 = PCA(n_components=2)
    rep_2d    = pca2.fit_transform(rep)
    coords_2d = PCA(n_components=2).fit_transform(res["coords"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    part_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    part_names  = {0: "Body", 1: "Left ear", 2: "Right ear"}

    for ax, data, title in zip(
        axes,
        [coords_2d, rep_2d],
        ["Raw coordinates (PCA)", "CANDY-SSL repr. (PCA)"]
    ):
        for c in [0, 1, 2]:
            mask = res["labels"] == c
            ax.scatter(data[mask, 0], data[mask, 1],
                       color=part_colors[c], label=part_names[c],
                       s=30, alpha=0.85, edgecolors="none")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlabel("PC 1", fontsize=11)
        ax.set_ylabel("PC 2", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)

    score_raw = semantic_alignment_score(res["coords"], res["labels"])
    score_rep = semantic_alignment_score(rep, res["labels"])
    axes[0].set_title(f"Raw coordinates  (score={score_raw:.3f})",
                      fontsize=12, fontweight="bold")
    axes[1].set_title(f"CANDY-SSL repr.  (score={score_rep:.3f})",
                      fontsize=12, fontweight="bold")

    fig.suptitle("Exp S3 — Learned vs. raw representations (coloured by part label)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path2 = os.path.join(FIGURES_DIR, "exp_s3_representations.png")
    fig.savefig(path2)
    plt.close(fig)
    print(f"Saved: {path2}")

    return path1, path2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Exp S3 — Self-Supervised CANDY (K-means cluster-completion)")
    print("=" * 60)

    res = train(num_points=100, epochs=500, lr=3e-3, num_clusters=3,
                log_every=25)

    plot_results(res)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  K-means init (before training) : {res['score_random']:.4f}")
    print(f"  PCA (3-component)              : {res['score_pca']:.4f}")
    print(f"  K-means centroids (unsup.)     : {res['score_kmeans']:.4f}")
    print(f"  CANDY-SSL (trained)            : {res['final_score']:.4f}")
    print(f"  Oracle supervised              : {res['score_oracle']:.4f}")
    print("=" * 60)
