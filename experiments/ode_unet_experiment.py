"""
experiments/ode_unet_experiment.py

Neural ODE vs U-Net Dynamics Experiment
========================================
Runs two attractor-theoretic neural dynamical systems side by side
and measures the distinct attractor structures each one produces:

  1. StandardNeuralODE  (state_dim=10, num_particles=500)
       Orthogonal weight matrix W + tanh nonlinearity, NO decay term.
       Theoretical prediction: neutral-orbit dynamics (conservative-like),
       particles maintain dispersion — contrast to Hopfield's strong point
       attractors.

  2. UNetDynamicsSystem (state_dim=3, num_particles=1000)
       Encoder-Decoder ODE with skip connections as driving force.
       Encoder contracts toward its own attractor; Decoder is slaved to
       Encoder via skip, producing a two-level hierarchical attractor.

Metrics
-------
  NeuralODE:
    - fixed_pt_dist(t)   : mean ||Y_i - Y*|| where Y* = -b @ W (analytical fixed point)
    - trajectory_energy(t): mean ||dY/dt||  (roughly constant → neutral stability)
    - particle_dispersion(t): std of particle positions (preserved → no collapse)

  U-Net:
    - encoder_spread(t)  : std of encoder state E across particles
    - decoder_spread(t)  : std of decoder state D across particles
    - skip_signal(t)     : mean ||E @ W_skip|| (skip-connection driving force)
    - enc_dec_corr(t)    : mean correlation between E and D per dimension

Scientific hypothesis
---------------------
  NeuralODE with orthogonal W (no decay): eigenvalues of Jacobian lie on
  the unit circle → neutral stability → particles orbit rather than collapse.
  fixed_pt_dist stays non-zero; trajectory_energy stays roughly constant.

  U-Net: encoder convergence drives decoder via skip connection.
  encoder_spread decreases while skip_signal remains significant, and
  enc_dec_corr rises toward 1 — demonstrating hierarchical attractor coupling.
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


# ---------------------------------------------------------------------------
# Metrics — NeuralODE
# ---------------------------------------------------------------------------

def neural_ode_fixed_point(system: StandardNeuralODE) -> np.ndarray:
    """Analytical fixed point Y* = -b @ W  (shape: (base_dim,))."""
    return -system.b @ system.W


def fixed_pt_dist(states_flat, num_particles, base_dim, Y_star):
    """Mean distance of each particle to the analytical fixed point."""
    Y = states_flat.reshape(num_particles, base_dim)
    return float(np.mean(np.linalg.norm(Y - Y_star, axis=1)))


def trajectory_energy(states_flat, num_particles, base_dim, system, t):
    """Mean ||dY/dt|| — proxy for how much the system is still 'moving'."""
    drift = system.drift(t, states_flat)
    D = drift.reshape(num_particles, base_dim)
    return float(np.mean(np.linalg.norm(D, axis=1)))


def particle_dispersion(states_flat, num_particles, base_dim):
    """Mean std of particle positions across all dimensions."""
    Y = states_flat.reshape(num_particles, base_dim)
    return float(np.mean(np.std(Y, axis=0)))


# ---------------------------------------------------------------------------
# Metrics — UNet
# ---------------------------------------------------------------------------

def unet_encoder_spread(states_flat, num_particles, base_dim):
    """Std of encoder state E across particles."""
    Y = states_flat.reshape(num_particles, base_dim * 2)
    E = Y[:, :base_dim]
    return float(np.mean(np.std(E, axis=0)))


def unet_decoder_spread(states_flat, num_particles, base_dim):
    """Std of decoder state D across particles."""
    Y = states_flat.reshape(num_particles, base_dim * 2)
    D = Y[:, base_dim:]
    return float(np.mean(np.std(D, axis=0)))


def unet_skip_signal(states_flat, num_particles, base_dim, W_skip):
    """Mean Frobenius norm of skip-connection projection E @ W_skip^T."""
    Y = states_flat.reshape(num_particles, base_dim * 2)
    E = Y[:, :base_dim]
    skip = E @ W_skip.T
    return float(np.mean(np.linalg.norm(skip, axis=1)))


def unet_enc_dec_corr(states_flat, num_particles, base_dim):
    """
    Mean per-dimension Pearson correlation between encoder E and decoder D
    across particles.  Range [-1, 1]; rises toward 1 when skip coupling
    succeeds in 'locking' decoder to encoder.
    """
    Y = states_flat.reshape(num_particles, base_dim * 2)
    E = Y[:, :base_dim]
    D = Y[:, base_dim:]
    corrs = []
    for d in range(base_dim):
        e_col = E[:, d]
        d_col = D[:, d]
        if np.std(e_col) < 1e-9 or np.std(d_col) < 1e-9:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(e_col, d_col)[0, 1]))
    return float(np.mean(corrs))


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class OdeUnetExperiment:
    def __init__(self, num_particles_ode=500, num_particles_unet=1000,
                 experiment_name="ode_unet_experiment"):
        self.num_particles_ode  = num_particles_ode
        self.num_particles_unet = num_particles_unet
        self.experiment_name    = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ode_system   = None
        self.unet_system  = None
        self.ode_result   = None
        self.unet_result  = None
        self.metrics      = {}

    def setup(self):
        print(f"[{self.experiment_name}] Setting up systems...")
        self.ode_system  = StandardNeuralODE(state_dim=10,
                                              num_particles=self.num_particles_ode)
        self.unet_system = UNetDynamicsSystem(state_dim=3,
                                              num_particles=self.num_particles_unet)
        self.solver = RK4Solver()
        print(f"  NeuralODE  : state_dim={self.ode_system.base_dim}, "
              f"particles={self.num_particles_ode}")
        print(f"  UNet       : state_dim={self.unet_system.base_dim}, "
              f"particles={self.num_particles_unet}")
        print("[OK] Systems ready")

    def run(self, t_end=2.0, dt=0.02):
        print(f"[{self.experiment_name}] Integrating (t_end={t_end}, dt={dt})...")
        y0_ode  = self.ode_system.get_initial_conditions()
        y0_unet = self.unet_system.get_initial_conditions()

        self.ode_result  = self.solver.solve(self.ode_system,  y0_ode,  (0.0, t_end), dt)
        self.unet_result = self.solver.solve(self.unet_system, y0_unet, (0.0, t_end), dt)
        print(f"  NeuralODE  trajectory: {self.ode_result.states.shape}")
        print(f"  UNet       trajectory: {self.unet_result.states.shape}")
        print("[OK] Integrations complete")

    def analyze(self):
        print(f"[{self.experiment_name}] Analysing attractor dynamics...")

        ode_states  = self.ode_result.states
        unet_states = self.unet_result.states
        ode_times   = self.ode_result.times
        unet_times  = self.unet_result.times

        bd = self.ode_system.base_dim
        Y_star = neural_ode_fixed_point(self.ode_system)

        # --- NeuralODE metrics ---
        ode_fpdist   = [fixed_pt_dist(s, self.num_particles_ode, bd, Y_star)
                        for s in ode_states]
        ode_energy   = [trajectory_energy(s, self.num_particles_ode, bd,
                                          self.ode_system, t)
                        for s, t in zip(ode_states, ode_times)]
        ode_disp     = [particle_dispersion(s, self.num_particles_ode, bd)
                        for s in ode_states]

        # --- U-Net metrics ---
        ud = self.unet_system.base_dim
        W_skip = self.unet_system.W_skip
        unet_enc_sp  = [unet_encoder_spread(s, self.num_particles_unet, ud)
                        for s in unet_states]
        unet_dec_sp  = [unet_decoder_spread(s, self.num_particles_unet, ud)
                        for s in unet_states]
        unet_skip    = [unet_skip_signal(s, self.num_particles_unet, ud, W_skip)
                        for s in unet_states]
        unet_corr    = [unet_enc_dec_corr(s, self.num_particles_unet, ud)
                        for s in unet_states]

        self.metrics.update({
            # NeuralODE
            "ode_fixed_pt_dist_initial": ode_fpdist[0],
            "ode_fixed_pt_dist_final":   ode_fpdist[-1],
            "ode_energy_initial":        ode_energy[0],
            "ode_energy_final":          ode_energy[-1],
            "ode_dispersion_initial":    ode_disp[0],
            "ode_dispersion_final":      ode_disp[-1],
            "ode_dispersion_ratio":      ode_disp[-1] / max(ode_disp[0], 1e-9),
            # U-Net
            "unet_enc_spread_initial":   unet_enc_sp[0],
            "unet_enc_spread_final":     unet_enc_sp[-1],
            "unet_dec_spread_initial":   unet_dec_sp[0],
            "unet_dec_spread_final":     unet_dec_sp[-1],
            "unet_skip_initial":         unet_skip[0],
            "unet_skip_final":           unet_skip[-1],
            "unet_enc_dec_corr_initial": unet_corr[0],
            "unet_enc_dec_corr_final":   unet_corr[-1],
        })

        self._ode_times    = ode_times
        self._unet_times   = unet_times
        self._ode_fpdist   = ode_fpdist
        self._ode_energy   = ode_energy
        self._ode_disp     = ode_disp
        self._unet_enc_sp  = unet_enc_sp
        self._unet_dec_sp  = unet_dec_sp
        self._unet_skip    = unet_skip
        self._unet_corr    = unet_corr

        print(f"\n  --- StandardNeuralODE ---")
        print(f"    Fixed-pt distance  : {ode_fpdist[0]:.4f} -> {ode_fpdist[-1]:.4f}  "
              f"(ratio {ode_fpdist[-1]/max(ode_fpdist[0],1e-9):.2f})")
        print(f"    Trajectory energy  : {ode_energy[0]:.4f} -> {ode_energy[-1]:.4f}")
        print(f"    Particle dispersion: {ode_disp[0]:.4f} -> {ode_disp[-1]:.4f}  "
              f"(ratio {self.metrics['ode_dispersion_ratio']:.2f})")
        print(f"\n  --- UNetDynamicsSystem ---")
        print(f"    Encoder spread : {unet_enc_sp[0]:.4f} -> {unet_enc_sp[-1]:.4f}")
        print(f"    Decoder spread : {unet_dec_sp[0]:.4f} -> {unet_dec_sp[-1]:.4f}")
        print(f"    Skip signal    : {unet_skip[0]:.4f} -> {unet_skip[-1]:.4f}")
        print(f"    Enc-Dec corr   : {unet_corr[0]:.4f} -> {unet_corr[-1]:.4f}")

    def visualize(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[{self.experiment_name}] Generating visualizations...")

        # ── Plot 1: metrics panel (2×4 subplots) ──────────────────────────
        fig, axes = plt.subplots(2, 4, figsize=(22, 9))

        # Row 0: NeuralODE
        axes[0, 0].plot(self._ode_times, self._ode_fpdist, "steelblue", linewidth=2)
        axes[0, 0].set_title("NeuralODE: Fixed-Point Distance\n"
                              r"$\|Y_i - Y^*\|$, $Y^* = -b W$")
        axes[0, 0].set_xlabel("Time t"); axes[0, 0].set_ylabel("Mean distance")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self._ode_times, self._ode_energy, "darkorange", linewidth=2)
        axes[0, 1].set_title("NeuralODE: Trajectory Energy\n"
                              r"mean $\|\dot{Y}\|$ (constant → neutral stability)")
        axes[0, 1].set_xlabel("Time t"); axes[0, 1].set_ylabel("Mean ||dY/dt||")
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(self._ode_times, self._ode_disp, "purple", linewidth=2)
        axes[0, 2].set_title("NeuralODE: Particle Dispersion\n"
                              "std across particles (preserved → no collapse)")
        axes[0, 2].set_xlabel("Time t"); axes[0, 2].set_ylabel("Mean coord std")
        axes[0, 2].grid(True, alpha=0.3)

        axes[0, 3].axis("off")
        summary_text = (
            "NeuralODE (orthogonal W)\n\n"
            "• No decay term → conservative-like\n"
            "• Eigenvalues of Jacobian on unit circle\n"
            "• Neutral stability: particles orbit\n"
            "  around fixed point, no collapse\n\n"
            f"Dispersion ratio: {self.metrics['ode_dispersion_ratio']:.3f}\n"
            f"(~1.0 = preserved, <<1.0 = collapsed)"
        )
        axes[0, 3].text(0.05, 0.5, summary_text, transform=axes[0, 3].transAxes,
                        fontsize=10, va="center",
                        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        axes[0, 3].set_title("NeuralODE Summary")

        # Row 1: U-Net
        axes[1, 0].plot(self._unet_times, self._unet_enc_sp, "steelblue",
                        linewidth=2, label="Encoder")
        axes[1, 0].plot(self._unet_times, self._unet_dec_sp, "tomato",
                        linewidth=2, label="Decoder")
        axes[1, 0].set_title("U-Net: Encoder & Decoder Spread\n"
                              "(decoder follows encoder via skip)")
        axes[1, 0].set_xlabel("Time t"); axes[1, 0].set_ylabel("Mean coord std")
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self._unet_times, self._unet_skip, "seagreen", linewidth=2)
        axes[1, 1].set_title("U-Net: Skip Connection Signal\n"
                              r"mean $\|E W_{\rm skip}^T\|$")
        axes[1, 1].set_xlabel("Time t"); axes[1, 1].set_ylabel("Signal magnitude")
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(self._unet_times, self._unet_corr, "darkorchid", linewidth=2)
        axes[1, 2].axhline(0, color="gray", linestyle="--", linewidth=1)
        axes[1, 2].set_title("U-Net: Encoder–Decoder Correlation\n"
                              "(rises → skip coupling locks decoder to encoder)")
        axes[1, 2].set_xlabel("Time t"); axes[1, 2].set_ylabel("Mean Pearson r")
        axes[1, 2].set_ylim(-1.1, 1.1); axes[1, 2].grid(True, alpha=0.3)

        axes[1, 3].axis("off")
        summary_unet = (
            "UNetDynamicsSystem\n\n"
            "• Encoder: decay + W_down (contracts)\n"
            "• Decoder: decay + W_up + skip driving\n"
            "• Skip connection = two-level hierarchy\n\n"
            f"Enc spread: {self.metrics['unet_enc_spread_initial']:.3f}"
            f" → {self.metrics['unet_enc_spread_final']:.3f}\n"
            f"Dec spread: {self.metrics['unet_dec_spread_initial']:.3f}"
            f" → {self.metrics['unet_dec_spread_final']:.3f}\n"
            f"Enc-Dec corr: {self.metrics['unet_enc_dec_corr_initial']:.3f}"
            f" → {self.metrics['unet_enc_dec_corr_final']:.3f}"
        )
        axes[1, 3].text(0.05, 0.5, summary_unet, transform=axes[1, 3].transAxes,
                        fontsize=10, va="center",
                        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8))
        axes[1, 3].set_title("U-Net Summary")

        fig.suptitle("Attractor Dynamics: NeuralODE (neutral orbits) vs U-Net (hierarchical attractor)",
                     fontsize=13)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/ode_unet_metrics.png", dpi=150)
        plt.close()

        # ── Plot 2: 3D particle snapshots ──────────────────────────────────
        fig = plt.figure(figsize=(20, 8))
        snap_fracs = [0.0, 0.5, 1.0]
        snap_labels = ["t=0 (initial)", "t=mid", "t=end"]

        # Row 0: NeuralODE — project to first 3 dims
        for col, (frac, lbl) in enumerate(zip(snap_fracs, snap_labels)):
            idx   = int(frac * (len(self.ode_result.states) - 1))
            state = self.ode_result.states[idx]
            pts   = state.reshape(self.num_particles_ode, self.ode_system.base_dim)[:, :3]
            ax    = fig.add_subplot(2, 3, col + 1, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c="steelblue", alpha=0.2, s=5)
            # Mark fixed point
            Y_star = neural_ode_fixed_point(self.ode_system)
            ax.scatter([Y_star[0]], [Y_star[1]], [Y_star[2]],
                       c="red", marker="*", s=200, zorder=5, label="Y*")
            ax.set_title(f"NeuralODE — {lbl}", fontsize=9)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.legend(fontsize=7)

        # Row 1: U-Net — project to decoder state (first 3 dims of decoder)
        for col, (frac, lbl) in enumerate(zip(snap_fracs, snap_labels)):
            idx   = int(frac * (len(self.unet_result.states) - 1))
            state = self.unet_result.states[idx]
            ud    = self.unet_system.base_dim
            pts   = state.reshape(self.num_particles_unet, ud * 2)[:, ud:]
            ax    = fig.add_subplot(2, 3, col + 4, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c="tomato", alpha=0.2, s=5)
            ax.set_title(f"U-Net Decoder — {lbl}", fontsize=9)
            ax.tick_params(labelsize=6)

        fig.suptitle("3D Particle Snapshots: NeuralODE (top) vs U-Net Decoder (bottom)",
                     fontsize=13)
        fig.tight_layout()
        plt.savefig(f"{save_dir}/ode_unet_snapshots.png", dpi=150)
        plt.close()

        print(f"[OK] Visualizations saved to {save_dir}/")

    def save_results(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        out = {
            "experiment": self.experiment_name,
            "timestamp":  self.timestamp,
            "config": {
                "num_particles_ode":  self.num_particles_ode,
                "num_particles_unet": self.num_particles_unet,
                "ode_state_dim":  self.ode_system.base_dim,
                "unet_state_dim": self.unet_system.base_dim,
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
    print("NeuralODE vs U-Net Dynamics Experiment")
    print("=" * 60)
    exp = OdeUnetExperiment(num_particles_ode=500, num_particles_unet=1000)
    exp.setup()
    exp.run(t_end=2.0, dt=0.02)
    exp.analyze()
    FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    exp.visualize(save_dir=FIGURES_DIR)
    exp.save_results(save_dir=FIGURES_DIR)
    print("\n[OK] Done.")
    return exp


if __name__ == "__main__":
    exp = main()
