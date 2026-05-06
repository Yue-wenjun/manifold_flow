"""
systems/neural.py

Continuous-time Neural Dynamics. Refactored to support N-Body phase space ensembles.
"""

import numpy as np
from typing import Optional, List
from ..core.types import StateVector, Projection3D, ParameterSet, Time
from ..core.base_system import DeterministicSystem

class CANDYNetwork(DeterministicSystem):
    def __init__(self, hidden_size: int = 10, num_particles: int = 1000):
        self.hidden_size = hidden_size
        self.num_particles = num_particles
        
        super().__init__(
            state_dim=num_particles * hidden_size * 2, 
            parameters={"alpha": 1.0, "decay": 1.0}
        )
        
        Wp_raw = np.random.randn(hidden_size, hidden_size) * 0.5
        self.Wp = np.tril(Wp_raw, -1) + np.eye(hidden_size)
        self.Wp_diag = np.random.randn(hidden_size) * 0.1
        self.Wzp = np.random.randn(hidden_size, hidden_size) * 0.5

    def _custom_activation(self, x: np.ndarray) -> np.ndarray:
        return (np.abs(x + 1.0) - np.abs(x - 1.0)) / 2.0

    def get_initial_conditions(self) -> StateVector:
        return np.random.normal(0, 2.0, size=self.state_dim)

    def drift(self, t: Time, y: StateVector) -> StateVector:
        Y = y.reshape(self.num_particles, self.hidden_size * 2)
        p = Y[:, :self.hidden_size]
        z = Y[:, self.hidden_size:]

        alpha = self.parameters["alpha"]
        decay = self.parameters["decay"]
        Wp_eff = self.Wp + np.diag(self.Wp_diag)

        dp = -decay * p + alpha * self._custom_activation(p @ Wp_eff.T)
        dz = -decay * z + alpha * self._custom_activation(p @ self.Wzp.T)

        return np.concatenate([dp, dz], axis=1).flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        Y = state.reshape(self.num_particles, self.hidden_size * 2)
        return Y[:, :3].copy()


class StandardNeuralODE(DeterministicSystem):
    def __init__(self, state_dim: int = 10, num_particles: int = 500):
        self.num_particles = num_particles
        self.base_dim = state_dim
        super().__init__(state_dim=num_particles * state_dim, parameters={"scale": 1.0})
        q, _ = np.linalg.qr(np.random.randn(state_dim, state_dim))
        self.W = q
        self.b = np.random.randn(state_dim) * 0.1

    def get_initial_conditions(self) -> StateVector:
        return np.random.uniform(-3.0, 3.0, size=self.state_dim)

    def drift(self, t: Time, y: StateVector) -> StateVector:
        Y = y.reshape(self.num_particles, self.base_dim)
        scale = self.parameters["scale"]
        dY = scale * np.tanh(Y @ self.W.T + self.b)
        return dY.flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        Y = state.reshape(self.num_particles, self.base_dim)
        return Y[:, :3].copy()


class HopfieldNetwork(DeterministicSystem):
    """
    Hopfield Network Vectorized phase space ensemble.
    Watches 1000 memory states collapse into stored pattern attractors.
    """
    def __init__(self, num_neurons: int = 4, num_particles: int = 1000):
        self.base_dim = num_neurons
        self.num_particles = num_particles
        super().__init__(
            state_dim=num_particles * num_neurons, 
            parameters={"cs": 1.0, "dr": 0.1}
        )
        
        # Define 3 distinct memories (attractors) to store
        patterns = np.array([
            [1.0, -1.0,  1.0, -1.0],
            [-1.0, 1.0, -1.0,  1.0],
            [1.0,  1.0, -1.0, -1.0]
        ])
        
        # Hebbian learning rule to compute weights
        self.weights = (patterns.T @ patterns) / 3.0
        np.fill_diagonal(self.weights, 0.0)

    def get_initial_conditions(self) -> StateVector:
        # Purely random memories in the high-dimensional space
        return np.random.uniform(-2.0, 2.0, size=self.state_dim)

    def drift(self, t: Time, y: StateVector) -> StateVector:
        Y = y.reshape(self.num_particles, self.base_dim)
        dr = self.parameters["dr"]
        cs = self.parameters["cs"]
        
        # Vectorized Hopfield dynamics: dY/dt = -decay * Y + cs * tanh(Y @ W^T)
        dY = -dr * Y + cs * np.tanh(Y @ self.weights.T)
        return dY.flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        Y = state.reshape(self.num_particles, self.base_dim)
        # We project the 4D network state down to the first 3 dimensions for WebGL
        return Y[:, :3].copy()


class TransformerAttentionSystem(DeterministicSystem):
    """
    Transformer Self-Attention Dynamics as an Interacting Particle System.
    Visualizes how 1000 token embeddings (particles) cluster and evolve 
    through continuous-time attention mechanism.
    Modern Hopfield Networks are mathematically equivalent to this process.
    """
    def __init__(self, embed_dim: int = 3, num_particles: int = 1000):
        self.embed_dim = embed_dim
        self.num_particles = num_particles
        super().__init__(
            state_dim=num_particles * embed_dim, 
            parameters={"lr": 2.0, "decay": 0.1, "temperature": 0.5}
        )
        
        # Q, K, V Projection Matrices
        # We use random orthogonal matrices to ensure stable, non-exploding dynamics
        # while creating distinct semantic subspaces for attention matching.
        q_q, _ = np.linalg.qr(np.random.randn(embed_dim, embed_dim))
        q_k, _ = np.linalg.qr(np.random.randn(embed_dim, embed_dim))
        q_v, _ = np.linalg.qr(np.random.randn(embed_dim, embed_dim))
        
        self.W_q = q_q
        self.W_k = q_k
        self.W_v = q_v

    def get_initial_conditions(self) -> StateVector:
        # Tokens start uniformly scattered in the embedding space
        return np.random.uniform(-8.0, 8.0, size=self.state_dim)

    def drift(self, t: Time, y: StateVector) -> StateVector:
        # Y represents the (1000, 3) matrix of all token embeddings
        Y = y.reshape(self.num_particles, self.embed_dim)
        
        lr = self.parameters["lr"]
        decay = self.parameters["decay"]
        temp = self.parameters["temperature"]
        
        # 1. Linear Projections
        Q = Y @ self.W_q
        K = Y @ self.W_k
        V = Y @ self.W_v
        
        # 2. Attention Scores: (Q @ K^T) / (sqrt(d) * temperature)
        # Shape: (1000, 1000) - each token computes affinity with all other 999 tokens
        scores = (Q @ K.T) / (np.sqrt(self.embed_dim) * temp)
        
        # 3. Softmax (with numerical stability trick to prevent overflow)
        scores_max = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        A = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # 4. Continuous Attention Update: dY/dt = -decay * Y + lr * Attention(Q, K, V)
        dY = -decay * Y + lr * (A @ V)
        
        return dY.flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        Y = state.reshape(self.num_particles, self.embed_dim)
        return Y[:, :3].copy()


class UNetDynamicsSystem(DeterministicSystem):
    """
    Continuous-time U-Net Dynamics.
    Models the Encoder-Decoder architecture with Skip Connections as 
    coupled N-Body vector fields.
    """
    def __init__(self, state_dim: int = 3, num_particles: int = 1000):
        self.base_dim = state_dim
        self.num_particles = num_particles
        # State dimension is doubled: [Encoder_State, Decoder_State]
        super().__init__(
            state_dim=num_particles * state_dim * 2, 
            parameters={"decay": 0.5, "skip_strength": 1.2}
        )
        
        # W_down (Encoder), W_up (Decoder), W_skip (Skip Connection)
        q_down, _ = np.linalg.qr(np.random.randn(state_dim, state_dim))
        q_up, _ = np.linalg.qr(np.random.randn(state_dim, state_dim))
        q_skip, _ = np.linalg.qr(np.random.randn(state_dim, state_dim))
        
        self.W_down = q_down * 1.5
        self.W_up = q_up * 1.5
        self.W_skip = q_skip

    def get_initial_conditions(self) -> StateVector:
        # Encoder starts with scattered data, Decoder starts empty (zeros)
        encoder_init = np.random.uniform(-4.0, 4.0, size=(self.num_particles, self.base_dim))
        decoder_init = np.random.normal(0, 0.1, size=(self.num_particles, self.base_dim))
        return np.concatenate([encoder_init, decoder_init], axis=1).flatten()

    def drift(self, t: Time, y: StateVector) -> StateVector:
        # Reshape to (1000, 6)
        Y = y.reshape(self.num_particles, self.base_dim * 2)
        E = Y[:, :self.base_dim]  # Encoder state
        D = Y[:, self.base_dim:]  # Decoder state
        
        decay = self.parameters["decay"]
        skip_str = self.parameters["skip_strength"]
        
        # Encoder Dynamics: Contraction/Downsampling
        dE = -decay * E + np.tanh(E @ self.W_down.T)
        
        # Decoder Dynamics: Expansion + Skip Connection Injection!
        # The skip connection acts as a driving force from the encoder
        dD = -decay * D + np.tanh(D @ self.W_up.T + skip_str * (E @ self.W_skip.T))
        
        return np.concatenate([dE, dD], axis=1).flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        # We visualize the Output (Decoder state)
        Y = state.reshape(self.num_particles, self.base_dim * 2)
        return Y[:, self.base_dim:].copy()


class CANDYDiffusionSystem(DeterministicSystem):
    """
    CANDY Diffusion (Continuous Adaptive Neural Dynamics Diffusion).
    Models your exact paper architecture:
    - Forward feature diffusion via CANDY masking
    - Reverse reconstruction via U-Net fusion and Graph Schedule (g(t))
    - Condenses into 'num_classes' segmentation targets.
    """
    def __init__(self, state_dim: int = 3, num_particles: int = 1000, num_classes: int = 4):
        self.base_dim = state_dim
        self.num_particles = num_particles
        super().__init__(
            state_dim=num_particles * state_dim, 
            parameters={
                "decay": 1.0,
                "unet_weight": 1.0,
                "origin_weight": 1.5,
                "candy_scale": 1.0,
                "T": 5.0
            }
        )
        
        self.Wp = np.eye(state_dim) + np.tril(np.random.randn(state_dim, state_dim) * 0.05, -1)
        self.W_fuse = -np.eye(state_dim) * 0.8
        self.W_orig = np.eye(state_dim) * 0.8
        self.W_unet = np.eye(state_dim) * 1.5
        
        # Define 4 segmentation targets (num_classes=4) simulating the origin/ground-truth
        class_centers = np.array([
            [ 4.0,  4.0,  4.0],
            [-4.0, -4.0,  4.0],
            [-4.0,  4.0, -4.0],
            [ 4.0, -4.0, -4.0]
        ])
        # Assign each particle a fixed target class
        target_indices = np.random.randint(0, num_classes, size=num_particles)
        self.targets = class_centers[target_indices]  # Shape: (1000, 3)

    def _custom_activation(self, x: np.ndarray) -> np.ndarray:
        # The exact mathematical activation from candy.py
        return (np.abs(x + 1.0) - np.abs(x - 1.0)) / 2.0

    def get_initial_conditions(self) -> StateVector:
        # Start from pure noise (Output of forward diffusion process)
        return np.random.normal(0, 3.0, size=self.state_dim)

    def drift(self, t: Time, y: StateVector) -> StateVector:
        Y = y.reshape(self.num_particles, self.base_dim)
        
        decay = self.parameters["decay"]
        unet_w = self.parameters["unet_weight"]
        orig_w = self.parameters["origin_weight"]
        candy_s = self.parameters["candy_scale"]
        T_param = self.parameters["T"]

        # Graph Schedule: linearly decays from 0.7 to 0.2 over diffusion time T
        g_t = max(0.2, 0.7 - (0.5 / max(T_param, 0.1)) * t)
        
        # 2. CANDY Masking operation
        candy_features = candy_s * self._custom_activation(Y @ self.Wp.T)
        
        # 3. Fusion Convolution (combining noisy features and origin targets)
        fusion_input = np.tanh(candy_features @ self.W_fuse.T + self.targets @ self.W_orig.T)
        
        # 4. Graph Factor Modulation (from your reverse diffusion loop)
        reverse_input = (1.0 - g_t) * fusion_input + g_t * self.targets
        
        # 5. U-Net Reconstruction Flow
        unet_reconstruction = np.tanh(reverse_input @ self.W_unet.T)
        
        # Final Vector Field: 
        # Particles flow towards the U-Net reconstruction while decaying chaotic momentum
        dY = -decay * Y + unet_w * unet_reconstruction + orig_w * g_t * (self.targets - Y)
        
        return dY.flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        return state.reshape(self.num_particles, self.base_dim).copy()