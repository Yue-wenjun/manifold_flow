"""
systems/diffusion.py

SDE and ODE-based Diffusion Models. 
True physical simulation without artificial freezing. Systems gracefully transition 
into Langevin dynamics at terminal time states.
"""

import numpy as np
from typing import Optional, List
from ..core.types import StateVector, Projection3D, ParameterSet, Time
from ..core.base_system import DeterministicSystem, StochasticSystem

class NoiseSchedule:
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: float) -> float: 
        return self.beta_min + t * (self.beta_max - self.beta_min)
        
    def integral_beta(self, t: float) -> float: 
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)
        
    def alpha_t(self, t: float) -> float: 
        return np.exp(-self.integral_beta(t))

class GMMScoreFunction:
    def __init__(self, centers: List[np.ndarray], schedule: NoiseSchedule):
        self.centers = centers
        self.schedule = schedule

    def compute_score(self, x: np.ndarray, t: float) -> np.ndarray:
        t = max(t, 0.001)  # prevent zero-variance domain error
        alpha = self.schedule.alpha_t(t)
        mean_scale = np.sqrt(alpha)
        
        var = max(1.0 - alpha, 0.001)

        log_probs = []
        score_components = []

        for mu in self.centers:
            diff = x - mean_scale * mu 
            log_p = -np.sum(diff**2, axis=1) / (2 * var) 
            log_probs.append(log_p)
            score_components.append(-diff / var)

        log_probs = np.array(log_probs) 
        max_log_p = np.max(log_probs, axis=0)
        probs = np.exp(log_probs - max_log_p)
        probs /= np.sum(probs, axis=0) 

        score = np.zeros_like(x)
        for i in range(len(self.centers)):
            score += probs[i, :][:, np.newaxis] * score_components[i]

        norm = np.linalg.norm(score, axis=1, keepdims=True)
        score = np.where(norm > 15.0, score / norm * 15.0, score)

        return score

class ForwardDiffusionSDE(StochasticSystem):
    def __init__(self, state_dim: int = 3, num_particles: int = 1000):
        self.base_dim = state_dim
        self.num_particles = num_particles
        super().__init__(
            state_dim=num_particles * state_dim, 
            parameters={
                "beta_min": 0.1, 
                "beta_max": 20.0,
                "temperature": 1.0  
            }
        )
        self.schedule = NoiseSchedule(0.1, 20.0)

    def get_initial_conditions(self) -> StateVector:
        Y = np.zeros((self.num_particles, self.base_dim))
        Y[:, 0] = np.random.normal(0.0, 0.2, self.num_particles)
        Y[:, 1] = np.random.normal(0.0, 0.2, self.num_particles)
        Y[:, 2] = np.random.normal(0.0, 0.2, self.num_particles)
        return Y.flatten()

    def drift(self, t: Time, y: StateVector) -> StateVector:
        self.schedule.beta_min = self.parameters["beta_min"]
        self.schedule.beta_max = self.parameters["beta_max"]
        
        t_eff = min(t, 0.999)
        return -0.5 * self.schedule.beta(t_eff) * y

    def diffusion(self, t: Time, y: StateVector) -> StateVector:
        self.schedule.beta_min = self.parameters["beta_min"]
        self.schedule.beta_max = self.parameters["beta_max"]
        temp = self.parameters["temperature"]
        
        t_eff = min(t, 0.999)
        return np.full(self.state_dim, np.sqrt(self.schedule.beta(t_eff)) * temp)

    def project_to_3d(self, state: StateVector) -> Projection3D:
        return state.reshape(self.num_particles, self.base_dim)[:, :3].copy()

class ReverseDiffusionSDE(StochasticSystem):
    def __init__(self, state_dim: int = 3, num_particles: int = 1000):
        self.base_dim = state_dim
        self.num_particles = num_particles
        super().__init__(
            state_dim=num_particles * state_dim, 
            parameters={
                "beta_min": 0.1,
                "beta_max": 20.0,
                "temperature": 1.0  
            }
        )
        self.schedule = NoiseSchedule()
        
        centers = [
            np.array([ 3.0,  3.0,  3.0]), np.array([-3.0, -3.0,  3.0]),
            np.array([-3.0,  3.0, -3.0]), np.array([ 3.0, -3.0, -3.0])
        ]
        padded_centers = []
        for c in centers:
            if state_dim > 3: c = np.concatenate([c, np.zeros(state_dim - 3)])
            padded_centers.append(c)
        self.score_fn = GMMScoreFunction(padded_centers, self.schedule)

    def get_initial_conditions(self) -> StateVector:
        return np.random.normal(0, 1.0, size=self.state_dim)

    def drift(self, t: Time, y: StateVector) -> StateVector:
        Y = y.reshape(self.num_particles, self.base_dim)
        
        self.schedule.beta_min = self.parameters["beta_min"]
        self.schedule.beta_max = self.parameters["beta_max"]
        
        tau = max(1.0 - t, 0.001)
        
        beta_tau = self.schedule.beta(tau)
        score = self.score_fn.compute_score(Y, tau)
        
        dY = 0.5 * beta_tau * Y + beta_tau * score
        return dY.flatten()

    def diffusion(self, t: Time, y: StateVector) -> StateVector:
        self.schedule.beta_min = self.parameters["beta_min"]
        self.schedule.beta_max = self.parameters["beta_max"]
        temp = self.parameters["temperature"]
        
        tau = max(1.0 - t, 0.001)
        
        return np.full(self.state_dim, np.sqrt(self.schedule.beta(tau)) * temp)

    def project_to_3d(self, state: StateVector) -> Projection3D:
        return state.reshape(self.num_particles, self.base_dim)[:, :3].copy()

class ProbabilityFlowODE(DeterministicSystem):
    def __init__(self, state_dim: int = 3, num_particles: int = 1000):
        self.base_dim = state_dim
        self.num_particles = num_particles
        super().__init__(
            state_dim=num_particles * state_dim, 
            parameters={
                "beta_min": 0.1,
                "beta_max": 20.0,
                "temperature": 1.0  
            }
        )
        self.schedule = NoiseSchedule()
        
        centers = [
            np.array([ 3.0,  3.0,  3.0]), np.array([-3.0, -3.0,  3.0]),
            np.array([-3.0,  3.0, -3.0]), np.array([ 3.0, -3.0, -3.0])
        ]
        padded_centers = []
        for c in centers:
            if state_dim > 3: c = np.concatenate([c, np.zeros(state_dim - 3)])
            padded_centers.append(c)
        self.score_fn = GMMScoreFunction(padded_centers, self.schedule)

    def get_initial_conditions(self) -> StateVector:
        return np.random.normal(0, 1.0, size=self.state_dim)

    def drift(self, t: Time, y: StateVector) -> StateVector:
        Y = y.reshape(self.num_particles, self.base_dim)
        
        self.schedule.beta_min = self.parameters["beta_min"]
        self.schedule.beta_max = self.parameters["beta_max"]
        temp = self.parameters["temperature"]
        
        tau = max(1.0 - t, 0.001)
        
        beta_tau = self.schedule.beta(tau)
        score = self.score_fn.compute_score(Y, tau)
        
        dY = 0.5 * beta_tau * Y + 0.5 * beta_tau * score * temp
        return dY.flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        return state.reshape(self.num_particles, self.base_dim)[:, :3].copy()