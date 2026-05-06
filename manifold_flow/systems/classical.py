"""
systems/classical.py

Classical chaotic attractors refactored for N-Body phase space ensembles.
Watch the Butterfly Effect tear a tight cluster of particles apart!
"""

import numpy as np
from ..core.types import StateVector, Projection3D, Time
from ..core.base_system import DeterministicSystem

class LorenzSystem(DeterministicSystem):
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0, num_particles: int = 1000):
        self.num_particles = num_particles
        super().__init__(state_dim=num_particles * 3, parameters={"sigma": sigma, "rho": rho, "beta": beta})

    def get_initial_conditions(self) -> StateVector:
        cluster = np.random.normal(loc=[1.0, 1.0, 1.0], scale=0.01, size=(self.num_particles, 3))
        return cluster.flatten()

    def drift(self, t: Time, y: StateVector) -> StateVector:
        Y = y.reshape(self.num_particles, 3)
        X, Y_coord, Z = Y[:, 0], Y[:, 1], Y[:, 2]
        p = self.parameters
        
        dX = p["sigma"] * (Y_coord - X)
        dY = X * (p["rho"] - Z) - Y_coord
        dZ = X * Y_coord - p["beta"] * Z
        
        return np.column_stack([dX, dY, dZ]).flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        return state.reshape(self.num_particles, 3).copy()


class RosslerSystem(DeterministicSystem):
    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7, num_particles: int = 1000):
        self.num_particles = num_particles
        super().__init__(state_dim=num_particles * 3, parameters={"a": a, "b": b, "c": c})

    def get_initial_conditions(self) -> StateVector:
        cluster = np.random.normal(loc=[1.0, 0.0, 0.0], scale=0.1, size=(self.num_particles, 3))
        return cluster.flatten()

    def drift(self, t: Time, y: StateVector) -> StateVector:
        Y = y.reshape(self.num_particles, 3)
        X, Y_coord, Z = Y[:, 0], Y[:, 1], Y[:, 2]
        p = self.parameters
        
        dX = -Y_coord - Z
        dY = X + p["a"] * Y_coord
        dZ = p["b"] + Z * (X - p["c"])
        
        return np.column_stack([dX, dY, dZ]).flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        return state.reshape(self.num_particles, 3).copy()


class ChuaSystem(DeterministicSystem):
    def __init__(self, alpha: float = 15.6, beta: float = 28.0, gamma: float = -1.143, 
                 a: float = -1.143, b: float = -0.714, num_particles: int = 1000):
        self.num_particles = num_particles
        super().__init__(state_dim=num_particles * 3, parameters={
            "alpha": alpha, "beta": beta, "gamma": gamma, "a": a, "b": b
        })

    def get_initial_conditions(self) -> StateVector:
        cluster = np.random.normal(loc=[0.1, 0.0, 0.0], scale=0.05, size=(self.num_particles, 3))
        return cluster.flatten()

    def drift(self, t: Time, y: StateVector) -> StateVector:
        Y = y.reshape(self.num_particles, 3)
        X, Y_coord, Z = Y[:, 0], Y[:, 1], Y[:, 2]
        p = self.parameters
        
        f_x = p["b"] * X + 0.5 * (p["a"] - p["b"]) * (np.abs(X + 1.0) - np.abs(X - 1.0))
        dX = p["alpha"] * (Y_coord - X - f_x)
        dY = X - Y_coord + Z
        dZ = -p["beta"] * Y_coord - p["gamma"] * Z
        
        return np.column_stack([dX, dY, dZ]).flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        return state.reshape(self.num_particles, 3).copy()