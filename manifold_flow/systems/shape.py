"""
systems/shape.py

Geometric shape attractors refactored for N-Body phase space ensembles.
"""

import numpy as np
from ..core.types import StateVector, Projection3D, Time
from ..core.base_system import DeterministicSystem

class TorusAttractor(DeterministicSystem):
    def __init__(self, rs: float = 1.0, num_particles: int = 1000):
        self.num_particles = num_particles
        super().__init__(state_dim=num_particles * 3, parameters={"rs": rs})

    def get_initial_conditions(self) -> StateVector:
        return np.random.uniform(-3.0, 3.0, size=self.state_dim)

    def drift(self, t: Time, y: StateVector) -> StateVector:
        Y = y.reshape(self.num_particles, 3)
        rs = self.parameters["rs"]

        v = Y[:, 2]
        dX = rs * np.cos(v) - 0.1 * Y[:, 0]
        dY = rs * np.sin(v) - 0.1 * Y[:, 1]
        dV = np.full(self.num_particles, 0.7 * rs)
        
        return np.column_stack([dX, dY, dV]).flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        return state.reshape(self.num_particles, 3).copy()


class RingAttractor(DeterministicSystem):
    def __init__(self, radius: float = 3.0, att_str: float = 1.5, flow_spd: float = 1.0, num_particles: int = 1000):
        self.radius = radius
        self.num_particles = num_particles
        super().__init__(state_dim=num_particles * 3, parameters={"att_str": att_str, "flow_spd": flow_spd})

    def get_initial_conditions(self) -> StateVector:
        return np.random.normal(0, 4.0, size=self.state_dim)

    def drift(self, t: Time, y: StateVector) -> StateVector:
        Y = y.reshape(self.num_particles, 3)
        X, Y_coord, Z = Y[:, 0], Y[:, 1], Y[:, 2]
        
        r = np.sqrt(X**2 + Y_coord**2 + 1e-6)
        att = self.parameters["att_str"]
        spd = self.parameters["flow_spd"]
        
        dX = att * (self.radius - r) * X / r - spd * Y_coord
        dY = att * (self.radius - r) * Y_coord / r + spd * X
        dZ = -att * Z
        
        return np.column_stack([dX, dY, dZ]).flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        return state.reshape(self.num_particles, 3).copy()


class PointAttractor(DeterministicSystem):
    def __init__(self, att_str: float = 1.0, num_particles: int = 1000):
        self.num_particles = num_particles
        super().__init__(state_dim=num_particles * 3, parameters={"att_str": att_str})

    def get_initial_conditions(self) -> StateVector:
        return np.random.uniform(-10.0, 10.0, size=self.state_dim)

    def drift(self, t: Time, y: StateVector) -> StateVector:
        att = self.parameters["att_str"]
        return -att * y

    def project_to_3d(self, state: StateVector) -> Projection3D:
        return state.reshape(self.num_particles, 3).copy()


class LineAttractor(DeterministicSystem):
    def __init__(self, att_str: float = 1.0, flow_spd: float = 0.5, num_particles: int = 1000):
        self.num_particles = num_particles
        super().__init__(state_dim=num_particles * 3, parameters={"att_str": att_str, "flow_spd": flow_spd})

    def get_initial_conditions(self) -> StateVector:
        return np.random.uniform(-5.0, 5.0, size=self.state_dim)

    def drift(self, t: Time, y: StateVector) -> StateVector:
        Y = y.reshape(self.num_particles, 3)
        att = self.parameters["att_str"]
        spd = self.parameters["flow_spd"]
        
        dX = -att * Y[:, 0]
        dY = -att * Y[:, 1]
        dZ = np.full(self.num_particles, spd)
        
        return np.column_stack([dX, dY, dZ]).flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        return state.reshape(self.num_particles, 3).copy()


class DiscreteAttractor(DeterministicSystem):
    def __init__(self, att_str: float = 1.0, switch_rate: float = 1.0, num_particles: int = 1000):
        self.num_particles = num_particles
        super().__init__(state_dim=num_particles * 3, parameters={"att_str": att_str, "switch_rate": switch_rate})

    def get_initial_conditions(self) -> StateVector:
        return np.random.normal(0, 3.0, size=self.state_dim)

    def drift(self, t: Time, y: StateVector) -> StateVector:
        Y = y.reshape(self.num_particles, 3)
        att = self.parameters["att_str"]
        sw = self.parameters["switch_rate"]
        
        dX = -att * Y[:, 0] + sw * 5.0 * np.sin(t)
        dY = -att * Y[:, 1] + sw * 5.0 * np.cos(t)
        dZ = -att * Y[:, 2] + sw * 5.0 * np.sin(2 * t)
        
        return np.column_stack([dX, dY, dZ]).flatten()

    def project_to_3d(self, state: StateVector) -> Projection3D:
        return state.reshape(self.num_particles, 3).copy()