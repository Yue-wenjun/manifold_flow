"""
solvers/base_solver.py

Abstract base classes for numerical integrators.
Strictly separates ODE and SDE solving paradigms.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

from ..core.types import StateVector, Time
from ..core.base_system import DeterministicSystem, StochasticSystem

class SolverResult:
    """Standardized output for all integrators."""
    def __init__(self, times: np.ndarray, states: np.ndarray):
        self.times = times
        self.states = states  # Shape: (num_steps, state_dim)

class ODESolver(ABC):
    """Base class for Deterministic System integrators."""
    
    @abstractmethod
    def solve(self, 
              system: DeterministicSystem, 
              y0: StateVector, 
              t_span: Tuple[Time, Time], 
              dt: float) -> SolverResult:
        pass

class SDESolver(ABC):
    """Base class for Stochastic System integrators."""
    
    @abstractmethod
    def solve(self, 
              system: StochasticSystem, 
              y0: StateVector, 
              t_span: Tuple[Time, Time], 
              dt: float) -> SolverResult:
        pass