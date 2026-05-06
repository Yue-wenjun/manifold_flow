"""
core/base_system.py

Top-level abstractions for dynamical systems.
Separates N-dimensional tensor computation from 3D visual projection.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from .types import StateVector, Projection3D, ParameterSet, Time

class Projectable3D(ABC):
    """
    Mixin contract ensuring any high-dimensional system can be 
    projected down to a 3D coordinate for the frontend Canvas/WebGL.
    """
    
    @abstractmethod
    def project_to_3d(self, state: StateVector) -> Projection3D:
        """
        Projects the N-dimensional state to a 3D coordinate [x, y, z].
        
        For 3D systems (Lorenz): Returns state directly.
        For ML systems: Applies PCA, UMAP, or selects specific feature dimensions.
        """
        pass

class DynamicalSystem(Projectable3D):
    """
    Base class for all dynamical systems (N-dimensional).
    """
    
    def __init__(self, state_dim: int, parameters: Optional[ParameterSet] = None):
        self.state_dim = state_dim
        self.parameters = parameters or {}

    @abstractmethod
    def get_initial_conditions(self) -> StateVector:
        """
        Returns the initial N-dimensional state as a NumPy array.
        """
        pass

    def update_parameters(self, new_params: ParameterSet) -> None:
        """
        Updates system parameters in-place.
        """
        self.parameters.update(new_params)

class DeterministicSystem(DynamicalSystem):
    """
    System governed by Ordinary Differential Equations (ODEs).
    Models exact trajectories (e.g., Classical Attractors, standard Neural ODEs).
    
    Equation form: dy/dt = drift(t, y)
    """
    
    @abstractmethod
    def drift(self, t: Time, y: StateVector) -> StateVector:
        """
        Computes the deterministic derivative vector.
        
        Args:
            t: Current time.
            y: Current N-dimensional state.
            
        Returns:
            The derivative dy/dt.
        """
        pass

class StochasticSystem(DynamicalSystem):
    """
    System governed by Stochastic Differential Equations (SDEs).
    Models probability flows and noise (e.g., Diffusion Models, SGLD).
    
    Equation form: dy = drift(t, y)dt + diffusion(t, y)dW
    """
    
    @abstractmethod
    def drift(self, t: Time, y: StateVector) -> StateVector:
        """
        Computes the deterministic drift term.
        """
        pass

    @abstractmethod
    def diffusion(self, t: Time, y: StateVector) -> StateVector:
        """
        Computes the stochastic diffusion term (noise scale).
        Returns a vector representing diagonal noise variance, 
        or a matrix for coupled noise.
        """
        pass