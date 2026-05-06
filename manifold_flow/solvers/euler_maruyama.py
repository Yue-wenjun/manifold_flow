"""
solvers/euler_maruyama.py

Euler-Maruyama integrator for Stochastic Differential Equations (SDEs).
Critically important for mathematically rigorous Diffusion Models.
"""

import numpy as np
from typing import Tuple, Optional

from .base_solver import SDESolver, SolverResult
from ..core.types import StateVector, Time
from ..core.base_system import StochasticSystem

class EulerMaruyamaSolver(SDESolver):
    def __init__(self, dt: float = 0.001):
        """
        Initialize the solver with a default time step.
        
        Args:
            dt: Default time step size.
        """
        self.dt = dt

    def solve_step(self, 
                  system: StochasticSystem, 
                  t: Time, 
                  y: StateVector, 
                  dt: Optional[float] = None) -> StateVector:
        """
        Solves a single step of the SDE.
        
        y_{n+1} = y_n + f(t, y_n)*dt + g(t, y_n)*dW
        """
        step_dt = dt if dt is not None else self.dt
        y = np.array(y, dtype=np.float64)
        sqrt_dt = np.sqrt(step_dt)
        
        # Deterministic component
        f = system.drift(t, y)
        
        # Stochastic component (Noise scaling)
        g = system.diffusion(t, y)
        
        # Brownian motion increment dW ~ N(0, dt)
        dW = np.random.normal(0, 1.0, size=y.shape)
        
        # Support both vector and matrix diffusion
        if len(g.shape) == 2:  # Coupled noise
            noise_term = (g @ dW) * sqrt_dt
        else:  # Diagonal/Vector noise
            noise_term = g * dW * sqrt_dt
            
        return y + f * step_dt + noise_term

    def solve(self, 
              system: StochasticSystem, 
              y0: StateVector, 
              t_span: Tuple[Time, Time], 
              dt: Optional[float] = None) -> SolverResult:
        
        step_dt = dt if dt is not None else self.dt
        t_start, t_end = t_span
        num_steps = int(np.ceil((t_end - t_start) / step_dt))
        times = np.linspace(t_start, t_end, num_steps + 1)
        
        states = np.zeros((num_steps + 1, system.state_dim))
        states[0] = y0
        
        y = np.array(y0, dtype=np.float64)
        
        for i in range(num_steps):
            y = self.solve_step(system, times[i], y, step_dt)
            states[i+1] = y
            
        return SolverResult(times, states)

# Alias for experiment compatibility
EulerMaruyamaUncorrected = EulerMaruyamaSolver
