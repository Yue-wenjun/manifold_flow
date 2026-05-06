"""
solvers/rk4_solver.py

Standard 4th-order Runge-Kutta integrator for ODEs.
"""

import numpy as np
from typing import Tuple

from .base_solver import ODESolver, SolverResult
from ..core.types import StateVector, Time
from ..core.base_system import DeterministicSystem

class RK4Solver(ODESolver):
    def solve(self, 
              system: DeterministicSystem, 
              y0: StateVector, 
              t_span: Tuple[Time, Time], 
              dt: float) -> SolverResult:
        
        t_start, t_end = t_span
        num_steps = int(np.ceil((t_end - t_start) / dt))
        times = np.linspace(t_start, t_end, num_steps + 1)
        
        # Pre-allocate trajectory array for performance
        states = np.zeros((num_steps + 1, system.state_dim))
        states[0] = y0
        
        y = np.array(y0, dtype=np.float64)
        
        for i in range(num_steps):
            t = times[i]
            
            k1 = system.drift(t, y)
            k2 = system.drift(t + dt/2, y + dt * k1 / 2)
            k3 = system.drift(t + dt/2, y + dt * k2 / 2)
            k4 = system.drift(t + dt, y + dt * k3)
            
            y = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            states[i+1] = y
            
        return SolverResult(times, states)