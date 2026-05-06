"""
core/types.py

Type definitions for Manifold Flow v2.0.
Enforces high-performance NumPy arrays for all state computations.
"""

import numpy as np
from typing import Dict, Union, Any

# N-dimensional state vector for internal mathematical computation
StateVector = np.ndarray

# 3-dimensional vector strictly for frontend visual projection
Projection3D = np.ndarray  # Expected shape: (3,)

# Parameters can be scalars or arrays (e.g., weight matrices)
ParameterValue = Union[float, int, np.ndarray]
ParameterSet = Dict[str, ParameterValue]

# Time representation
Time = float