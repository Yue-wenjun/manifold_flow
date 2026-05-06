"""
test_engine.py

Automated sanity check for Manifold Flow v2.0 Engine.
Validates instantiation, SDE/ODE solving, parameter updates, and 3D projection.
"""

import numpy as np
import traceback
import sys

from manifold_flow.core.base_system import DeterministicSystem, StochasticSystem
from manifold_flow.solvers.rk4_solver import RK4Solver
from manifold_flow.solvers.euler_maruyama import EulerMaruyamaSolver

from manifold_flow.systems.classical import LorenzSystem, RosslerSystem, ChuaSystem
from manifold_flow.systems.shape import TorusAttractor, DiscreteAttractor
from manifold_flow.systems.diffusion import ForwardDiffusionSDE, ReverseDiffusionSDE, ProbabilityFlowODE
from manifold_flow.systems.manifold import TSNEDynamicsSystem, UMAPDynamicsSystem
from manifold_flow.systems.neural import CANDYNetwork, StandardNeuralODE

def run_diagnostics():
    print("="*60)
    print("🚀 Starting Manifold Flow v2.0 Engine Diagnostics")
    print("="*60)

    systems_to_test = [
        ("Lorenz Attractor", LorenzSystem()),
        ("Rossler Attractor", RosslerSystem()),
        ("Chua Circuit", ChuaSystem()),
        ("Torus Attractor", TorusAttractor()),
        ("Discrete Attractor (4D)", DiscreteAttractor()),
        ("Forward Diffusion (SDE)", ForwardDiffusionSDE(state_dim=3)),
        ("Reverse Diffusion (SDE)", ReverseDiffusionSDE(state_dim=3)),
        ("Probability Flow (ODE)", ProbabilityFlowODE(state_dim=3)),
        ("t-SNE Dynamics (N-Body)", TSNEDynamicsSystem(perplexity=5.0)),
        ("UMAP Dynamics (N-Body)", UMAPDynamicsSystem()),
        ("CANDY Network", CANDYNetwork(hidden_size=10)),
        ("Standard Neural ODE", StandardNeuralODE(state_dim=10))
    ]

    ode_solver = RK4Solver()
    sde_solver = EulerMaruyamaSolver()

    dt = 0.01
    t_span = (0.0, 0.05)
    
    passed = 0
    failed = 0

    for name, system in systems_to_test:
        print(f"\n[{name}]")
        try:
            y0 = system.get_initial_conditions()
            assert y0.shape == (system.state_dim,), f"Initial condition shape mismatch: {y0.shape} vs {system.state_dim}"
            print(f"  ✓ Initial Condition OK (Dim: {system.state_dim})")

            if isinstance(system, StochasticSystem):
                result = sde_solver.solve(system, y0, t_span, dt)
                solver_name = "Euler-Maruyama"
            elif isinstance(system, DeterministicSystem):
                result = ode_solver.solve(system, y0, t_span, dt)
                solver_name = "RK4"
            else:
                raise TypeError("System must inherit from DeterministicSystem or StochasticSystem")

            assert result.states.shape[1] == system.state_dim, "Solver output dimension mismatch"
            print(f"  ✓ {solver_name} Integration OK (Generated {len(result.times)} steps)")

            last_state = result.states[-1]
            proj_3d = system.project_to_3d(last_state)

            is_valid_projection = proj_3d.shape == (3,) or (len(proj_3d.shape) == 2 and proj_3d.shape[1] == 3)
            assert is_valid_projection, f"Projection failed, got shape {proj_3d.shape}"
            print(f"  ✓ 3D Projection OK (Output Shape: {proj_3d.shape})")
            old_params = system.parameters.copy()
            dummy_update = {list(old_params.keys())[0]: 999.9} if old_params else {}
            if dummy_update:
                system.update_parameters(dummy_update)
                updated_key = list(dummy_update.keys())[0]
                assert system.parameters[updated_key] == 999.9, "Parameter update failed to mutate state"
                print(f"  ✓ Parameter Update OK ({updated_key} -> 999.9)")
            
            passed += 1

        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED: {str(e)}")
            traceback.print_exc(limit=2, file=sys.stdout)

    print("\n" + "="*60)
    print(f"🏁 Diagnostics Complete: {passed} Passed, {failed} Failed.")
    print("="*60)

if __name__ == "__main__":
    run_diagnostics()