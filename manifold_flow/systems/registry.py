"""
systems/registry.py

Manifold Flow v2.0 System Registry.
Cleanly manages all the refactored tensor-based ODE/SDE dynamical systems.
"""

from typing import Dict, List, Type, Optional, Any
from dataclasses import dataclass
from enum import Enum
from ..core.base_system import DynamicalSystem

class SystemCategory(Enum):
    CLASSICAL = "classical"
    NEURAL = "neural"
    SHAPE = "shape"
    DIFFUSION = "diffusion"
    MANIFOLD = "manifold"

@dataclass
class SystemInfo:
    name: str
    system_class: Type[DynamicalSystem]
    category: SystemCategory
    description: str
    parameters: Dict[str, Any]
    dimension: int
    documentation: str = ""

class SystemRegistry:
    def __init__(self):
        self._systems: Dict[str, SystemInfo] = {}
        self._register_v2_systems()

    def register(self, name: str, system_class: Type[DynamicalSystem], category: SystemCategory, description: str):
        temp_instance = system_class()
        self._systems[name] = SystemInfo(
            name=name, system_class=system_class, category=category, description=description,
            parameters=temp_instance.parameters, dimension=temp_instance.state_dim,
            documentation=f"v2.0 Tensor Engine Model: {name}"
        )

    def get_system(self, name: str, **kwargs) -> DynamicalSystem:
        if name not in self._systems: raise ValueError(f"Unknown system: {name}")
        system = self._systems[name].system_class()
        if kwargs: system.update_parameters(kwargs)
        return system

    def list_systems(self, category: Optional[str] = None) -> List[SystemInfo]:
        systems = list(self._systems.values())
        if category: systems = [s for s in systems if s.category.value == category]
        return sorted(systems, key=lambda s: (s.category.value, s.name))

    def get_system_info(self, name: str) -> SystemInfo:
        if name not in self._systems: raise ValueError(f"Unknown system: {name}")
        return self._systems[name]

    def _register_v2_systems(self):
        from .classical import LorenzSystem, RosslerSystem, ChuaSystem
        from .shape import TorusAttractor, DiscreteAttractor, RingAttractor, PointAttractor, LineAttractor
        from .diffusion import ForwardDiffusionSDE, ReverseDiffusionSDE, ProbabilityFlowODE
        from .manifold import TSNEDynamicsSystem, UMAPDynamicsSystem
        from .neural import CANDYNetwork, StandardNeuralODE, HopfieldNetwork, TransformerAttentionSystem, UNetDynamicsSystem, CANDYDiffusionSystem

        self.register("lorenz", LorenzSystem, SystemCategory.CLASSICAL, "Lorenz Chaotic Attractor")
        self.register("rossler", RosslerSystem, SystemCategory.CLASSICAL, "Rossler Chaotic Attractor")
        self.register("chua", ChuaSystem, SystemCategory.CLASSICAL, "Chua's Circuit")
        
        self.register("torus", TorusAttractor, SystemCategory.SHAPE, "Torus Attractor (3D)")
        self.register("discrete", DiscreteAttractor, SystemCategory.SHAPE, "Discrete Switching Attractor")
        self.register("ring", RingAttractor, SystemCategory.SHAPE, "Ring Attractor")
        self.register("point", PointAttractor, SystemCategory.SHAPE, "Point Attractor (Black Hole)")
        self.register("line", LineAttractor, SystemCategory.SHAPE, "Line Attractor")
        
        self.register("forward_diffusion", ForwardDiffusionSDE, SystemCategory.DIFFUSION, "Forward Diffusion SDE")
        self.register("reverse_diffusion", ReverseDiffusionSDE, SystemCategory.DIFFUSION, "Reverse Generation SDE")
        self.register("probability_flow", ProbabilityFlowODE, SystemCategory.DIFFUSION, "Probability Flow ODE")
        
        self.register("tsne", TSNEDynamicsSystem, SystemCategory.MANIFOLD, "t-SNE Embedding Dynamics")
        self.register("umap", UMAPDynamicsSystem, SystemCategory.MANIFOLD, "UMAP Manifold Learning")
        
        self.register("candy", CANDYNetwork, SystemCategory.NEURAL, "CANDY Network Dynamics")
        self.register("neural_ode", StandardNeuralODE, SystemCategory.NEURAL, "Standard Neural ODE")
        self.register("hopfield", HopfieldNetwork, SystemCategory.NEURAL, "Hopfield Memory Network")
        self.register("transformer", TransformerAttentionSystem, SystemCategory.NEURAL, "Transformer Attention Dynamics")
        self.register("unet", UNetDynamicsSystem, SystemCategory.NEURAL, "Continuous U-Net with Skip Connections")
        self.register("candy_diffusion", CANDYDiffusionSystem, SystemCategory.NEURAL, "CANDY Diffusion with Graph Schedule")

_global_registry = SystemRegistry()
def get_system(name: str, **kwargs) -> DynamicalSystem: return _global_registry.get_system(name, **kwargs)
def list_systems(category: Optional[str] = None) -> List[SystemInfo]: return _global_registry.list_systems(category)
def get_system_info(name: str) -> SystemInfo: return _global_registry.get_system_info(name)