"""Semantic PSO for typed heuristic program spaces.

The package is deliberately independent from EoH's whole-program evolution
loop.  Task adapters provide a typed module registry, a deterministic compiler
and an evaluator; :class:`SPSOEngine` supplies the reusable search process.
"""

from .engine import SPSOConfig, SPSOEngine
from .models import ModuleChoice, ParticlePosition, SetPosition
from .registry import ModuleRegistry, ModuleSpec, ParameterSpec, SlotSpec

__all__ = [
    "ModuleChoice",
    "ModuleRegistry",
    "ModuleSpec",
    "ParameterSpec",
    "ParticlePosition",
    "SetPosition",
    "SPSOConfig",
    "SPSOEngine",
    "SlotSpec",
]
