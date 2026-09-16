"""Compiler protocol shared by task adapters."""

from __future__ import annotations

from typing import Protocol

from .models import ParticlePosition
from .registry import ModuleRegistry


class HeuristicCompiler(Protocol):
    registry: ModuleRegistry

    def compile(self, position: ParticlePosition) -> tuple[str, str]:
        """Return (source_code, human-readable algorithm description)."""
