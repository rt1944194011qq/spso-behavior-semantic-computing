"""Default semantic-memory MA-EoH engine and explicit legacy compatibility."""

from .legacy_engine import MAEEngine as LegacyMAEEngine
from .semantic_engine import MAEEngine

__all__ = ["MAEEngine", "LegacyMAEEngine"]
