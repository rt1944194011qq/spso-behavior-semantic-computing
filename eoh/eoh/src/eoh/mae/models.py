from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProgramIndividual:
    individual_id: str
    description: str
    code: str
    code_hash: str
    fitness: float | None
    valid: bool
    generation: int
    agent_id: int
    source_operator: str
    guidance_operator: str | None = None
    variation_operator: str | None = None
    operator_path: str = "H0"
    parent_ids: list[str] = field(default_factory=list)
    source_cut_ids: list[str] = field(default_factory=list)
    prompt_hash: str | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None
    cached_fitness: bool = False
    evaluation_seconds: float | None = None
    evaluation_details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProgramIndividual":
        return cls(**data)


@dataclass
class AgentState:
    agent_id: int
    position_ids: list[str] = field(default_factory=list)
    pbest_id: str | None = None
    pbest_fitness: float | None = None
    velocity_h1: dict[str, float] = field(default_factory=dict)
    velocity_h2: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        return cls(**data)


@dataclass(frozen=True)
class CutItem:
    individual_id: str
    probability: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    valid: bool
    stage: str | None = None
    reason: str | None = None
