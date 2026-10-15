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


@dataclass
class ExperienceRecord:
    """A reusable difference directed from the old current toward a reference."""

    experience_id: str
    source_role: str
    source_id: str
    target_id: str
    difference_prompt: str
    source_fitness: float
    probability: float
    created_generation: int
    last_seen_generation: int
    observations: int = 1
    direction_type: str = "improve"
    target_fitness: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperienceRecord":
        restored = dict(data)
        # Read checkpoints produced before the direction labels were clarified.
        if "source_id" not in restored and "worse_id" in restored:
            restored["source_id"] = restored.pop("worse_id")
            restored["target_id"] = restored.pop("better_id")
        if "target_fitness" not in restored:
            # Older semantic-memory checkpoints stored the target fitness in
            # the field named source_fitness.  Preserve their numeric behavior
            # while making all newly written records explicit about both ends.
            restored["target_fitness"] = restored.get("source_fitness")
        return cls(**restored)


@dataclass
class SemanticAgentState:
    agent_id: int
    current_id: str
    pbest_id: str
    pbest_fitness: float
    velocity: dict[str, ExperienceRecord] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "current_id": self.current_id,
            "pbest_id": self.pbest_id,
            "pbest_fitness": self.pbest_fitness,
            "velocity": {key: item.to_dict() for key, item in self.velocity.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticAgentState":
        return cls(
            agent_id=int(data["agent_id"]),
            current_id=data["current_id"],
            pbest_id=data["pbest_id"],
            pbest_fitness=float(data["pbest_fitness"]),
            velocity={
                key: ExperienceRecord.from_dict(value)
                for key, value in data.get("velocity", {}).items()
            },
        )


@dataclass(frozen=True)
class ExperienceCutItem:
    experience_id: str
    source_role: str
    memory_probability: float
    boosted_probability: float
    selection_probability: float
    reserved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
