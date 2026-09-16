"""Immutable value objects used by the semantic search engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _freeze_params(params: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    if not params:
        return ()
    return tuple(sorted((str(k), v) for k, v in params.items()))


@dataclass(frozen=True)
class ModuleChoice:
    """One validated module choice in one typed slot."""

    slot: str
    module: str
    params: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def create(cls, slot: str, module: str, params: Mapping[str, Any] | None = None) -> "ModuleChoice":
        return cls(slot=str(slot), module=str(module), params=_freeze_params(params))

    def params_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def to_dict(self) -> dict[str, Any]:
        return {"slot": self.slot, "module": self.module, "params": self.params_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModuleChoice":
        return cls.create(data["slot"], data["module"], data.get("params"))


@dataclass(frozen=True)
class ParticlePosition:
    """A complete ordered module sequence."""

    choices: tuple[ModuleChoice, ...]

    @classmethod
    def from_mapping(cls, choices: Mapping[str, ModuleChoice]) -> "ParticlePosition":
        return cls(tuple(choices.values()))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParticlePosition":
        if "choices" in data:
            raw = data["choices"]
        else:
            raw = data.get("sequence", [])
        return cls(tuple(ModuleChoice.from_dict(item) for item in raw))

    def as_mapping(self) -> dict[str, ModuleChoice]:
        return {choice.slot: choice for choice in self.choices}

    def to_dict(self) -> dict[str, Any]:
        return {"choices": [choice.to_dict() for choice in self.choices]}

    def canonical(self, slot_order: tuple[str, ...]) -> tuple[Any, ...]:
        mapping = self.as_mapping()
        return tuple(
            (slot, mapping[slot].module, mapping[slot].params)
            for slot in slot_order
        )

    def algorithm_text(self, slot_order: tuple[str, ...]) -> str:
        mapping = self.as_mapping()
        parts = []
        for slot in slot_order:
            choice = mapping[slot]
            params = choice.params_dict()
            suffix = f"({params})" if params else ""
            parts.append(f"{slot}={choice.module}{suffix}")
        return " | ".join(parts)
