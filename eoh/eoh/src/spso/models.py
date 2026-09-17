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
    """A complete ordered module sequence.

    This is the decoded/program position: exactly one module choice is kept in
    every slot, and the choice may carry compiler parameters.  The PSO search
    state itself is represented by :class:`SetPosition`; keeping the two types
    separate prevents possibility values from leaking into the compiler API.
    """

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


@dataclass(frozen=True)
class SetPosition:
    """A crisp set-valued particle position.

    ``slots`` stores ``(slot_id, (module_id, ...))`` pairs.  Module IDs have
    no possibility values here; those values live in the separate velocity
    mapping.  The registry validates slot membership, uniqueness and each
    slot's configured capacity before a position enters the engine.
    """

    slots: tuple[tuple[str, tuple[str, ...]], ...]

    @classmethod
    def create(cls, slots: Mapping[str, Any]) -> "SetPosition":
        normalized = []
        for slot_id, module_ids in slots.items():
            normalized.append((str(slot_id), tuple(str(module_id) for module_id in module_ids)))
        return cls(tuple(normalized))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SetPosition":
        raw = data.get("slots", data)
        if not isinstance(raw, Mapping):
            raise ValueError("set position must contain a slots mapping")
        return cls.create(raw)

    def as_mapping(self) -> dict[str, tuple[str, ...]]:
        return {slot_id: tuple(module_ids) for slot_id, module_ids in self.slots}

    def modules(self, slot_id: str) -> tuple[str, ...]:
        try:
            return self.as_mapping()[slot_id]
        except KeyError as exc:
            raise ValueError(f"set position has no slot {slot_id!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"slots": {
            slot_id: list(module_ids) for slot_id, module_ids in self.slots
        }}

    def canonical(self, slot_order: tuple[str, ...]) -> tuple[Any, ...]:
        mapping = self.as_mapping()
        return tuple((slot, tuple(mapping[slot])) for slot in slot_order)

    def text(self, slot_order: tuple[str, ...]) -> str:
        mapping = self.as_mapping()
        return " | ".join(f"{slot}={list(mapping[slot])}" for slot in slot_order)
