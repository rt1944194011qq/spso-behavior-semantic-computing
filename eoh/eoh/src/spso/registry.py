"""Typed module and parameter registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .models import ModuleChoice, ParticlePosition


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: str = "float"  # float, int, choice
    default: Any = None
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    def validate(self, value: Any) -> Any:
        if self.kind == "int":
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"parameter {self.name!r} must be an integer")
            value = int(value)
        elif self.kind == "float":
            value = float(value)
        if self.kind == "choice" and value not in self.choices:
            raise ValueError(f"parameter {self.name!r} must be one of {self.choices}")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"parameter {self.name!r} is below its minimum")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"parameter {self.name!r} is above its maximum")
        return value


@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    description: str
    parameters: tuple[ParameterSpec, ...] = ()
    tags: tuple[str, ...] = ()
    reverse_of: str | None = None
    similar_to: tuple[str, ...] = ()

    def default_params(self) -> dict[str, Any]:
        return {
            spec.name: spec.default
            for spec in self.parameters
            if spec.default is not None
        }


@dataclass(frozen=True)
class SlotSpec:
    slot_id: str
    description: str
    modules: tuple[ModuleSpec, ...]
    min_cut: int = 1
    max_cut: int = 4


class ModuleRegistry:
    """Registry enforcing slot membership and parameter contracts."""

    def __init__(self, slots: Iterable[SlotSpec]):
        self.slots = tuple(slots)
        if not self.slots:
            raise ValueError("at least one slot is required")
        self._slots = {slot.slot_id: slot for slot in self.slots}
        if len(self._slots) != len(self.slots):
            raise ValueError("slot IDs must be unique")
        self._modules: dict[tuple[str, str], ModuleSpec] = {}
        for slot in self.slots:
            if not slot.modules:
                raise ValueError(f"slot {slot.slot_id!r} has no modules")
            ids = set()
            for module in slot.modules:
                if module.module_id in ids:
                    raise ValueError(f"duplicate module {module.module_id!r} in slot {slot.slot_id!r}")
                ids.add(module.module_id)
                self._modules[(slot.slot_id, module.module_id)] = module

    @property
    def slot_ids(self) -> tuple[str, ...]:
        return tuple(slot.slot_id for slot in self.slots)

    def slot(self, slot_id: str) -> SlotSpec:
        try:
            return self._slots[slot_id]
        except KeyError as exc:
            raise ValueError(f"unknown slot {slot_id!r}") from exc

    def module(self, slot_id: str, module_id: str) -> ModuleSpec:
        try:
            return self._modules[(slot_id, module_id)]
        except KeyError as exc:
            raise ValueError(f"module {module_id!r} is not valid in slot {slot_id!r}") from exc

    def validate_choice(self, choice: ModuleChoice) -> ModuleChoice:
        spec = self.module(choice.slot, choice.module)
        supplied = choice.params_dict()
        allowed = {param.name: param for param in spec.parameters}
        unknown = set(supplied) - set(allowed)
        if unknown:
            raise ValueError(f"unknown parameters for {choice.module!r}: {sorted(unknown)}")
        values = spec.default_params()
        values.update(supplied)
        values = {name: allowed[name].validate(value) for name, value in values.items()}
        return ModuleChoice.create(choice.slot, choice.module, values)

    def validate_position(self, position: ParticlePosition) -> ParticlePosition:
        mapping = position.as_mapping()
        if set(mapping) != set(self.slot_ids):
            raise ValueError(f"position must contain exactly slots {self.slot_ids}")
        validated = tuple(self.validate_choice(mapping[slot]) for slot in self.slot_ids)
        return ParticlePosition(validated)

    def random_position(self, rng) -> ParticlePosition:
        import random

        if not isinstance(rng, random.Random):
            raise TypeError("rng must be random.Random")
        choices = []
        for slot in self.slots:
            module = rng.choice(slot.modules)
            choices.append(ModuleChoice.create(slot.slot_id, module.module_id, module.default_params()))
        return self.validate_position(ParticlePosition(tuple(choices)))

    def describe(self) -> str:
        lines = []
        for slot in self.slots:
            lines.append(f"[{slot.slot_id}] {slot.description}")
            for module in slot.modules:
                params = ", ".join(p.name for p in module.parameters) or "none"
                lines.append(f"- {module.module_id}: {module.description}; params={params}")
        return "\n".join(lines)
