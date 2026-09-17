"""Typed module and parameter registry."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import threading
from typing import Any, Iterable, Mapping

from .models import ModuleChoice, ParticlePosition, SetPosition


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
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"parameter {self.name!r} must be an integer")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"parameter {self.name!r} must be finite")
            if int(value) != value:
                raise ValueError(f"parameter {self.name!r} must be an integer")
            value = int(value)
        elif self.kind == "float":
            if isinstance(value, bool):
                raise ValueError(f"parameter {self.name!r} must be a number")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"parameter {self.name!r} must be finite")
        if self.kind == "choice" and value not in self.choices:
            raise ValueError(f"parameter {self.name!r} must be one of {self.choices}")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"parameter {self.name!r} is below its minimum")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"parameter {self.name!r} is above its maximum")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "choices": list(self.choices),
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterSpec":
        return cls(
            name=str(data["name"]),
            kind=str(data.get("kind", "float")),
            default=data.get("default"),
            choices=tuple(data.get("choices", ())),
            minimum=data.get("minimum"),
            maximum=data.get("maximum"),
        )


@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    description: str
    parameters: tuple[ParameterSpec, ...] = ()
    tags: tuple[str, ...] = ()
    reverse_of: str | None = None
    similar_to: tuple[str, ...] = ()
    # ``recipe`` is a typed task-adapter DSL.  It is deliberately not raw
    # Python supplied by the LLM.
    recipe: Mapping[str, Any] = field(default_factory=dict)
    generation: int = 0
    source_operator: str = "seed"
    semantic_family: str = ""
    polarity: str = "neutral"
    content_hash: str = ""

    def default_params(self) -> dict[str, Any]:
        return {
            spec.name: spec.default
            for spec in self.parameters
            if spec.default is not None
        }

    def normalized_hash(self) -> str:
        payload = {
            "slot": self.module_id,
            "description": self.description,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "recipe": dict(self.recipe),
            "family": self.semantic_family,
            "polarity": self.polarity,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()

    def implementation_fingerprint(self, slot_id: str) -> str:
        """Hash only executable recipe and bounded parameter semantics.

        Module IDs, prose descriptions and tags intentionally do not enter
        this fingerprint. They are metadata and must not make a duplicate
        implementation appear new.
        """
        payload = {
            "slot": str(slot_id),
            # Only fields consumed by the task renderer are semantic.  Family,
            # polarity and prose are metadata and must not bypass deduplication.
            "recipe_kind": str(self.recipe.get("kind", "")),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "description": self.description,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "tags": list(self.tags),
            "reverse_of": self.reverse_of,
            "similar_to": list(self.similar_to),
            "recipe": dict(self.recipe),
            "generation": self.generation,
            "source_operator": self.source_operator,
            "semantic_family": self.semantic_family,
            "polarity": self.polarity,
            "content_hash": self.content_hash or self.normalized_hash(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModuleSpec":
        spec = cls(
            module_id=str(data["module_id"]),
            description=str(data.get("description", "")),
            parameters=tuple(ParameterSpec.from_dict(item) for item in data.get("parameters", ())),
            tags=tuple(data.get("tags", ())),
            reverse_of=data.get("reverse_of"),
            similar_to=tuple(data.get("similar_to", ())),
            recipe=dict(data.get("recipe", {})),
            generation=int(data.get("generation", 0)),
            source_operator=str(data.get("source_operator", "seed")),
            semantic_family=str(data.get("semantic_family", "")),
            polarity=str(data.get("polarity", "neutral")),
            content_hash=str(data.get("content_hash", "")),
        )
        return replace(spec, content_hash=spec.content_hash or spec.normalized_hash())


@dataclass(frozen=True)
class SlotSpec:
    slot_id: str
    description: str
    modules: tuple[ModuleSpec, ...]
    min_cut: int = 1
    max_cut: int = 4
    set_capacity: int | None = None

    @property
    def capacity(self) -> int:
        """Configured ``K_s``; small custom slots default to all modules."""
        return min(3, len(self.modules)) if self.set_capacity is None else self.set_capacity


class ModuleRegistry:
    """Registry enforcing slot membership and parameter contracts."""

    def __init__(self, slots: Iterable[SlotSpec]):
        self._lock = threading.RLock()
        self._slot_list = list(slots)
        if not self._slot_list:
            raise ValueError("at least one slot is required")
        self._slots = {slot.slot_id: slot for slot in self._slot_list}
        if len(self._slots) != len(self._slot_list):
            raise ValueError("slot IDs must be unique")
        self._modules: dict[tuple[str, str], ModuleSpec] = {}
        for slot in self._slot_list:
            if not slot.modules:
                raise ValueError(f"slot {slot.slot_id!r} has no modules")
            if not 1 <= slot.capacity <= len(slot.modules):
                raise ValueError(
                    f"slot {slot.slot_id!r} set_capacity must be in [1, number of modules]"
                )
            if not 1 <= slot.min_cut <= slot.max_cut:
                raise ValueError(f"slot {slot.slot_id!r} has invalid cut bounds")
            ids = set()
            fingerprints = set()
            for module in slot.modules:
                if module.module_id in ids:
                    raise ValueError(f"duplicate module {module.module_id!r} in slot {slot.slot_id!r}")
                ids.add(module.module_id)
                fingerprint = module.implementation_fingerprint(slot.slot_id)
                if fingerprint in fingerprints:
                    raise ValueError(f"duplicate implementation in slot {slot.slot_id!r}")
                fingerprints.add(fingerprint)
                self._modules[(slot.slot_id, module.module_id)] = module

    @property
    def slots(self) -> tuple[SlotSpec, ...]:
        with self._lock:
            return tuple(self._slot_list)

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

    def validate_set_position(self, position: SetPosition) -> SetPosition:
        """Validate a crisp set-valued position against every slot schema."""
        mapping = position.as_mapping()
        if len(mapping) != len(position.slots):
            raise ValueError("set position contains duplicate slot IDs")
        if set(mapping) != set(self.slot_ids):
            raise ValueError(f"set position must contain exactly slots {self.slot_ids}")
        validated = []
        for slot in self.slots:
            module_ids = tuple(mapping[slot.slot_id])
            if not module_ids:
                raise ValueError(f"set position slot {slot.slot_id!r} cannot be empty")
            if len(module_ids) > slot.capacity:
                raise ValueError(
                    f"set position slot {slot.slot_id!r} exceeds capacity {slot.capacity}"
                )
            if len(set(module_ids)) != len(module_ids):
                raise ValueError(f"set position slot {slot.slot_id!r} contains duplicates")
            for module_id in module_ids:
                self.module(slot.slot_id, module_id)
            validated.append((slot.slot_id, module_ids))
        return SetPosition(tuple(validated))

    def register_module(self, slot_id: str, module: ModuleSpec) -> ModuleSpec:
        """Atomically add one validated module to a slot.

        Registration is performed by the generation coordinator before worker
        threads start compiling/evaluating children.  The method still uses a
        lock so checkpoint and inspection readers see a consistent library.
        """
        with self._lock:
            slot = self.slot(slot_id)
            if module.module_id in {item.module_id for item in slot.modules}:
                raise ValueError(f"duplicate module {module.module_id!r} in slot {slot_id!r}")
            if module.content_hash and module.content_hash != module.normalized_hash():
                raise ValueError(f"content hash mismatch for {slot_id}/{module.module_id}")
            fingerprint = module.implementation_fingerprint(slot_id)
            if any(item.implementation_fingerprint(slot_id) == fingerprint for item in slot.modules):
                raise ValueError(f"duplicate implementation in slot {slot_id!r}")
            normalized = replace(module, content_hash=module.content_hash or module.normalized_hash())
            updated = replace(slot, modules=slot.modules + (normalized,))
            self._slot_list = [updated if item.slot_id == slot_id else item for item in self._slot_list]
            self._slots = {item.slot_id: item for item in self._slot_list}
            self._modules[(slot_id, normalized.module_id)] = normalized
        return normalized

    def library_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "slots": [
                    {
                        "slot_id": slot.slot_id,
                        "description": slot.description,
                        "min_cut": slot.min_cut,
                        "max_cut": slot.max_cut,
                        "set_capacity": slot.capacity,
                        "modules": [module.to_dict() for module in slot.modules],
                    }
                    for slot in self._slot_list
                ]
            }

    def library_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.library_payload(), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()

    def restore_library(self, payload: Mapping[str, Any]) -> None:
        """Restore the module library before any position is decoded."""
        slots = []
        for raw_slot in payload.get("slots", []):
            slots.append(SlotSpec(
                slot_id=str(raw_slot["slot_id"]),
                description=str(raw_slot.get("description", "")),
                modules=tuple(ModuleSpec.from_dict(item) for item in raw_slot.get("modules", ())),
                min_cut=int(raw_slot.get("min_cut", 1)),
                max_cut=int(raw_slot.get("max_cut", 4)),
                set_capacity=int(raw_slot.get("set_capacity", 3)),
            ))
        replacement = ModuleRegistry(slots)
        with self._lock:
            self._slot_list = list(replacement.slots)
            self._slots = dict(replacement._slots)
            self._modules = dict(replacement._modules)

    def random_position(self, rng) -> ParticlePosition:
        import random

        if not isinstance(rng, random.Random):
            raise TypeError("rng must be random.Random")
        choices = []
        for slot in self.slots:
            module = rng.choice(slot.modules)
            choices.append(ModuleChoice.create(slot.slot_id, module.module_id, module.default_params()))
        return self.validate_position(ParticlePosition(tuple(choices)))

    def random_set_position(self, rng) -> SetPosition:
        """Draw a full-capacity crisp set for each slot."""
        import random

        if not isinstance(rng, random.Random):
            raise TypeError("rng must be random.Random")
        values = {}
        for slot in self.slots:
            module_ids = [module.module_id for module in slot.modules]
            values[slot.slot_id] = tuple(rng.sample(module_ids, slot.capacity))
        return self.validate_set_position(SetPosition.create(values))

    def random_position_from_set(self, position: SetPosition, rng) -> ParticlePosition:
        """Decode one module per slot from a validated set position."""
        import random

        if not isinstance(rng, random.Random):
            raise TypeError("rng must be random.Random")
        position = self.validate_set_position(position)
        choices = []
        for slot in self.slots:
            module_id = rng.choice(position.modules(slot.slot_id))
            module = self.module(slot.slot_id, module_id)
            choices.append(ModuleChoice.create(slot.slot_id, module_id, module.default_params()))
        return self.validate_position(ParticlePosition(tuple(choices)))

    def singleton_set_position(self, position: ParticlePosition) -> SetPosition:
        """Adapt a decoded program to a crisp set for old checkpoints/callers."""
        position = self.validate_position(position)
        return self.validate_set_position(SetPosition.create({
            slot_id: (position.as_mapping()[slot_id].module,)
            for slot_id in self.slot_ids
        }))

    def describe(self) -> str:
        lines = []
        for slot in self.slots:
            lines.append(f"[{slot.slot_id}] {slot.description}")
            for module in slot.modules:
                params = ", ".join(p.name for p in module.parameters) or "none"
                lines.append(f"- {module.module_id}: {module.description}; params={params}")
        return "\n".join(lines)
