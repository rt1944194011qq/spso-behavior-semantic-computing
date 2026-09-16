"""Categorical velocity updates and cut-set construction."""

from __future__ import annotations

from collections.abc import Mapping

from .models import ParticlePosition
from .registry import ModuleRegistry


def initial_velocity(registry: ModuleRegistry, value: float = 0.0) -> dict[str, dict[str, float]]:
    return {
        slot.slot_id: {module.module_id: float(value) for module in slot.modules}
        for slot in registry.slots
    }


def update_velocity(
    old_velocity: Mapping[str, Mapping[str, float]],
    current: ParticlePosition,
    pbest: ParticlePosition,
    gbest: ParticlePosition,
    registry: ModuleRegistry,
    rng,
    omega: float,
    c_personal: float,
    c_global: float,
    c_explore: float,
) -> dict[str, dict[str, float]]:
    """Update preference scores for categorical module choices.

    A module receives attraction when it appears in pbest or gbest.  The
    exploration term is deterministic under the engine's seeded RNG.
    """
    current_map = current.as_mapping()
    pbest_map = pbest.as_mapping()
    gbest_map = gbest.as_mapping()
    result: dict[str, dict[str, float]] = {}
    for slot in registry.slots:
        old = old_velocity.get(slot.slot_id, {})
        result[slot.slot_id] = {}
        for module in slot.modules:
            module_id = module.module_id
            score = omega * float(old.get(module_id, 0.0))
            if pbest_map[slot.slot_id].module == module_id:
                score += c_personal
            if gbest_map[slot.slot_id].module == module_id:
                score += c_global
            if current_map[slot.slot_id].module == module_id:
                score += 0.1 * c_personal
            score += c_explore * rng.random()
            result[slot.slot_id][module_id] = score
    return result


def cut_sets(
    velocity: Mapping[str, Mapping[str, float]],
    current: ParticlePosition,
    pbest: ParticlePosition,
    gbest: ParticlePosition,
    registry: ModuleRegistry,
    alpha: float,
) -> dict[str, list[str]]:
    """Keep high-velocity candidates and always retain the three anchors."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    anchors = (current.as_mapping(), pbest.as_mapping(), gbest.as_mapping())
    result: dict[str, list[str]] = {}
    for slot in registry.slots:
        scores = dict(velocity.get(slot.slot_id, {}))
        if not scores:
            scores = {module.module_id: 0.0 for module in slot.modules}
        maximum = max(scores.values())
        threshold = alpha * maximum if maximum > 0 else maximum
        selected = [module_id for module_id, score in scores.items() if score >= threshold]
        for mapping in anchors:
            module_id = mapping[slot.slot_id].module
            if module_id not in selected:
                selected.append(module_id)
        ordered = sorted(selected, key=lambda module_id: (-scores.get(module_id, 0.0), module_id))
        result[slot.slot_id] = ordered[: slot.max_cut]
    return result
