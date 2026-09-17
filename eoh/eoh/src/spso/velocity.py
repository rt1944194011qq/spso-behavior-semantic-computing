"""Set-based PSO velocity updates and alpha-cut position construction.

The implementation follows the possibility-set operators from the supplied
set-based PSO paper. A position is a crisp set of module IDs; a velocity is a
mapping from module IDs to possibility values in ``[0, 1]``. The compiler does
not consume either object directly: the engine decodes the constructed set
through the semantic selector into a one-module-per-slot ``ParticlePosition``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import ParticlePosition, SetPosition
from .registry import ModuleRegistry


Velocity = dict[str, dict[str, float]]


def _as_set_position(
    position: SetPosition | ParticlePosition,
    registry: ModuleRegistry,
) -> SetPosition:
    if isinstance(position, SetPosition):
        return registry.validate_set_position(position)
    if isinstance(position, ParticlePosition):
        return registry.singleton_set_position(position)
    raise TypeError("position must be SetPosition or ParticlePosition")


def _possibility(value: Any) -> float:
    """Return a finite possibility value clipped to the paper's [0, 1]."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, value))


def _scaled_possibility(coefficient: float, random_value: float) -> float:
    """Implement the paper's ``min(1, c*r)`` possibility scaling."""
    return _possibility(float(coefficient) * float(random_value))


def initial_velocity(registry: ModuleRegistry, value: float = 0.0) -> Velocity:
    """Create a velocity over every module in every slot."""
    value = _possibility(value)
    return {
        slot.slot_id: {module.module_id: value for module in slot.modules}
        for slot in registry.slots
    }


def set_difference(left: SetPosition, right: SetPosition) -> dict[str, set[str]]:
    """Return the slot-wise crisp set difference ``left - right``."""
    right_map = right.as_mapping()
    return {
        slot_id: set(module_ids) - set(right_map[slot_id])
        for slot_id, module_ids in left.slots
    }


def update_velocity(
    old_velocity: Mapping[str, Mapping[str, float]],
    current: SetPosition | ParticlePosition,
    pbest: SetPosition | ParticlePosition,
    gbest: SetPosition | ParticlePosition,
    registry: ModuleRegistry,
    rng,
    omega: float,
    c_personal: float,
    c_global: float,
    c_explore: float = 0.0,
) -> Velocity:
    """Apply the set-based possibility velocity update.

    For slot ``s`` and module ``m`` this is

    ``max(min(1, omega*v),
         min(1, c1*r1) if m in P_s-X_s else 0,
         min(1, c2*r2) if m in G_s-X_s else 0)``.

    ``c_explore`` remains accepted for compatibility with older callers, but
    is intentionally ignored. Random ``r1/r2`` and the alpha-cut/position
    construction stage provide the paper-defined stochasticity.
    """
    del c_explore
    current = _as_set_position(current, registry)
    pbest = _as_set_position(pbest, registry)
    gbest = _as_set_position(gbest, registry)
    current_map = current.as_mapping()
    pbest_map = pbest.as_mapping()
    gbest_map = gbest.as_mapping()

    result: Velocity = {}
    for slot in registry.slots:
        slot_id = slot.slot_id
        old = old_velocity.get(slot_id, {})
        current_set = set(current_map[slot_id])
        pbest_delta = set(pbest_map[slot_id]) - current_set
        gbest_delta = set(gbest_map[slot_id]) - current_set
        slot_result: dict[str, float] = {}
        for module in slot.modules:
            module_id = module.module_id
            terms = [_possibility(float(omega) * float(old.get(module_id, 0.0)))]
            if module_id in pbest_delta:
                terms.append(_scaled_possibility(c_personal, rng.random()))
            if module_id in gbest_delta:
                terms.append(_scaled_possibility(c_global, rng.random()))
            slot_result[module_id] = max(terms)
        result[slot_id] = slot_result
    return result


def alpha_cut_sets(
    velocity: Mapping[str, Mapping[str, float]],
    registry: ModuleRegistry,
    alpha: float,
) -> dict[str, list[str]]:
    """Return the paper's crisp alpha-cut ``{m | v[m] >= alpha}``."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    result: dict[str, list[str]] = {}
    for slot in registry.slots:
        scores = velocity.get(slot.slot_id, {})
        selected = [
            module.module_id
            for module in slot.modules
            if _possibility(scores.get(module.module_id, 0.0)) >= alpha
        ]
        selected.sort(
            key=lambda module_id: (-_possibility(scores.get(module_id, 0.0)), module_id)
        )
        result[slot.slot_id] = selected[: slot.max_cut]
    return result


def cut_sets(
    velocity: Mapping[str, Mapping[str, float]],
    current=None,
    pbest=None,
    gbest=None,
    registry: ModuleRegistry | None = None,
    alpha: float = 0.5,
) -> dict[str, list[str]]:
    """Compatibility wrapper for callers that used the old ``cut_sets`` name.

    The old implementation injected current/pbest/gbest anchors into the cut
    and used a relative-to-maximum threshold. The set-based rule does neither;
    anchors are handled explicitly by :func:`construct_set_position`.
    """
    del current, pbest, gbest
    if registry is None:
        raise TypeError("registry is required")
    return alpha_cut_sets(velocity, registry, alpha)


def construct_set_position(
    raw_cut: Mapping[str, list[str]],
    current: SetPosition | ParticlePosition,
    registry: ModuleRegistry,
    rng,
) -> SetPosition:
    """Construct the next crisp position from an alpha-cut.

    For each slot, candidates from the alpha-cut are preferred, the current
    set is then retained, and remaining modules are sampled if the configured
    capacity ``K_s`` has not been reached.
    """
    current = _as_set_position(current, registry)
    current_map = current.as_mapping()
    values: dict[str, tuple[str, ...]] = {}
    for slot in registry.slots:
        slot_id = slot.slot_id
        allowed = {module.module_id for module in slot.modules}
        selected: list[str] = []

        def add(module_id: str):
            if module_id in allowed and module_id not in selected:
                selected.append(module_id)

        for module_id in raw_cut.get(slot_id, []):
            add(module_id)
        for module_id in current_map[slot_id]:
            add(module_id)

        remaining = [
            module.module_id for module in slot.modules if module.module_id not in selected
        ]
        rng.shuffle(remaining)
        for module_id in remaining:
            if len(selected) >= slot.capacity:
                break
            add(module_id)

        if not selected:
            raise ValueError(f"cannot construct a position for slot {slot_id!r}")
        values[slot_id] = tuple(selected[: slot.capacity])
    return registry.validate_set_position(SetPosition.create(values))
