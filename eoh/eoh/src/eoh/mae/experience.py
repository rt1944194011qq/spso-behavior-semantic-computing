"""Persistent semantic experience and bounded, probability-based cut sampling."""

from __future__ import annotations

import math
import random
from dataclasses import replace

from .models import ExperienceCutItem, ExperienceRecord


SOURCE_ROLES = ("personal", "global", "left", "right")


def clip_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def birth_probability(
    source_fitness: float,
    target_fitness: float,
    best_fitness: float,
    scale: float,
    minimum: float,
    maximum: float,
    target_weight: float = 0.60,
) -> float:
    """Set memory birth probability from both ends of a semantic difference.

    Fitness values are already converted so that smaller means better.  Each
    endpoint is measured relative to the best ranked fitness using the same
    generation-level robust scale.  The target receives slightly more weight
    because the stored prompt directs the source program toward that target.
    """
    values = (source_fitness, target_fitness, best_fitness, scale, target_weight)
    if scale <= 0 or not all(math.isfinite(value) for value in values):
        raise ValueError("finite fitness values, weight, and positive scale are required")
    if not 0.0 <= target_weight <= 1.0:
        raise ValueError("target_weight must be in [0, 1]")

    source_quality = math.exp(
        -max(0.0, source_fitness - best_fitness) / scale
    )
    target_quality = math.exp(
        -max(0.0, target_fitness - best_fitness) / scale
    )
    pair_quality = (
        (1.0 - target_weight) * source_quality
        + target_weight * target_quality
    )
    return clip_probability(minimum + (maximum - minimum) * pair_quality)


def decay_and_prune(
    velocity: dict[str, ExperienceRecord],
    *,
    half_life: float,
    drop_below: float,
    capacity: int,
) -> dict[str, ExperienceRecord]:
    """Decay only previously stored memories; leave new entries for later merge."""
    retention = 2.0 ** (-1.0 / half_life)
    kept = {
        key: replace(item, probability=clip_probability(item.probability * retention))
        for key, item in velocity.items()
        if item.probability * retention >= drop_below
    }
    return prune_velocity(kept, capacity)


def prune_velocity(
    velocity: dict[str, ExperienceRecord], capacity: int
) -> dict[str, ExperienceRecord]:
    if len(velocity) <= capacity:
        return velocity

    # Avoid a single source crowding all other sources out of the memory bank.
    ranked = sorted(
        velocity.values(),
        key=lambda item: (-item.probability, -item.last_seen_generation, item.experience_id),
    )
    chosen: list[ExperienceRecord] = []
    for role in SOURCE_ROLES:
        match = next((item for item in ranked if item.source_role == role), None)
        if match is not None and len(chosen) < capacity:
            chosen.append(match)
    for item in ranked:
        if len(chosen) >= capacity:
            break
        if item not in chosen:
            chosen.append(item)
    return {item.experience_id: item for item in chosen}


def calibrated_multiplier(weights: list[float], target: int) -> float:
    """Solve sum(min(1, lambda * weight)) == target by bisection."""
    if target <= 0:
        return 0.0
    positive = [clip_probability(weight) for weight in weights if weight > 0]
    if len(positive) < target:
        raise ValueError("not enough positive-probability memories for target")
    lower, upper = 0.0, 1.0
    while sum(min(1.0, upper * weight) for weight in positive) < target:
        upper *= 2.0
    for _ in range(60):
        middle = (lower + upper) / 2.0
        if sum(min(1.0, middle * weight) for weight in positive) < target:
            lower = middle
        else:
            upper = middle
    return upper


def _weighted_pick(
    items: list[ExperienceRecord], weights: dict[str, float], rng: random.Random
) -> ExperienceRecord:
    total = sum(max(0.0, weights[item.experience_id]) for item in items)
    if total <= 0:
        return rng.choice(items)
    point = rng.random() * total
    for item in items:
        point -= max(0.0, weights[item.experience_id])
        if point <= 0:
            return item
    return items[-1]


def _weighted_sample(
    items: list[ExperienceRecord],
    count: int,
    weights: dict[str, float],
    rng: random.Random,
) -> list[ExperienceRecord]:
    remaining = list(items)
    result = []
    for _ in range(min(count, len(remaining))):
        choice = _weighted_pick(remaining, weights, rng)
        result.append(choice)
        remaining.remove(choice)
    return result


def sample_cut(
    velocity: dict[str, ExperienceRecord],
    *,
    target_size: int,
    neighbor_reserved_slots: int,
    generation: int,
    agent_id: int,
    enabled_operators: tuple[str, ...],
    h1_boost: float,
    h2_boost: float,
    rng: random.Random,
) -> tuple[list[ExperienceCutItem], dict]:
    """Bernoulli draw, then capacity repair; preserve neighbor participation."""
    entries = sorted(velocity.values(), key=lambda item: item.experience_id)
    if not entries:
        return [], {"lambda": 0.0, "eligible": 0, "target": 0, "reserved": []}
    target = min(target_size, len(entries))
    enabled = set(enabled_operators)
    boosted = {}
    for item in entries:
        beta = (
            h1_boost if item.source_role == "personal" and "H1" in enabled
            else h2_boost if item.source_role == "global" and "H2" in enabled
            else 1.0
        )
        # The boost is selection-only. It never mutates item.probability.
        boosted[item.experience_id] = clip_probability(item.probability * beta)

    reserved: list[ExperienceRecord] = []
    reserved_probabilities: dict[str, float] = {}
    preferred = ("left", "right") if (generation + agent_id) % 2 == 0 else ("right", "left")
    for role in preferred:
        if len(reserved) >= min(neighbor_reserved_slots, target):
            break
        choices = [item for item in entries if item.source_role == role]
        if choices:
            selected = _weighted_pick(choices, boosted, rng)
            reserved.append(selected)
            total_weight = sum(boosted[item.experience_id] for item in choices)
            reserved_probabilities[selected.experience_id] = (
                boosted[selected.experience_id] / total_weight
                if total_weight else 1.0 / len(choices)
            )

    remaining = [item for item in entries if item not in reserved]
    remaining_target = target - len(reserved)
    multiplier = calibrated_multiplier(
        [boosted[item.experience_id] for item in remaining], remaining_target
    ) if remaining_target else 0.0
    probabilities = {
        item.experience_id: clip_probability(multiplier * boosted[item.experience_id])
        for item in remaining
    }
    drawn = [
        item for item in remaining
        if rng.random() < probabilities[item.experience_id]
    ]
    if len(drawn) > remaining_target:
        drawn = _weighted_sample(drawn, remaining_target, probabilities, rng)
    elif len(drawn) < remaining_target:
        unselected = [item for item in remaining if item not in drawn]
        drawn += _weighted_sample(
            unselected, remaining_target - len(drawn), probabilities, rng
        )

    selected = reserved + drawn
    cut = [
        ExperienceCutItem(
            experience_id=item.experience_id,
            source_role=item.source_role,
            memory_probability=item.probability,
            boosted_probability=boosted[item.experience_id],
            selection_probability=(
                reserved_probabilities[item.experience_id]
                if item in reserved else probabilities[item.experience_id]
            ),
            reserved=item in reserved,
        )
        for item in selected
    ]
    return cut, {
        "lambda": multiplier,
        "eligible": len(entries),
        "target": target,
        "reserved": [item.experience_id for item in reserved],
        "boosted_probabilities": boosted,
        "selection_probabilities": probabilities,
    }
