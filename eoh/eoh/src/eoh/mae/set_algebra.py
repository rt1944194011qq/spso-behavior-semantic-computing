from __future__ import annotations

from collections.abc import Iterable

WeightedSet = dict[str, float]


def clip_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def scalar_multiply(weighted_set: WeightedSet, coefficient: float) -> WeightedSet:
    """Multiply all membership probabilities and clip them to [0, 1]."""
    return {
        key: clip_probability(probability * coefficient)
        for key, probability in weighted_set.items()
        if probability > 0 and coefficient > 0
    }


def set_add(*sets: WeightedSet) -> WeightedSet:
    """Union weighted sets; identical elements keep the greatest probability."""
    result: WeightedSet = {}
    for weighted_set in sets:
        for key, probability in weighted_set.items():
            result[key] = max(result.get(key, 0.0), clip_probability(probability))
    return result


def set_difference(left: Iterable[str], right: Iterable[str]) -> set[str]:
    return set(left) - set(right)


def weighted_difference(
    left: Iterable[str], right: Iterable[str], probability: float
) -> WeightedSet:
    value = clip_probability(probability)
    return {key: value for key in set_difference(left, right)}


def update_velocity(
    old_velocity: WeightedSet,
    current_ids: Iterable[str],
    personal_id: str | None,
    global_id: str | None,
    left_id: str | None,
    right_id: str | None,
    *,
    omega: float,
    personal_weight: float,
    global_weight: float,
    left_weight: float,
    right_weight: float,
    random_values: tuple[float, float, float, float],
) -> WeightedSet:
    current = set(current_ids)
    rp, rg, rl, rr = random_values

    def contribution(item_id: str | None, value: float) -> WeightedSet:
        if item_id is None or item_id in current:
            return {}
        return {item_id: clip_probability(value)}

    return set_add(
        scalar_multiply(old_velocity, omega),
        contribution(personal_id, personal_weight * rp),
        contribution(global_id, global_weight * rg),
        contribution(left_id, left_weight * rl),
        contribution(right_id, right_weight * rr),
    )
