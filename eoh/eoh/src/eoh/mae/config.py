from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class MAEConfig:
    """Configuration for synchronous multi-agent heuristic evolution."""

    num_agents: int = 10
    # The semantic-memory engine keeps exactly one complete program per agent.
    # LegacyMAEEngine can still be given a larger position_size explicitly.
    position_size: int = 1
    generations: int = 20
    initial_multiplier: int = 10
    seed: int = 2024
    objective_direction: str = "min"

    alpha: float = 0.5
    max_cut_size: int = 10
    omega: float = 0.50
    neighbor_left: float = 0.40
    neighbor_right: float = 0.40
    h1_personal: float = 0.90
    h1_global: float = 0.25
    h2_personal: float = 0.25
    h2_global: float = 0.90

    # Semantic-memory MA-EoH v2. Probabilities below are *stored* memory
    # probabilities; H1/H2 boosts only affect the current cut draw.
    experience_birth_min: float = 0.20
    experience_birth_max: float = 0.60
    experience_target_weight: float = 0.60
    experience_half_life: float = 4.0
    experience_drop_below: float = 0.05
    experience_capacity: int = 24
    cut_target_size: int = 4
    cut_min_size: int = 2
    neighbor_reserved_slots: int = 1
    h1_selection_boost: float = 1.5
    h2_selection_boost: float = 1.5
    enabled_operators: tuple[str, ...] = ("H1", "H2", "H3", "H4", "H5")
    variation_parent_counts: dict[str, int] | None = None

    variation_operators: tuple[str, ...] = ("H3", "H4", "H5")
    selection_size: int = 5
    llm_retries: int = 3
    fill_retries: int = 2
    num_llm_workers: int = 8
    num_evaluators: int = 8
    isolate_evaluation: bool = True

    require_description: bool = True
    require_symmetric_output: bool = True
    require_nonnegative_output: bool = True
    output_dir: str = "./results_mae/run_001"
    resume_from: str | None = None
    evaluation_fingerprint: str = "default"
    debug: bool = False
    detailed_logging: bool = True
    progress_log_every: int = 10
    heartbeat_interval_seconds: float = 30.0
    record_full_prompts: bool = True
    record_full_responses: bool = True

    def __post_init__(self) -> None:
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        if self.position_size < 1:
            raise ValueError("position_size must be >= 1")
        if self.generations < 0:
            raise ValueError("generations must be >= 0")
        if self.objective_direction not in {"min", "max"}:
            raise ValueError("objective_direction must be 'min' or 'max'")
        if self.initial_multiplier < 1:
            raise ValueError("initial_multiplier must be >= 1")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.max_cut_size < self.position_size:
            raise ValueError("max_cut_size must be >= position_size")
        if self.selection_size < 2:
            raise ValueError("selection_size must be >= 2")
        if not 0 <= self.experience_birth_min <= self.experience_birth_max <= 1:
            raise ValueError("experience birth probabilities must be ordered within [0, 1]")
        if not 0 <= self.experience_target_weight <= 1:
            raise ValueError("experience_target_weight must be in [0, 1]")
        if self.experience_half_life <= 0:
            raise ValueError("experience_half_life must be positive")
        if not 0 <= self.experience_drop_below < 1:
            raise ValueError("experience_drop_below must be in [0, 1)")
        if self.experience_capacity < 1 or self.cut_target_size < 1:
            raise ValueError("experience_capacity and cut_target_size must be positive")
        if not 1 <= self.cut_min_size <= self.cut_target_size <= self.experience_capacity:
            raise ValueError("cut sizes must satisfy 1 <= min <= target <= capacity")
        if not 0 <= self.neighbor_reserved_slots < self.cut_target_size:
            raise ValueError("neighbor_reserved_slots must be below cut_target_size")
        if self.h1_selection_boost < 1 or self.h2_selection_boost < 1:
            raise ValueError("H1/H2 selection boosts must be >= 1")
        if len(set(self.enabled_operators)) != len(self.enabled_operators):
            raise ValueError("enabled_operators must not contain duplicates")
        if "H0" in self.enabled_operators:
            raise ValueError("H0 is an automatic initializer/fallback, not an enabled operator")
        if self.variation_parent_counts is not None:
            if any(count < 1 for count in self.variation_parent_counts.values()):
                raise ValueError("variation parent counts must be positive")
        missing_arity = set(self.enabled_operators) - {"H1", "H2", "H3", "H4", "H5"}
        missing_arity -= set(self.variation_parent_counts or {})
        if missing_arity:
            raise ValueError(
                f"new variation operators need parent counts: {sorted(missing_arity)}"
            )
        if self.num_llm_workers < 1 or self.num_evaluators < 1:
            raise ValueError("worker counts must be >= 1")
        if self.progress_log_every < 1:
            raise ValueError("progress_log_every must be >= 1")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be > 0")
        allowed = {"H3", "H4", "H5"}
        unknown = set(self.variation_operators) - allowed
        if unknown:
            raise ValueError(f"unknown variation operators: {sorted(unknown)}")
        for name in (
            "omega", "neighbor_left", "neighbor_right", "h1_personal",
            "h1_global", "h2_personal", "h2_global",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).resolve()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["enabled_operators"] = list(self.enabled_operators)
        if self.position_size == 1:
            for legacy_only in (
                "alpha", "max_cut_size", "omega", "neighbor_left",
                "neighbor_right", "h1_personal", "h1_global",
                "h2_personal", "h2_global", "variation_operators",
                "selection_size",
            ):
                data.pop(legacy_only)
            data["algorithm_version"] = "semantic-memory-v2"
        else:
            data["variation_operators"] = list(self.variation_operators)
            data["algorithm_version"] = "legacy-multi-position-v1"
        return data
