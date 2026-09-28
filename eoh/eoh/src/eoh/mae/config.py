from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class MAEConfig:
    """Configuration for synchronous multi-agent heuristic evolution."""

    num_agents: int = 10
    position_size: int = 5
    generations: int = 20
    initial_multiplier: int = 2
    seed: int = 2024

    alpha: float = 0.5
    max_cut_size: int = 10
    omega: float = 0.50
    neighbor_left: float = 0.40
    neighbor_right: float = 0.40
    h1_personal: float = 0.90
    h1_global: float = 0.25
    h2_personal: float = 0.25
    h2_global: float = 0.90

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
        if self.position_size < 2:
            raise ValueError("position_size must be >= 2")
        if self.generations < 0:
            raise ValueError("generations must be >= 0")
        if self.initial_multiplier < 1:
            raise ValueError("initial_multiplier must be >= 1")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.max_cut_size < self.position_size:
            raise ValueError("max_cut_size must be >= position_size")
        if self.selection_size < 2:
            raise ValueError("selection_size must be >= 2")
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
        data["variation_operators"] = list(self.variation_operators)
        return data
