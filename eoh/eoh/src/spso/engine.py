"""Concurrent semantic PSO engine.

Candidate generation and evaluation are pipelined like the original EoH
runner: sampler threads perform LLM-side work and a shared evaluation pool
runs the expensive task evaluations.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .compiler import HeuristicCompiler
from .models import ModuleChoice, ParticlePosition
from .registry import ModuleRegistry
from .selector import SelectionResult, SemanticSelector
from .velocity import cut_sets, initial_velocity, update_velocity

logger = logging.getLogger("spso")


@dataclass
class SPSOConfig:
    population_size: int = 10
    generations: int = 3
    initial_samples: int | None = None
    max_evaluations: int | None = None
    alpha: float = 0.5
    omega: float = 0.7
    c_personal: float = 1.4
    c_global: float = 1.8
    c_explore: float = 0.25
    seed: int = 2024
    checkpoint_every: int = 1
    output_dir: str = "./spso_results"
    use_llm: bool = True
    resume_from: str | None = None
    num_samplers: int = 16
    num_evaluators: int = 16

    def __post_init__(self):
        if self.population_size < 1 or self.generations < 0:
            raise ValueError("population_size must be positive and generations cannot be negative")
        if self.initial_samples is not None and self.initial_samples < self.population_size:
            raise ValueError("initial_samples must be at least population_size")
        if self.max_evaluations is not None and self.max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if self.num_samplers < 1 or self.num_evaluators < 1:
            raise ValueError("num_samplers and num_evaluators must be positive")


@dataclass
class EvaluatedCandidate:
    position: ParticlePosition
    objective: float | None
    code: str | None
    algorithm: str | None
    source: str
    strategy: str
    sample_id: int
    cached: bool = False

    def to_dict(self, slot_order: tuple[str, ...]) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "position": self.position.to_dict(),
            "position_text": self.position.algorithm_text(slot_order),
            "objective": self.objective,
            "code": self.code,
            "algorithm": self.algorithm,
            "source": self.source,
            "strategy": self.strategy,
            "cached": self.cached,
        }


@dataclass
class _Particle:
    position: ParticlePosition
    velocity: dict[str, dict[str, float]]
    pbest_position: ParticlePosition | None = None
    pbest_objective: float | None = None
    current_objective: float | None = None


def _tupleize(value):
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_tupleize(item) for item in value)
    return value


class SPSOEngine:
    """Typed semantic PSO with pluggable compilation and evaluation.

    ``evaluate_code`` must return a finite scalar objective or ``None``.  The
    compiler is responsible for producing a complete function that satisfies
    the task adapter's public input/output contract.
    """

    def __init__(
        self,
        registry: ModuleRegistry,
        compiler: HeuristicCompiler,
        evaluate_code: Callable[[str], float | None],
        config: SPSOConfig | None = None,
        selector: SemanticSelector | None = None,
    ):
        self.registry = registry
        self.compiler = compiler
        self.evaluate_code = evaluate_code
        self.config = config or SPSOConfig()
        self.rng = random.Random(self.config.seed)
        self.selector = selector or SemanticSelector(registry)
        self.output_dir = Path(self.config.output_dir)
        self.sample_dir = self.output_dir / "samples"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_count = 0
        self.evaluation_count = 0
        self.cache: dict[str, dict[str, Any]] = {}
        self.records: list[dict[str, Any]] = []
        self.gbest_position: ParticlePosition | None = None
        self.gbest_objective: float | None = None
        self._state_lock = threading.RLock()
        self._inflight: dict[str, threading.Event] = {}
        self._sampler_executor: ThreadPoolExecutor | None = None
        self._eval_executor: ThreadPoolExecutor | None = None

    def _key(self, position: ParticlePosition) -> str:
        return json.dumps(position.to_dict(), sort_keys=True, separators=(",", ":"))

    def _strategy_mutation(self, position: ParticlePosition, strategy: str, rng=None) -> ParticlePosition:
        """Apply one validated A/B/C semantic modification."""
        rng = rng or self.rng
        if strategy == "B":
            return position
        mapping = position.as_mapping()
        slot = rng.choice(self.registry.slots)
        current = mapping[slot.slot_id]
        spec = self.registry.module(slot.slot_id, current.module)
        if strategy == "A":
            target = next(
                (module for module in slot.modules if module.module_id == spec.reverse_of),
                None,
            )
            if target is None:
                target = rng.choice(slot.modules)
            mapping[slot.slot_id] = ModuleChoice.create(
                slot.slot_id, target.module_id, target.default_params()
            )
        elif strategy == "C":
            similar_ids = set(spec.similar_to)
            similar = [module for module in slot.modules if module.module_id in similar_ids]
            if similar:
                target = rng.choice(similar)
                mapping[slot.slot_id] = ModuleChoice.create(
                    slot.slot_id, target.module_id, target.default_params()
                )
            elif spec.parameters:
                # Parameter-neighbour search is delegated to task parameter grids.
                params = spec.default_params()
                parameter = rng.choice(spec.parameters)
                if parameter.kind in {"float", "int"} and parameter.minimum is not None and parameter.maximum is not None:
                    params[parameter.name] = (parameter.minimum + parameter.maximum) / 2
                    if parameter.kind == "int":
                        params[parameter.name] = int(round(params[parameter.name]))
                mapping[slot.slot_id] = ModuleChoice.create(slot.slot_id, current.module, params)
        return self.registry.validate_position(ParticlePosition(tuple(mapping.values())))

    def _run_evaluation(self, code: str) -> float | None:
        """Run one independent evaluation in the shared evaluation pool."""
        if self._eval_executor is None:
            return self.evaluate_code(code)
        return self._eval_executor.submit(self.evaluate_code, code).result()

    def _evaluate(self, position: ParticlePosition, source: str, strategy: str) -> EvaluatedCandidate:
        """Evaluate one candidate, safely callable from many sampler threads.

        Duplicate positions share one in-flight evaluation. Every request is
        still recorded as a sample, while only the first request consumes an
        independent evaluation slot.
        """
        position = self.registry.validate_position(position)
        key = self._key(position)
        code, algorithm = self.compiler.compile(position)

        with self._state_lock:
            self.sample_count += 1
            sample_id = self.sample_count
            cached = self.cache.get(key)
            budget_exhausted = False
            if cached is not None:
                owner = False
                wait_event = None
            else:
                wait_event = self._inflight.get(key)
                owner = wait_event is None
                if owner:
                    if (
                        self.config.max_evaluations is not None
                        and self.evaluation_count >= self.config.max_evaluations
                    ):
                        budget_exhausted = True
                    else:
                        wait_event = threading.Event()
                        self._inflight[key] = wait_event
                        self.evaluation_count += 1

        if cached is not None:
            candidate = EvaluatedCandidate(
                position, cached["objective"], cached["code"], cached["algorithm"],
                source, strategy, sample_id, True,
            )
            self._record(candidate)
            return candidate

        if budget_exhausted:
            candidate = EvaluatedCandidate(
                position, None, code, algorithm, source, strategy, sample_id, False,
            )
            self._record(candidate)
            return candidate

        if not owner:
            # Another sampler is already evaluating this exact position.
            wait_event.wait()
            with self._state_lock:
                cached = self.cache[key]
            candidate = EvaluatedCandidate(
                position, cached["objective"], cached["code"], cached["algorithm"],
                source, strategy, sample_id, True,
            )
            self._record(candidate)
            return candidate

        try:
            objective = self._run_evaluation(code)
            objective = float(objective) if objective is not None else None
        except Exception as exc:
            logger.warning("candidate evaluation failed: %s", exc)
            objective = None
        if objective is not None and (objective != objective or objective in (float("inf"), float("-inf"))):
            objective = None

        with self._state_lock:
            self.cache[key] = {"objective": objective, "code": code, "algorithm": algorithm}
            wait_event.set()
            self._inflight.pop(key, None)
        candidate = EvaluatedCandidate(
            position, objective, code, algorithm, source, strategy, sample_id, False,
        )
        self._record(candidate)
        return candidate

    def _record(self, candidate: EvaluatedCandidate):
        record = candidate.to_dict(self.registry.slot_ids)
        with self._state_lock:
            self.records.append(record)
            with (self.sample_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if candidate.objective is not None and (
                self.gbest_objective is None or candidate.objective < self.gbest_objective
            ):
                self.gbest_objective = candidate.objective
                self.gbest_position = candidate.position
                with (self.sample_dir / "best.json").open("w", encoding="utf-8") as handle:
                    json.dump(record, handle, ensure_ascii=False, indent=2)

    def _initialise(self) -> list[_Particle]:
        count = self.config.initial_samples or 2 * self.config.population_size
        # Draw positions in the coordinator thread so seeded runs do not race
        # on the shared RNG. Compile/evaluate work is parallelized below.
        positions = [self.registry.random_position(self.rng) for _ in range(count)]
        if self._sampler_executor is None:
            candidates = [self._evaluate(position, "initial", "B") for position in positions]
        else:
            futures = [
                self._sampler_executor.submit(self._evaluate, position, "initial", "B")
                for position in positions
            ]
            candidates = [future.result() for future in futures]
        valid = [candidate for candidate in candidates if candidate.objective is not None]
        if len(valid) < self.config.population_size:
            raise RuntimeError("not enough valid initial candidates to form a population")
        valid.sort(key=lambda candidate: candidate.objective)
        particles = []
        for candidate in valid[: self.config.population_size]:
            particles.append(
                _Particle(
                    candidate.position,
                    initial_velocity(self.registry),
                    candidate.position,
                    candidate.objective,
                    candidate.objective,
                )
            )
        return particles

    def _produce_evolution_candidate(self, plan) -> EvaluatedCandidate:
        """Sampler-side LLM selection followed by shared-pool evaluation."""
        _particle, current, pbest, gbest, cuts, objectives, velocity, seed = plan
        selection: SelectionResult = self.selector.select(
            cuts, current, pbest, gbest, objectives, velocity
        )
        # Sampler threads must not share the coordinator's seeded RNG.
        local_rng = random.Random(seed)
        proposal = self._strategy_mutation(selection.position, selection.strategy, local_rng)
        return self._evaluate(proposal, selection.source, selection.strategy)

    def _update_global(self, particles: list[_Particle]):
        best = min(
            (particle for particle in particles if particle.pbest_objective is not None),
            key=lambda particle: particle.pbest_objective,
            default=None,
        )
        if best is not None and (
            self.gbest_objective is None or best.pbest_objective < self.gbest_objective
        ):
            self.gbest_objective = best.pbest_objective
            self.gbest_position = best.pbest_position

    def _checkpoint(self, particles: list[_Particle], generation: int):
        payload = {
            "generation": generation,
            "sample_count": self.sample_count,
            "evaluation_count": self.evaluation_count,
            "gbest_objective": self.gbest_objective,
            "gbest_position": self.gbest_position.to_dict() if self.gbest_position else None,
            "particles": [
                {
                    "position": particle.position.to_dict(),
                    "velocity": particle.velocity,
                    "pbest_position": particle.pbest_position.to_dict() if particle.pbest_position else None,
                    "pbest_objective": particle.pbest_objective,
                    "current_objective": particle.current_objective,
                }
                for particle in particles
            ],
            "cache": self.cache,
            "rng_state": self.rng.getstate(),
            "config": self.config.__dict__,
        }
        path = self.checkpoint_dir / f"generation_{generation:04d}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=list)

    def _restore(self, path: str | os.PathLike[str]) -> tuple[list[_Particle], int]:
        """Restore all search state needed for a deterministic continuation."""
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        particles = []
        for raw in payload["particles"]:
            pbest = raw.get("pbest_position")
            particles.append(
                _Particle(
                    self.registry.validate_position(ParticlePosition.from_dict(raw["position"])),
                    raw["velocity"],
                    self.registry.validate_position(ParticlePosition.from_dict(pbest)) if pbest else None,
                    raw.get("pbest_objective"),
                    raw.get("current_objective"),
                )
            )
        self.sample_count = int(payload.get("sample_count", 0))
        self.evaluation_count = int(payload.get("evaluation_count", 0))
        self.cache = payload.get("cache", {})
        raw_gbest = payload.get("gbest_position")
        self.gbest_position = (
            self.registry.validate_position(ParticlePosition.from_dict(raw_gbest))
            if raw_gbest else None
        )
        self.gbest_objective = payload.get("gbest_objective")
        if "rng_state" in payload:
            self.rng.setstate(_tupleize(payload["rng_state"]))
        return particles, int(payload.get("generation", 0))

    def run(self) -> dict[str, Any]:
        started = time.time()
        self._sampler_executor = ThreadPoolExecutor(
            max_workers=self.config.num_samplers,
            thread_name_prefix="spso-sampler",
        )
        self._eval_executor = ThreadPoolExecutor(
            max_workers=self.config.num_evaluators,
            thread_name_prefix="spso-eval",
        )
        try:
            if self.config.resume_from:
                particles, start_generation = self._restore(self.config.resume_from)
            else:
                particles = self._initialise()
                self._update_global(particles)
                start_generation = 0
                self._checkpoint(particles, 0)
            budget = self.config.max_evaluations
            for generation in range(start_generation + 1, self.config.generations + 1):
                if budget is not None and self.evaluation_count >= budget:
                    break
                snapshot = self.gbest_position
                if snapshot is None:
                    raise RuntimeError("gbest is undefined after initialisation")

                # Update all categorical velocities in the coordinator thread,
                # then freeze one generation snapshot before starting workers.
                plans = []
                for particle in particles:
                    if particle.pbest_position is None:
                        continue
                    particle.velocity = update_velocity(
                        particle.velocity,
                        particle.position,
                        particle.pbest_position,
                        snapshot,
                        self.registry,
                        self.rng,
                        self.config.omega,
                        self.config.c_personal,
                        self.config.c_global,
                        self.config.c_explore,
                    )
                    cuts = cut_sets(
                        particle.velocity,
                        particle.position,
                        particle.pbest_position,
                        snapshot,
                        self.registry,
                        self.config.alpha,
                    )
                    objectives = {
                        "current": particle.current_objective,
                        "pbest": particle.pbest_objective,
                        "gbest": self.gbest_objective,
                    }
                    plans.append((
                        particle,
                        particle.position,
                        particle.pbest_position,
                        snapshot,
                        cuts,
                        objectives,
                        particle.velocity,
                        self.rng.getrandbits(64),
                    ))

                futures = [
                    self._sampler_executor.submit(self._produce_evolution_candidate, plan)
                    for plan in plans
                ]
                candidates = [future.result() for future in futures]
                for plan, candidate in zip(plans, candidates):
                    particle = plan[0]
                    if candidate.objective is not None:
                        particle.position = candidate.position
                        particle.current_objective = candidate.objective
                        if particle.pbest_objective is None or candidate.objective < particle.pbest_objective:
                            particle.pbest_position = candidate.position
                            particle.pbest_objective = candidate.objective
                self._update_global(particles)
                if generation % max(1, self.config.checkpoint_every) == 0:
                    self._checkpoint(particles, generation)
        finally:
            # Samplers may be waiting for evaluation futures, so close them
            # first while the evaluation pool is still available.
            if self._sampler_executor is not None:
                self._sampler_executor.shutdown(wait=True)
                self._sampler_executor = None
            if self._eval_executor is not None:
                self._eval_executor.shutdown(wait=True)
                self._eval_executor = None
        elapsed = time.time() - started
        summary = {
            "best_objective": self.gbest_objective,
            "best_position": self.gbest_position.to_dict() if self.gbest_position else None,
            "best_position_text": self.gbest_position.algorithm_text(self.registry.slot_ids) if self.gbest_position else None,
            "samples": self.sample_count,
            "independent_evaluations": self.evaluation_count,
            "cache_entries": len(self.cache),
            "elapsed_seconds": elapsed,
        }
        with (self.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        return summary
