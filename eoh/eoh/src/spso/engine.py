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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .compiler import HeuristicCompiler
from .models import ModuleChoice, ParticlePosition, SetPosition
from .registry import ModuleRegistry
from .selector import SelectionResult, SemanticSelector
from .velocity import (
    alpha_cut_sets,
    construct_set_position,
    initial_velocity,
    update_velocity,
)

logger = logging.getLogger("spso")


@dataclass
class SPSOConfig:
    population_size: int = 10
    generations: int = 3
    initial_samples: int | None = None
    max_evaluations: int | None = None
    alpha: float | None = None
    omega: float = 0.7
    c_personal: float = 1.4
    c_global: float = 1.8
    # Kept as a compatibility field for old configs. Set-based PSO does not
    # add this term to velocity; randomness comes from r1/r2 and alpha-cut.
    c_explore: float = 0.0
    seed: int = 2024
    checkpoint_every: int = 1
    output_dir: str = "./spso_results"
    use_llm: bool = True
    resume_from: str | None = None
    num_samplers: int = 16
    num_evaluators: int = 16
    cache_evaluations: bool = True
    modules_per_slot: int = 10
    # Core API compatibility mode. The paper runner explicitly passes p1..p4;
    # p0 keeps old direct-engine callers usable without an LLM service.
    heuristic_operators: tuple[str, ...] = ("p0",)
    use_module_descriptions: bool = True
    llm_model: str | None = None
    max_operator_retries: int = 2

    def __post_init__(self):
        if self.population_size < 1 or self.generations < 0:
            raise ValueError("population_size must be positive and generations cannot be negative")
        if self.initial_samples is not None and self.initial_samples < self.population_size:
            raise ValueError("initial_samples must be at least population_size")
        if self.max_evaluations is not None and self.max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        if self.alpha is not None and not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be None or in [0, 1]")
        if self.omega < 0 or self.c_personal < 0 or self.c_global < 0:
            raise ValueError("omega, c_personal and c_global must be nonnegative")
        if self.num_samplers < 1 or self.num_evaluators < 1:
            raise ValueError("num_samplers and num_evaluators must be positive")
        if self.modules_per_slot < 1:
            raise ValueError("modules_per_slot must be positive")
        operators = tuple(str(item).lower() for item in self.heuristic_operators)
        if not operators or len(set(operators)) != len(operators):
            raise ValueError("heuristic_operators must be nonempty and duplicate-free")
        if any(operator not in {"p0", "p1", "p2", "p3", "p4"} for operator in operators):
            raise ValueError("heuristic_operators must contain only p0..p4")
        self.heuristic_operators = operators
        if self.max_operator_retries < 0:
            raise ValueError("max_operator_retries must be nonnegative")


@dataclass
class EvaluatedCandidate:
    position: ParticlePosition
    set_position: SetPosition
    objective: float | None
    code: str | None
    algorithm: str | None
    source: str
    strategy: str
    sample_id: int
    cached: bool = False
    details: dict[str, Any] | None = None
    generation: int = 0
    particle_id: str | None = None
    parent_id: str | None = None
    operator: str = "p0"
    base_position: ParticlePosition | None = None
    prompt_ids: tuple[str, ...] = ()
    failure_reason: str | None = None

    def to_dict(self, slot_order: tuple[str, ...]) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "position": self.position.to_dict(),
            "set_position": self.set_position.to_dict(),
            "position_text": self.position.algorithm_text(slot_order),
            "objective": self.objective,
            "code": self.code,
            "algorithm": self.algorithm,
            "source": self.source,
            "strategy": self.strategy,
            "cached": self.cached,
            "details": self.details,
            "generation": self.generation,
            "particle_id": self.particle_id,
            "parent_id": self.parent_id,
            "operator": self.operator,
            "base_position": self.base_position.to_dict() if self.base_position else None,
            "base_sequence": self.base_position.algorithm_text(slot_order) if self.base_position else None,
            "prompt_ids": list(self.prompt_ids),
            "failure_reason": self.failure_reason,
        }


@dataclass
class _Particle:
    position: ParticlePosition
    set_position: SetPosition
    velocity: dict[str, dict[str, float]]
    pbest_position: ParticlePosition | None = None
    pbest_set_position: SetPosition | None = None
    pbest_objective: float | None = None
    current_objective: float | None = None
    current_details: dict[str, Any] | None = None
    particle_id: str = ""
    parent_id: str | None = None


def _tupleize(value):
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_tupleize(item) for item in value)
    return value


class SPSOEngine:
    """Typed semantic PSO with pluggable compilation and evaluation.

    ``evaluate_code`` returns a scalar, None, or {objective, details}. The
    compiler is responsible for producing a complete function that satisfies
    the task adapter's public input/output contract.
    """

    def __init__(
        self,
        registry: ModuleRegistry,
        compiler: HeuristicCompiler,
        evaluate_code: Callable[[str], float | dict[str, Any] | None],
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
        self.cut_log_dir = self.output_dir / "cut_log"
        self.run_log_path = self.output_dir / "run_log.txt"
        self.generation_metrics_path = self.output_dir / "generation_metrics.jsonl"
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cut_log_dir.mkdir(parents=True, exist_ok=True)
        self.sample_count = 0
        self.evaluation_count = 0
        self.cache: dict[str, dict[str, Any]] = {}
        self.cache_hits = 0
        self.records: list[dict[str, Any]] = []
        self.gbest_position: ParticlePosition | None = None
        self.gbest_set_position: SetPosition | None = None
        self.gbest_objective: float | None = None
        self._state_lock = threading.RLock()
        self._inflight: dict[str, threading.Event] = {}
        self._sampler_executor: ThreadPoolExecutor | None = None
        self._eval_executor: ThreadPoolExecutor | None = None
        self.next_particle_id = 1
        self.operator_counts: dict[str, dict[str, int]] = {}
        self.metadata = {
            "seed": self.config.seed,
            "operator_list": list(self.config.heuristic_operators),
            "use_module_descriptions": self.config.use_module_descriptions,
            "llm_model": self.config.llm_model,
            "modules_per_slot": self.config.modules_per_slot,
            "evaluation_settings": {},
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _log(self, message: str):
        """Write one timestamped line to both the terminal and run_log.txt."""
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
        with self._state_lock:
            print(line, flush=True)
            with self.run_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    @staticmethod
    def _format_objective(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.8g}"

    @staticmethod
    def _gap_percent(value: float | None, best: float | None) -> float | None:
        """Relative gap for a minimisation objective.

        A small denominator is common when a task reaches numerical zero. In
        that case equal values are reported as 0%, while a non-equal value is
        left as unavailable instead of displaying an unstable percentage.
        """
        if value is None or best is None:
            return None
        if abs(best) < 1e-12:
            return 0.0 if abs(value - best) < 1e-12 else None
        return max(0.0, 100.0 * (value - best) / abs(best))

    def _log_generation(self, generation: int, particles: list[_Particle]):
        """Log every particle's fitness and gap to the current global best."""
        best = self.gbest_objective
        self._log(
            f"--- Generation {generation}/{self.config.generations}  "
            f"pop={len(particles)}  best={self._format_objective(best)}"
        )
        rows = []
        for index, particle in enumerate(particles, start=1):
            fitness = particle.current_objective
            gap = self._gap_percent(fitness, best)
            absolute_gap = None if fitness is None or best is None else fitness - best
            gap_text = "N/A(best≈0)" if gap is None else f"{gap:.6f}%"
            self._log(
                f"  particle={index:02d}  fitness={self._format_objective(fitness):<14}  "
                f"gap_to_best={gap_text:<14}  "
                f"delta={self._format_objective(absolute_gap):<12}  "
                f"pbest={self._format_objective(particle.pbest_objective)}"
            )
            rows.append({
                "particle": index,
                "fitness": fitness,
                "pbest": particle.pbest_objective,
                "absolute_gap_to_best": absolute_gap,
                "gap_to_best_percent": gap,
                "position": particle.position.to_dict(),
                "position_text": particle.position.algorithm_text(self.registry.slot_ids),
                "set_position": particle.set_position.to_dict(),
                "pbest_set_position": (
                    particle.pbest_set_position.to_dict()
                    if particle.pbest_set_position else None
                ),
                "details": particle.current_details,
            })
            for item in (particle.current_details or {}).get("instances", []):
                self._log(
                    f"    instance={item['instance']} length={item['tour_length']:.12g} "
                    f"reference={item['reference_length']:.12g} "
                    f"gap={item['gap_percent']:.8g}% valid={item['valid_tour']}"
                )
        with self.generation_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "generation": generation,
                "best_objective": best,
                "particles": rows,
            }, ensure_ascii=False) + "\n")

    def _module_code_for_log(self, slot_id: str, module_id: str, params: dict[str, Any]) -> str | None:
        provider = getattr(self.selector, "module_code_provider", None)
        if provider is None:
            return None
        try:
            return provider(slot_id, module_id, params)
        except Exception as exc:
            logger.debug("could not render module code for cut log %s/%s: %s", slot_id, module_id, exc)
            return None

    def _cut_log_record(
        self,
        generation: int,
        particle_index: int,
        current: ParticlePosition,
        pbest: ParticlePosition,
        gbest: ParticlePosition,
        current_set: SetPosition,
        pbest_set: SetPosition,
        gbest_set: SetPosition,
        raw_cut: dict[str, list[str]],
        candidate_sets: dict[str, list[str]],
        objectives: dict[str, float | None],
        velocity: dict[str, dict[str, float]],
        alpha: float,
    ) -> dict[str, Any]:
        current_map = current_set.as_mapping()
        pbest_map = pbest_set.as_mapping()
        gbest_map = gbest_set.as_mapping()
        slots = []
        for slot in self.registry.slots:
            slot_velocity = {
                module.module_id: float(velocity.get(slot.slot_id, {}).get(module.module_id, 0.0))
                for module in slot.modules
            }
            anchors = {
                "current": list(current_map[slot.slot_id]),
                "pbest": list(pbest_map[slot.slot_id]),
                "gbest": list(gbest_map[slot.slot_id]),
            }
            candidates = []
            for module_id in candidate_sets[slot.slot_id]:
                spec = self.registry.module(slot.slot_id, module_id)
                params = spec.default_params()
                candidate = {
                    "module": module_id,
                    "description": spec.description,
                    "default_params": params,
                    "velocity": slot_velocity.get(module_id, 0.0),
                    "in_current": module_id in current_map[slot.slot_id],
                    "in_pbest": module_id in pbest_map[slot.slot_id],
                    "in_gbest": module_id in gbest_map[slot.slot_id],
                }
                code = self._module_code_for_log(slot.slot_id, module_id, params)
                if code is not None:
                    candidate["code"] = code
                candidates.append(candidate)
            slots.append({
                "slot": slot.slot_id,
                "description": slot.description,
                "min_cut": slot.min_cut,
                "max_cut": slot.max_cut,
                "set_capacity": slot.capacity,
                "alpha_threshold": alpha,
                "raw_alpha_cut": raw_cut[slot.slot_id],
                "constructed_set": candidate_sets[slot.slot_id],
                "anchors": anchors,
                "velocity_scores": dict(sorted(
                    slot_velocity.items(),
                    key=lambda item: (-item[1], item[0]),
                )),
                "cut_modules": candidates,
            })
        return {
            "generation": generation,
            "particle": particle_index,
            "alpha": alpha,
            "objectives": objectives,
            "current_position": current.to_dict(),
            "current_position_text": current.algorithm_text(self.registry.slot_ids),
            "pbest_position": pbest.to_dict(),
            "pbest_position_text": pbest.algorithm_text(self.registry.slot_ids),
            "gbest_position": gbest.to_dict(),
            "gbest_position_text": gbest.algorithm_text(self.registry.slot_ids),
            "current_set_position": current_set.to_dict(),
            "pbest_set_position": pbest_set.to_dict(),
            "gbest_set_position": gbest_set.to_dict(),
            "slots": slots,
        }

    def _write_cut_log(self, generation: int, rows: list[dict[str, Any]]):
        path = self.cut_log_dir / f"generation_{generation:04d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_best_heuristic(self, candidate: EvaluatedCandidate):
        if candidate.code is None:
            return
        path = self.sample_dir / "best_heuristic.py"
        content = candidate.code.rstrip() + "\n"
        path.write_text(content, encoding="utf-8")

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

    def _run_evaluation(self, code: str) -> float | dict[str, Any] | None:
        """Run one independent evaluation in the shared evaluation pool."""
        if self._eval_executor is None:
            return self.evaluate_code(code)
        return self._eval_executor.submit(self.evaluate_code, code).result()

    def _evaluate(
        self,
        position: ParticlePosition,
        source: str,
        strategy: str,
        set_position: SetPosition | None = None,
        *,
        generation: int = 0,
        particle_id: str | None = None,
        parent_id: str | None = None,
        operator: str = "p0",
        base_position: ParticlePosition | None = None,
        prompt_ids: tuple[str, ...] = (),
        failure_reason: str | None = None,
    ) -> EvaluatedCandidate:
        """Evaluate one candidate, safely callable from many sampler threads.

        Duplicate positions share one in-flight evaluation. Every request is
        still recorded as a sample, while only the first request consumes an
        independent evaluation slot.
        """
        position = self.registry.validate_position(position)
        set_position = self.registry.validate_set_position(
            set_position or self.registry.singleton_set_position(position)
        )
        key = self._key(position)
        code, algorithm = self.compiler.compile(position)

        use_cache = self.config.cache_evaluations
        with self._state_lock:
            self.sample_count += 1
            sample_id = self.sample_count
            cached = self.cache.get(key) if use_cache else None
            budget_exhausted = False
            if cached is not None:
                owner = False
                wait_event = None
            else:
                wait_event = self._inflight.get(key) if use_cache else None
                owner = wait_event is None
                if owner:
                    if (
                        self.config.max_evaluations is not None
                        and self.evaluation_count >= self.config.max_evaluations
                    ):
                        budget_exhausted = True
                    else:
                        wait_event = threading.Event() if use_cache else None
                        if use_cache:
                            self._inflight[key] = wait_event
                        self.evaluation_count += 1

        if cached is not None:
            candidate = EvaluatedCandidate(
                position=position,
                set_position=set_position,
                objective=cached["objective"],
                code=cached["code"],
                algorithm=cached["algorithm"],
                source=source,
                strategy=strategy,
                sample_id=sample_id,
                cached=True,
                details=cached.get("details"),
                generation=generation,
                particle_id=particle_id,
                parent_id=parent_id,
                operator=operator,
                base_position=base_position,
                prompt_ids=prompt_ids,
                failure_reason=failure_reason,
            )
            self.cache_hits += 1
            self._record(candidate)
            return candidate

        if budget_exhausted:
            candidate = EvaluatedCandidate(
                position=position,
                set_position=set_position,
                objective=None,
                code=code,
                algorithm=algorithm,
                source=source,
                strategy=strategy,
                sample_id=sample_id,
                cached=False,
                generation=generation,
                particle_id=particle_id,
                parent_id=parent_id,
                operator=operator,
                base_position=base_position,
                prompt_ids=prompt_ids,
                failure_reason=failure_reason or "evaluation_budget_exhausted",
            )
            self._record(candidate)
            return candidate

        if not owner:
            # Another sampler is already evaluating this exact position.
            wait_event.wait()
            with self._state_lock:
                cached = self.cache[key]
            candidate = EvaluatedCandidate(
                position=position,
                set_position=set_position,
                objective=cached["objective"],
                code=cached["code"],
                algorithm=cached["algorithm"],
                source=source,
                strategy=strategy,
                sample_id=sample_id,
                cached=True,
                details=cached.get("details"),
                generation=generation,
                particle_id=particle_id,
                parent_id=parent_id,
                operator=operator,
                base_position=base_position,
                prompt_ids=prompt_ids,
                failure_reason=failure_reason,
            )
            self.cache_hits += 1
            self._record(candidate)
            return candidate

        details = None
        evaluation_failure_reason = None
        try:
            result = self._run_evaluation(code)
            if isinstance(result, dict):
                objective, details = result.get("objective"), result.get("details")
            else:
                objective = result
            objective = float(objective) if objective is not None else None
        except Exception as exc:
            logger.warning("candidate evaluation failed: %s", exc)
            objective = None
            evaluation_failure_reason = (
                f"evaluation exception: {type(exc).__name__}: {exc}"
            )
        if objective is None and evaluation_failure_reason is None:
            evaluation_failure_reason = (
                "evaluation returned None (most likely outer timeout or "
                "problem.evaluate swallowed an exception)"
            )
        if objective is not None and (objective != objective or objective in (float("inf"), float("-inf"))):
            objective = None

        with self._state_lock:
            if use_cache:
                self.cache[key] = {"objective": objective, "code": code, "algorithm": algorithm, "details": details}
                wait_event.set()
                self._inflight.pop(key, None)
        candidate = EvaluatedCandidate(
            position=position,
            set_position=set_position,
            objective=objective,
            code=code,
            algorithm=algorithm,
            source=source,
            strategy=strategy,
            sample_id=sample_id,
            cached=False,
            details=details,
            generation=generation,
            particle_id=particle_id,
            parent_id=parent_id,
            operator=operator,
            base_position=base_position,
            prompt_ids=prompt_ids,
            failure_reason=(
                (failure_reason or evaluation_failure_reason)
                if objective is None else None
            ),
        )
        self._record(candidate)
        return candidate

    def _record(self, candidate: EvaluatedCandidate):
        record = candidate.to_dict(self.registry.slot_ids)
        with self._state_lock:
            self.records.append(record)
            stats = self.operator_counts.setdefault(candidate.operator, {
                "attempted": 0, "valid": 0, "failed": 0, "cached": 0,
            })
            stats["attempted"] += 1
            stats["valid" if candidate.objective is not None else "failed"] += 1
            if candidate.cached:
                stats["cached"] += 1
            with (self.sample_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if candidate.objective is not None and (
                self.gbest_objective is None or candidate.objective < self.gbest_objective
            ):
                self.gbest_objective = candidate.objective
                self.gbest_position = candidate.position
                self.gbest_set_position = candidate.set_position
                with (self.sample_dir / "best.json").open("w", encoding="utf-8") as handle:
                    json.dump(record, handle, ensure_ascii=False, indent=2)
                self._write_best_heuristic(candidate)

    def _initialise(self) -> list[_Particle]:
        count = self.config.initial_samples or 2 * self.config.population_size
        evolution = getattr(self.selector, "evolution", None)
        if evolution is not None:
            # p0 is an LLM request over the frozen ten-per-slot library.  A
            # malformed/duplicate batch raises explicitly; it never silently
            # falls back to random legacy modules.
            positions = evolution.initial_sequences(
                count,
                seed=self.config.seed,
                require_tunable=("p4" in self.config.heuristic_operators),
            )
            set_positions = [
                construct_set_position(
                    {}, self.registry.singleton_set_position(position), self.registry, self.rng
                )
                for position in positions
            ]
        else:
            # Compatibility mode for the old unit-level API.  Online runner
            # construction attaches TSPModuleEvolution before calling run().
            positions = [
                self.registry.random_position(self.rng)
                for _ in range(count)
            ]
            set_positions = [
                construct_set_position(
                    {}, self.registry.singleton_set_position(position), self.registry, self.rng
                )
                for position in positions
            ]
        if self._sampler_executor is None:
            candidates = [
                self._evaluate(position, "initial", "p0", set_position,
                               generation=0, particle_id=f"p{index:05d}",
                               operator="p0")
                for index, (position, set_position) in enumerate(zip(positions, set_positions), start=1)
            ]
        else:
            futures = [
                self._sampler_executor.submit(
                    self._evaluate, position, "initial", "p0", set_position,
                    generation=0, particle_id=f"p{index:05d}", operator="p0"
                )
                for index, (position, set_position) in enumerate(zip(positions, set_positions), start=1)
            ]
            candidates = [future.result() for future in futures]
        valid = [candidate for candidate in candidates if candidate.objective is not None]
        if len(valid) < self.config.population_size:
            failures = Counter(
                candidate.failure_reason or "evaluation returned None (timeout or problem.evaluate failure)"
                for candidate in candidates
                if candidate.objective is None
            )
            detail = "; ".join(f"{reason}: {count}" for reason, count in failures.most_common())
            self._log(
                f"  initialisation failed: valid={len(valid)}/{len(candidates)}; "
                f"failure_reasons={detail}"
            )
            raise RuntimeError(
                "not enough valid initial candidates to form a population; "
                f"valid={len(valid)}/{len(candidates)}; {detail}"
            )
        valid.sort(key=lambda candidate: candidate.objective)
        particles = []
        seen = set()
        for candidate in valid:
            key = self._key(candidate.position)
            if key in seen:
                continue
            seen.add(key)
            particle_id = f"p{self.next_particle_id:05d}"
            self.next_particle_id += 1
            particles.append(
                _Particle(
                    position=candidate.position,
                    set_position=candidate.set_position,
                    velocity=initial_velocity(self.registry),
                    pbest_position=candidate.position,
                    pbest_set_position=candidate.set_position,
                    pbest_objective=candidate.objective,
                    current_objective=candidate.objective,
                    current_details=candidate.details,
                    particle_id=particle_id,
                )
            )
            if len(particles) >= self.config.population_size:
                break
        if len(particles) < self.config.population_size:
            raise RuntimeError("not enough distinct valid initial candidates to form a population")
        return particles

    def _produce_evolution_candidate(self, plan) -> EvaluatedCandidate:
        """Compile/evaluate one already validated evolution proposal."""
        candidate_spec = plan
        return self._evaluate(
            candidate_spec["position"],
            "evolution",
            candidate_spec["operator"],
            candidate_spec["set_position"],
            generation=candidate_spec["generation"],
            particle_id=candidate_spec["particle_id"],
            parent_id=candidate_spec["parent_id"],
            operator=candidate_spec["operator"],
            base_position=candidate_spec.get("base_position"),
            prompt_ids=tuple(candidate_spec.get("prompt_ids", ())),
        )

    def _register_child_modules(self, proposals, generation: int) -> None:
        """Register new p1/p2/p3 recipes in deterministic plan order."""
        from dataclasses import replace
        from .registry import ModuleSpec
        for proposal in proposals:
            for item in getattr(proposal, "new_modules", ()):
                recipe = dict(item.recipe)
                values = dict(recipe.pop("parameters", {}))
                if not item.module_id:
                    raise ValueError("new module proposal has no stable module ID")
                parameter_specs = tuple(
                    replace(parameter, default=values[parameter.name])
                    if parameter.name in values else parameter
                    for parameter in item.parameters
                )
                module = ModuleSpec(
                    module_id=item.module_id,
                    description=item.description,
                    parameters=parameter_specs,
                    tags=("llm_generated", item.source_operator),
                    recipe=recipe,
                    generation=int(generation),
                    source_operator=item.source_operator,
                    semantic_family=str(recipe.get("family", "")),
                    polarity=str(recipe.get("polarity", "neutral")),
                )
                self.registry.register_module(item.slot, module)
                # Preserve the validated values for the child sequence. The
                # proposal already carries them in its ModuleChoice.
                del values

    def _child_set_with_new_modules(self, base_set: SetPosition, proposal) -> SetPosition:
        if not getattr(proposal, "new_modules", None):
            return self.registry.validate_set_position(base_set)
        values = {slot: list(ids) for slot, ids in base_set.as_mapping().items()}
        for item in proposal.new_modules:
            members = values[item.slot]
            if item.module_id not in members:
                if len(members) >= self.registry.slot(item.slot).capacity:
                    members.pop()
                members.append(item.module_id)
        return self.registry.validate_set_position(SetPosition.create(values))

    @staticmethod
    def _failed_candidate(position, set_position, generation, particle_id, parent_id, operator, reason, base_position=None, prompt_ids=()):
        return EvaluatedCandidate(
            position=position,
            set_position=set_position,
            objective=None,
            code=None,
            algorithm=None,
            source="failed_generation",
            strategy=operator,
            sample_id=-1,
            generation=generation,
            particle_id=particle_id,
            parent_id=parent_id,
            operator=operator,
            base_position=base_position,
            prompt_ids=tuple(prompt_ids),
            failure_reason=reason,
        )

    def _generate_parent_proposals(self, task):
        """Sampler worker: generate one base and all operators for one parent.

        The task contains frozen positions/cuts only. This worker never calls
        the engine RNG and never registers a module; registration is performed
        by the coordinator after all futures return.
        """
        evolution = getattr(self.selector, "evolution", None)
        particle, particle_index, next_set, cuts, generation = task
        base = particle.position
        base_calls = []
        base_error = None
        try:
            if evolution is not None:
                base, base_calls = evolution.base_sequence(
                    cuts, generation=generation, parent_id=particle.particle_id
                )
        except Exception as exc:
            base_error = f"base p0 generation failed: {type(exc).__name__}: {exc}"
        output = []
        for operator in self.config.heuristic_operators:
            proposal = None
            reason = base_error
            try:
                if reason is None and operator == "p0":
                    from types import SimpleNamespace
                    proposal = SimpleNamespace(
                        operator="p0", position=base, set_position=None,
                        new_modules=[], base_position=base, prompt_ids=base_calls,
                    )
                elif reason is None:
                    proposal = evolution.propose(
                        operator, particle.position, base, cuts,
                        generation=generation, parent_id=particle.particle_id,
                    )
                if reason is None and (proposal is None or proposal.position is None):
                    raise RuntimeError(f"{operator} returned no position")
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
            output.append((particle, particle_index, particle, next_set, base, proposal, operator, reason))
        return output

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
            self.gbest_set_position = best.pbest_set_position

    def _checkpoint(self, particles: list[_Particle], generation: int):
        payload = {
            "generation": generation,
            "sample_count": self.sample_count,
            "evaluation_count": self.evaluation_count,
            "gbest_objective": self.gbest_objective,
            "gbest_position": self.gbest_position.to_dict() if self.gbest_position else None,
            "gbest_set_position": self.gbest_set_position.to_dict() if self.gbest_set_position else None,
            "module_library": self.registry.library_payload(),
            "module_library_hash": self.registry.library_hash(),
            "metadata": self.metadata,
            "operator_counts": self.operator_counts,
            "cache_hits": self.cache_hits,
            "next_particle_id": self.next_particle_id,
            "operator_list": list(self.config.heuristic_operators),
            "particles": [
                {
                    "position": particle.position.to_dict(),
                    "set_position": particle.set_position.to_dict(),
                    "velocity": particle.velocity,
                    "pbest_position": particle.pbest_position.to_dict() if particle.pbest_position else None,
                    "pbest_set_position": (
                        particle.pbest_set_position.to_dict()
                        if particle.pbest_set_position else None
                    ),
                    "pbest_objective": particle.pbest_objective,
                    "current_objective": particle.current_objective,
                    "current_details": particle.current_details,
                    "particle_id": particle.particle_id,
                    "parent_id": particle.parent_id,
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
        # The library is restored before any position/velocity validation.
        if payload.get("module_library"):
            self.registry.restore_library(payload["module_library"])
        particles = []
        for raw in payload["particles"]:
            pbest = raw.get("pbest_position")
            position = self.registry.validate_position(ParticlePosition.from_dict(raw["position"]))
            raw_set = raw.get("set_position")
            set_position = (
                self.registry.validate_set_position(SetPosition.from_dict(raw_set))
                if raw_set else self.registry.singleton_set_position(position)
            )
            pbest_position = (
                self.registry.validate_position(ParticlePosition.from_dict(pbest))
                if pbest else None
            )
            raw_pbest_set = raw.get("pbest_set_position")
            pbest_set_position = (
                self.registry.validate_set_position(SetPosition.from_dict(raw_pbest_set))
                if raw_pbest_set else (
                    self.registry.singleton_set_position(pbest_position)
                    if pbest_position else None
                )
            )
            particles.append(
                _Particle(
                    position=position,
                    set_position=set_position,
                    velocity=raw["velocity"],
                    pbest_position=pbest_position,
                    pbest_set_position=pbest_set_position,
                    pbest_objective=raw.get("pbest_objective"),
                    current_objective=raw.get("current_objective"),
                    current_details=raw.get("current_details"),
                    particle_id=str(raw.get("particle_id", f"p{len(particles)+1:05d}")),
                    parent_id=raw.get("parent_id"),
                )
            )
        self.sample_count = int(payload.get("sample_count", 0))
        self.evaluation_count = int(payload.get("evaluation_count", 0))
        self.cache_hits = int(payload.get("cache_hits", 0))
        self.operator_counts = payload.get("operator_counts", {})
        self.next_particle_id = int(payload.get("next_particle_id", len(particles) + 1))
        self.metadata = payload.get("metadata", self.metadata)
        self.cache = payload.get("cache", {})
        raw_gbest = payload.get("gbest_position")
        self.gbest_position = (
            self.registry.validate_position(ParticlePosition.from_dict(raw_gbest))
            if raw_gbest else None
        )
        raw_gbest_set = payload.get("gbest_set_position")
        self.gbest_set_position = (
            self.registry.validate_set_position(SetPosition.from_dict(raw_gbest_set))
            if raw_gbest_set else (
                self.registry.singleton_set_position(self.gbest_position)
                if self.gbest_position else None
            )
        )
        self.gbest_objective = payload.get("gbest_objective")
        if "rng_state" in payload:
            self.rng.setstate(_tupleize(payload["rng_state"]))
        return particles, int(payload.get("generation", 0))

    def run(self) -> dict[str, Any]:
        started = time.time()
        initial_count = self.config.initial_samples or 2 * self.config.population_size
        planned = initial_count + self.config.generations * self.config.population_size * len(self.config.heuristic_operators)
        if self.config.max_evaluations is not None and self.config.max_evaluations < planned:
            raise ValueError(
                f"max_evaluations={self.config.max_evaluations} would truncate the planned "
                f"{planned} samples; increase it or set it to None"
            )
        if not self.config.resume_from:
            self.run_log_path.write_text("", encoding="utf-8")
            self.generation_metrics_path.write_text("", encoding="utf-8")
        self.metadata["planned_sample_evaluations"] = planned
        self.metadata["library_hash"] = self.registry.library_hash()
        (self.output_dir / "metadata.json").write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log("=" * 64)
        self._log("  S-PSO EoH-style typed module evolution")
        self._log(f"  generations={self.config.generations} pop={self.config.population_size} initial={initial_count} operators={list(self.config.heuristic_operators)}")
        self._log(f"  planned candidate records={planned}; descriptions={'on' if self.config.use_module_descriptions else 'off'}")
        self._log("=" * 64)
        self._sampler_executor = ThreadPoolExecutor(max_workers=self.config.num_samplers, thread_name_prefix="spso-sampler")
        self._eval_executor = ThreadPoolExecutor(max_workers=self.config.num_evaluators, thread_name_prefix="spso-eval")
        try:
            if self.config.resume_from:
                particles, start_generation = self._restore(self.config.resume_from)
            else:
                particles = self._initialise()
                self._update_global(particles)
                start_generation = 0
                self._checkpoint(particles, 0)
                self._log_generation(0, particles)
            if self.config.resume_from:
                self._log_generation(start_generation, particles)
            evolution = getattr(self.selector, "evolution", None)
            if evolution is None and self.config.heuristic_operators != ("p0",):
                raise RuntimeError("EoH-style p1..p4 evolution requires selector.evolution")
            for generation in range(start_generation + 1, self.config.generations + 1):
                snapshot = self.gbest_position
                snapshot_set = self.gbest_set_position
                if snapshot is None or snapshot_set is None:
                    raise RuntimeError("gbest is undefined after initialisation")
                plans = []
                cut_rows = []
                parent_tasks = []
                # Freeze all PSO state, velocity and cuts before sampler work.
                for particle_index, particle in enumerate(particles, start=1):
                    if particle.pbest_position is None or particle.pbest_set_position is None:
                        raise RuntimeError("particle has no pbest state")
                    particle.velocity = update_velocity(
                        particle.velocity, particle.set_position, particle.pbest_set_position,
                        snapshot_set, self.registry, self.rng, self.config.omega,
                        self.config.c_personal, self.config.c_global, self.config.c_explore,
                    )
                    alpha = self.config.alpha if self.config.alpha is not None else self.rng.random()
                    raw_cut = alpha_cut_sets(particle.velocity, self.registry, alpha)
                    next_set = construct_set_position(raw_cut, particle.set_position, self.registry, self.rng)
                    cuts = {slot: list(ids) for slot, ids in next_set.as_mapping().items()}
                    objectives = {"current": particle.current_objective, "pbest": particle.pbest_objective, "gbest": self.gbest_objective}
                    cut_rows.append(self._cut_log_record(
                        generation, particle_index, particle.position, particle.pbest_position, snapshot,
                        particle.set_position, particle.pbest_set_position, snapshot_set, raw_cut, cuts,
                        objectives, particle.velocity, alpha,
                    ))
                    parent_tasks.append((particle, particle_index, next_set, cuts, generation))
                self._write_cut_log(generation, cut_rows)
                proposal_futures = [
                    self._sampler_executor.submit(self._generate_parent_proposals, task)
                    for task in parent_tasks
                ]
                # Futures are consumed in parent order, preserving the
                # deterministic parent/operator registration order.
                child_proposals = [
                    item
                    for future in proposal_futures
                    for item in future.result()
                ]
                # Register all dynamic modules in stable parent/operator order.
                registered = []
                for item in child_proposals:
                    _parent, _index, _same, _base_set, _base, proposal, _operator, reason = item
                    if reason is not None or proposal is None:
                        registered.append(reason)
                        continue
                    try:
                        if getattr(proposal, "new_modules", None):
                            self._register_child_modules([proposal], generation)
                        registered.append(None)
                    except Exception as exc:
                        registered.append(f"module registration failed: {type(exc).__name__}: {exc}")
                for item, registration_reason in zip(child_proposals, registered):
                    parent, parent_index, _same, base_set, base, proposal, operator, reason = item
                    reason = reason or registration_reason
                    if reason is None:
                        try:
                            child_set = self._child_set_with_new_modules(base_set, proposal)
                            proposal.position = self.registry.validate_position(proposal.position)
                        except Exception as exc:
                            reason = f"child validation failed: {type(exc).__name__}: {exc}"
                    if reason is not None:
                        position = base
                        child_set = base_set
                        candidate = self._failed_candidate(
                            position, child_set, generation,
                            f"p{self.next_particle_id:05d}", parent.particle_id,
                            operator,
                            reason, base, getattr(proposal, "prompt_ids", ()),
                        )
                        self.next_particle_id += 1
                        with self._state_lock:
                            self.sample_count += 1
                            candidate.sample_id = self.sample_count
                        self._record(candidate)
                        plans.append({"failed_candidate": candidate, "parent": parent})
                        continue
                    plans.append({
                        "position": proposal.position,
                        "set_position": child_set,
                        "operator": operator,
                        "generation": generation,
                        "particle_id": f"p{self.next_particle_id:05d}",
                        "parent_id": parent.particle_id,
                        "base_position": base,
                        "prompt_ids": proposal.prompt_ids,
                        "parent_index": parent_index,
                        "parent": parent,
                    })
                    self.next_particle_id += 1
                valid_plans = [plan for plan in plans if "failed_candidate" not in plan]
                futures = [self._sampler_executor.submit(self._produce_evolution_candidate, plan) for plan in valid_plans]
                valid_children = [future.result() for future in futures]
                valid_iter = iter(valid_children)
                children = [
                    plan["failed_candidate"] if "failed_candidate" in plan else next(valid_iter)
                    for plan in plans
                ]
                candidates_by_plan = list(zip(plans, children))
                # EoH-style elitism over old parents plus every valid child.
                ranked = [(particle.current_objective, order, "parent", particle, None) for order, particle in enumerate(particles)]
                for order, (plan, candidate) in enumerate(candidates_by_plan, start=len(ranked)):
                    if candidate.objective is not None:
                        ranked.append((candidate.objective, order, "child", plan["parent"], candidate))
                ranked.sort(key=lambda item: (item[0], item[1]))
                survivors = []
                seen_positions = set()
                survival_by_operator = {operator: 0 for operator in self.config.heuristic_operators}
                for _objective, _order, kind, parent, candidate in ranked:
                    position = parent.position if kind == "parent" else candidate.position
                    key = self._key(position)
                    if key in seen_positions:
                        continue
                    seen_positions.add(key)
                    if kind == "parent":
                        survivors.append(parent)
                    else:
                        child_pbest = parent.pbest_position
                        child_pbest_set = parent.pbest_set_position
                        child_pbest_objective = parent.pbest_objective
                        if child_pbest_objective is None or candidate.objective < child_pbest_objective:
                            child_pbest, child_pbest_set, child_pbest_objective = candidate.position, candidate.set_position, candidate.objective
                        survivors.append(_Particle(
                            position=candidate.position, set_position=candidate.set_position,
                            velocity={slot: dict(values) for slot, values in parent.velocity.items()},
                            pbest_position=child_pbest, pbest_set_position=child_pbest_set,
                            pbest_objective=child_pbest_objective, current_objective=candidate.objective,
                            current_details=candidate.details, particle_id=candidate.particle_id or "",
                            parent_id=candidate.parent_id,
                        ))
                        survival_by_operator[candidate.operator] = survival_by_operator.get(candidate.operator, 0) + 1
                    if len(survivors) >= self.config.population_size:
                        break
                if len(survivors) != self.config.population_size:
                    raise RuntimeError("elitism could not form a full population from valid distinct positions")
                particles = survivors
                self._update_global(particles)
                self._log_generation(generation, particles)
                self._log(f"  generated_children={len(children)} valid_children={sum(c.objective is not None for c in children)} survival_by_operator={survival_by_operator}")
                if generation % max(1, self.config.checkpoint_every) == 0:
                    self._checkpoint(particles, generation)
        finally:
            if self._sampler_executor is not None:
                self._sampler_executor.shutdown(wait=True)
                self._sampler_executor = None
            if self._eval_executor is not None:
                self._eval_executor.shutdown(wait=True)
                self._eval_executor = None
            evolution = getattr(self.selector, "evolution", None)
            if evolution is not None and hasattr(evolution, "llm"):
                evolution.llm.flush()
        elapsed = time.time() - started
        summary = {
            "best_objective": self.gbest_objective,
            "best_position": self.gbest_position.to_dict() if self.gbest_position else None,
            "best_position_text": self.gbest_position.algorithm_text(self.registry.slot_ids) if self.gbest_position else None,
            "samples": self.sample_count,
            "independent_evaluations": self.evaluation_count,
            "cached_samples": self.cache_hits,
            "operator_counts": self.operator_counts,
            "cache_entries": len(self.cache),
            "elapsed_seconds": elapsed,
            "run_log": str(self.run_log_path),
            "generation_metrics": str(self.generation_metrics_path),
            "cut_log": str(self.cut_log_dir),
            "module_library_hash": self.registry.library_hash(),
        }
        with (self.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        self._log(f"Evolution finished. best={self._format_objective(self.gbest_objective)} samples={self.sample_count} independent_evaluations={self.evaluation_count} time={elapsed / 60:.1f}m")
        return summary
