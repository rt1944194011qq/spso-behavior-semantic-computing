from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..eoh.evolution import _eval_with_timeout
from .config import MAEConfig
from .generator import ProgramDraft, ProgramGenerator
from .models import AgentState, CutItem, ProgramIndividual
from .reporting import ExperimentRecorder
from .set_algebra import update_velocity
from .topology import ring_neighbors
from .validation import ProgramValidator, parameter_only_mutation, structural_fingerprint


@dataclass
class _CandidateSpec:
    agent_id: int
    generation: int
    draft: ProgramDraft
    source_operator: str
    guidance: str | None = None
    variation: str | None = None
    parent_ids: list[str] | None = None
    source_cut_ids: list[str] | None = None


class MAEEngine:
    """Synchronous multi-agent evolution with parallel LLM and evaluation stages."""

    def __init__(
        self,
        config: MAEConfig,
        problem,
        generator: ProgramGenerator,
        validator: ProgramValidator | None = None,
    ) -> None:
        self.config = config
        self.problem = problem
        self.generator = generator
        self.validator = validator or ProgramValidator(problem.template_program)

        self.archive: dict[str, ProgramIndividual] = {}
        self.agents: list[AgentState] = []
        self.gbest_id: str | None = None
        self._individual_counter = 0
        self._evaluation_count = 0
        self._cache_hits = 0
        self._evaluation_cache: dict[str, dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self._failure_counts: dict[int, Counter] = {}
        self._path_failure_counts: dict[int, Counter] = {}
        self._llm_pool: ThreadPoolExecutor | None = None
        self._eval_pool: ThreadPoolExecutor | None = None

        self.output_dir = config.output_path
        self._prepare_output()
        self.logger = self._setup_logger()
        self.recorder = ExperimentRecorder(
            self.output_dir, detailed=self.config.detailed_logging
        )
        self.recorder.record_environment(self.config.to_dict(), self.problem)

    # ------------------------------------------------------------------ setup

    def _prepare_output(self) -> None:
        config_path = self.output_dir / "config.json"
        if config_path.exists() and not self.config.resume_from:
            raise FileExistsError(
                f"Output directory already contains a MA-EoH run: {self.output_dir}. "
                "Choose a new run name, or set resume_from to a checkpoint."
            )
        for relative in (
            "archive", "agents", "velocities", "cuts", "prompts", "lineage",
            "failures", "metrics", "checkpoints", "best",
        ):
            (self.output_dir / relative).mkdir(parents=True, exist_ok=True)
        self._write_json(config_path, self.config.to_dict())

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"eoh.mae.{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()
        level = logging.DEBUG if self.config.debug else logging.INFO
        formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(
            self.output_dir / "run_log.txt", mode="a", encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    def _close_logger(self) -> None:
        """Release file handles so Windows can move or remove a finished run."""
        for handler in list(self.logger.handlers):
            try:
                handler.flush()
                handler.close()
            finally:
                self.logger.removeHandler(handler)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)

    @staticmethod
    def _append_jsonl(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    def _seed_for(self, *parts: object) -> int:
        joined = "|".join(str(item) for item in (self.config.seed, *parts))
        return int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16], 16)

    def _completed_futures(
        self,
        futures: dict[Future, Any],
        *,
        phase: str,
        generation: int | None,
    ):
        """Yield completed futures and emit heartbeats while all work is busy."""
        pending = set(futures)
        started = time.perf_counter()
        while pending:
            done, pending = wait(
                pending,
                timeout=self.config.heartbeat_interval_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                elapsed = time.perf_counter() - started
                self.recorder.emit(
                    "phase_heartbeat",
                    generation=generation,
                    topic="progress",
                    phase=phase,
                    pending=len(pending),
                    total=len(futures),
                    elapsed_seconds=elapsed,
                )
                self.logger.info(
                    "  heartbeat phase=%s pending=%d/%d elapsed=%.1fs",
                    phase,
                    len(pending),
                    len(futures),
                    elapsed,
                )
                continue
            for future in done:
                yield future

    @staticmethod
    def _code_hash(code: str) -> str:
        normalized = "\n".join(line.rstrip() for line in code.strip().splitlines())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _next_individual_id(self) -> str:
        self._individual_counter += 1
        return f"ind-{self._individual_counter:08d}"

    # ------------------------------------------------------------ evaluation

    def _evaluation_key(self, code_hash: str) -> str:
        return f"{code_hash}:{self.config.evaluation_fingerprint}"

    def _evaluate_code(self, code: str) -> tuple[float | None, dict[str, Any] | None]:
        if self.config.isolate_evaluation:
            result = _eval_with_timeout(self.problem, code, self.problem.timeout)
        else:
            result = self.problem.evaluate(code)
        if result is None:
            return None, None
        details = None
        if isinstance(result, dict):
            details = result.get("details")
            result = result.get("fitness")
        try:
            value = float(result)
        except (TypeError, ValueError):
            return None, details
        if not math.isfinite(value):
            return None, details
        value = 0.0 if abs(value) < 1e-12 else value
        return value, details

    def _record_failure(self, spec: _CandidateSpec, stage: str, reason: str) -> None:
        operator_path = (
            f"{spec.guidance}->{spec.variation}"
            if spec.guidance and spec.variation
            else spec.source_operator
        )
        record = {
            "generation": spec.generation,
            "agent_id": spec.agent_id,
            "source_operator": spec.source_operator,
            "guidance_operator": spec.guidance,
            "variation_operator": spec.variation,
            "operator_path": operator_path,
            "parent_ids": spec.parent_ids or [],
            "source_cut_ids": spec.source_cut_ids or [],
            "prompt_hash": spec.draft.prompt_hash,
            "failure_stage": stage,
            "failure_reason": reason,
        }
        self._append_jsonl(self.output_dir / "failures" / "failures.jsonl", record)
        self._failure_counts.setdefault(spec.generation, Counter())[stage] += 1
        self._path_failure_counts.setdefault(spec.generation, Counter())[operator_path] += 1
        self.recorder.emit(
            "candidate_rejected",
            generation=spec.generation,
            agent_id=spec.agent_id,
            topic="candidate",
            **{
                key: value
                for key, value in record.items()
                if key not in {"generation", "agent_id"}
            },
        )

    def _repair_h5(self, spec: _CandidateSpec, reason: str) -> bool:
        """Repair an H5 draft with a deterministic parameter-only mutation.

        The LLM remains the primary generator.  This fallback is used only when
        H5 violates its structural constraint, or when its returned code cannot
        be validated.  It prevents an otherwise usable H5 branch from producing
        zero candidates solely because the model changed the program structure.
        """
        if spec.variation != "H5" or not spec.parent_ids:
            return False
        parent = self.archive.get(spec.parent_ids[0])
        if parent is None:
            return False

        repaired = parameter_only_mutation(
            parent.code,
            seed=(
                f"g{spec.generation}-a{spec.agent_id}-"
                f"{spec.guidance}-{spec.variation}-{parent.individual_id}"
            ),
        )
        if not repaired:
            return False

        old_draft = spec.draft
        spec.draft = ProgramDraft(
            description=(
                (old_draft.description or parent.description).strip()
                + " [H5 automatic parameter-only repair]"
            ),
            code=repaired,
            prompt_hash=old_draft.prompt_hash,
            llm_seconds=old_draft.llm_seconds,
            llm_attempts=old_draft.llm_attempts,
            raw_response=old_draft.raw_response,
        )
        self.recorder.emit(
            "h5_repaired",
            generation=spec.generation,
            agent_id=spec.agent_id,
            topic="candidate",
            detailed_only=True,
            operator_path=f"{spec.guidance}->{spec.variation}",
            parent_id=parent.individual_id,
            reason=reason,
        )
        return True
    def _record_program(self, individual: ProgramIndividual) -> None:
        self.archive[individual.individual_id] = individual
        self._append_jsonl(
            self.output_dir / "archive" / "programs.jsonl", individual.to_dict()
        )
        self._append_jsonl(
            self.output_dir / "lineage" / "individuals.jsonl", individual.to_dict()
        )
        self.recorder.emit(
            "candidate_accepted",
            generation=individual.generation,
            agent_id=individual.agent_id,
            topic="candidate",
            individual=individual.to_dict(),
        )
        for parent_id in individual.parent_ids:
            self._append_jsonl(
                self.output_dir / "lineage" / "edges.jsonl",
                {
                    "parent_id": parent_id,
                    "child_id": individual.individual_id,
                    "generation": individual.generation,
                    "operator_path": individual.operator_path,
                },
            )

    def _evaluate_specs(
        self,
        specs: list[_CandidateSpec],
        seen_by_agent: dict[int, set[str]],
    ) -> list[ProgramIndividual]:
        assert self._eval_pool is not None
        accepted: list[tuple[_CandidateSpec, str]] = []
        for spec in specs:
            draft = spec.draft
            h5_repaired = False
            candidate_ref = draft.prompt_hash or (
                f"g{spec.generation}-a{spec.agent_id}-{spec.source_operator}"
            )
            self.recorder.emit(
                "candidate_received",
                generation=spec.generation,
                agent_id=spec.agent_id,
                topic="candidate",
                detailed_only=True,
                candidate_ref=candidate_ref,
                source_operator=spec.source_operator,
                guidance_operator=spec.guidance,
                variation_operator=spec.variation,
                parent_ids=spec.parent_ids or [],
                source_cut_ids=spec.source_cut_ids or [],
                llm_seconds=draft.llm_seconds,
                llm_attempts=draft.llm_attempts,
                description=draft.description,
                code_chars=len(draft.code) if draft.code else 0,
            )
            if not draft.code:
                if self._repair_h5(spec, "LLM returned no usable H5 code"):
                    draft = spec.draft
                    h5_repaired = True
                else:
                    self._record_failure(
                        spec,
                        draft.failure_stage or "llm_parse",
                        draft.failure_reason or "no code returned",
                    )
                    continue
            if self.config.require_description and not draft.description:
                if self._repair_h5(spec, "H5 description was missing"):
                    draft = spec.draft
                    h5_repaired = True
                else:
                    self._record_failure(spec, "missing_description", "description is required")
                    continue

            validation_started = time.perf_counter()
            validation = self.validator.validate(draft.code)
            validation_seconds = time.perf_counter() - validation_started
            if not validation.valid and self._repair_h5(
                spec, f"initial validation failed: {validation.reason}"
            ):
                draft = spec.draft
                h5_repaired = True
                validation_started = time.perf_counter()
                validation = self.validator.validate(draft.code)
                validation_seconds = time.perf_counter() - validation_started

            self.recorder.emit(
                "candidate_validated",
                generation=spec.generation,
                agent_id=spec.agent_id,
                topic="candidate",
                detailed_only=True,
                candidate_ref=candidate_ref,
                valid=validation.valid,
                validation_stage=validation.stage,
                validation_reason=validation.reason,
                validation_seconds=validation_seconds,
                h5_repaired=h5_repaired,
            )
            if not validation.valid:
                self._record_failure(
                    spec,
                    validation.stage or "validation",
                    validation.reason or "invalid program",
                )
                continue
            if spec.parent_ids and spec.variation in ("H4", "H5"):
                try:
                    child_structure = structural_fingerprint(draft.code)
                    parent_structures = [
                        structural_fingerprint(self.archive[parent_id].code)
                        for parent_id in spec.parent_ids
                    ]
                except (KeyError, SyntaxError) as exc:
                    self._record_failure(spec, "similarity_violation", str(exc))
                    continue
                if spec.variation == "H4" and child_structure in parent_structures:
                    self._record_failure(
                        spec,
                        "similarity_violation",
                        "H4 child has the same parameter-insensitive AST as a parent",
                    )
                    continue
                if (
                    spec.variation == "H5"
                    and child_structure != parent_structures[0]
                    and self._repair_h5(spec, "LLM changed H5 program structure")
                ):
                    draft = spec.draft
                    h5_repaired = True
                    child_structure = structural_fingerprint(draft.code)
                if spec.variation == "H5" and child_structure != parent_structures[0]:
                    self._record_failure(
                        spec,
                        "similarity_violation",
                        "H5 changed structure instead of parameters only",
                    )
                    continue
            code_hash = self._code_hash(draft.code)
            if code_hash in seen_by_agent.setdefault(spec.agent_id, set()):
                self._record_failure(spec, "duplicate_code", code_hash)
                continue
            seen_by_agent[spec.agent_id].add(code_hash)
            accepted.append((spec, code_hash))

        by_key: dict[str, list[tuple[_CandidateSpec, str]]] = {}
        for spec, code_hash in accepted:
            by_key.setdefault(self._evaluation_key(code_hash), []).append((spec, code_hash))

        fitness_by_key: dict[
            str, tuple[float | None, bool, dict[str, Any] | None, float | None]
        ] = {}
        futures: dict[Future, tuple[str, float]] = {}
        for key, grouped in by_key.items():
            with self._cache_lock:
                cached = self._evaluation_cache.get(key)
            if cached is not None:
                cached_fitness = float(cached["fitness"])
                cached_details = cached.get("details")
                fitness_by_key[key] = (cached_fitness, True, cached_details, 0.0)
                self._cache_hits += len(grouped)
                self.recorder.emit(
                    "evaluation_cache_hit",
                    generation=grouped[0][0].generation,
                    agent_id=grouped[0][0].agent_id,
                    topic="evaluation",
                    evaluation_key=key,
                    grouped_candidates=len(grouped),
                    fitness=cached_fitness,
                    evaluation_details=cached_details,
                )
            else:
                future = self._eval_pool.submit(self._evaluate_code, grouped[0][0].draft.code)
                futures[future] = (key, time.perf_counter())
                self.recorder.emit(
                    "evaluation_submitted",
                    generation=grouped[0][0].generation,
                    agent_id=grouped[0][0].agent_id,
                    topic="evaluation",
                    detailed_only=True,
                    evaluation_key=key,
                    grouped_candidates=len(grouped),
                )

        completed_evaluations = 0
        for future in self._completed_futures(
            futures, phase="evaluation", generation=specs[0].generation if specs else None
        ):
            key, evaluation_started = futures[future]
            try:
                fitness, evaluation_details = future.result()
            except Exception as exc:
                self.logger.debug("evaluation future failed: %s", exc)
                fitness = None
                evaluation_details = None
            evaluation_seconds = time.perf_counter() - evaluation_started
            self._evaluation_count += 1
            completed_evaluations += 1
            if fitness is not None:
                with self._cache_lock:
                    self._evaluation_cache[key] = {
                        "fitness": fitness,
                        "details": evaluation_details,
                    }
            fitness_by_key[key] = (
                fitness, False, evaluation_details, evaluation_seconds
            )
            grouped = by_key[key]
            self.recorder.emit(
                "evaluation_completed",
                generation=grouped[0][0].generation,
                agent_id=grouped[0][0].agent_id,
                topic="evaluation",
                evaluation_key=key,
                grouped_candidates=len(grouped),
                fitness=fitness,
                evaluation_details=evaluation_details,
                duration_seconds=evaluation_seconds,
                completed_in_batch=completed_evaluations,
                total_in_batch=len(futures),
            )
            if (
                completed_evaluations % self.config.progress_log_every == 0
                or completed_evaluations == len(futures)
            ):
                self.recorder.emit(
                    "evaluation_progress",
                    generation=grouped[0][0].generation,
                    topic="progress",
                    completed=completed_evaluations,
                    total=len(futures),
                    pending=len(futures) - completed_evaluations,
                )
                self.logger.info(
                    "  evaluation progress %d/%d", completed_evaluations, len(futures)
                )

        for key, grouped in by_key.items():
            _fitness, was_cached, _details, _seconds = fitness_by_key.get(
                key, (None, False, None, None)
            )
            if not was_cached and len(grouped) > 1:
                self._cache_hits += len(grouped) - 1

        result: list[ProgramIndividual] = []
        for key, grouped in by_key.items():
            fitness, cached, evaluation_details, evaluation_seconds = fitness_by_key.get(
                key, (None, False, None, None)
            )
            for grouped_index, (spec, code_hash) in enumerate(grouped):
                if fitness is None:
                    self._record_failure(spec, "evaluation_failure", "evaluation returned None")
                    continue
                operator_path = (
                    f"{spec.guidance}->{spec.variation}"
                    if spec.guidance and spec.variation
                    else spec.source_operator
                )
                individual = ProgramIndividual(
                    individual_id=self._next_individual_id(),
                    description=spec.draft.description or "",
                    code=spec.draft.code or "",
                    code_hash=code_hash,
                    fitness=fitness,
                    valid=True,
                    generation=spec.generation,
                    agent_id=spec.agent_id,
                    source_operator=spec.source_operator,
                    guidance_operator=spec.guidance,
                    variation_operator=spec.variation,
                    operator_path=operator_path,
                    parent_ids=spec.parent_ids or [],
                    source_cut_ids=spec.source_cut_ids or [],
                    prompt_hash=spec.draft.prompt_hash,
                    cached_fitness=cached or grouped_index > 0,
                    evaluation_seconds=evaluation_seconds,
                    evaluation_details=evaluation_details,
                )
                self._record_program(individual)
                result.append(individual)
        return result

    # -------------------------------------------------------------- selection

    def _fitness_key(self, individual_id: str) -> tuple[float, str]:
        individual = self.archive[individual_id]
        fitness = individual.fitness if individual.fitness is not None else math.inf
        return fitness, individual.code_hash

    def _select_survivors(self, ids: list[str], count: int) -> list[str]:
        unique_by_hash: dict[str, str] = {}
        for individual_id in ids:
            individual = self.archive[individual_id]
            current = unique_by_hash.get(individual.code_hash)
            if current is None or self._fitness_key(individual_id) < self._fitness_key(current):
                unique_by_hash[individual.code_hash] = individual_id
        return sorted(unique_by_hash.values(), key=self._fitness_key)[:count]

    def _best_id(self, ids: list[str]) -> str | None:
        valid = [item_id for item_id in ids if self.archive[item_id].fitness is not None]
        return min(valid, key=self._fitness_key) if valid else None

    def _update_global_best(self, generation: int) -> None:
        previous_id = self.gbest_id
        pbest_ids = [agent.pbest_id for agent in self.agents if agent.pbest_id]
        self.gbest_id = self._best_id(pbest_ids)
        if self.gbest_id != previous_id:
            self.recorder.emit(
                "gbest_updated",
                generation=generation,
                topic="best",
                previous_id=previous_id,
                previous_fitness=(
                    self.archive[previous_id].fitness
                    if previous_id and previous_id in self.archive
                    else None
                ),
                new_id=self.gbest_id,
                new_fitness=(
                    self.archive[self.gbest_id].fitness if self.gbest_id else None
                ),
            )

    # ---------------------------------------------------------- initialization

    def _submit_h0_round(
        self, requests: dict[int, int], round_index: int
    ) -> list[_CandidateSpec]:
        assert self._llm_pool is not None
        futures: dict[Future, tuple[int, int]] = {}
        for agent_id, count in requests.items():
            for local_index in range(count):
                sample_index = round_index * 100000 + local_index
                future = self._llm_pool.submit(
                    self.generator.generate_h0, agent_id, sample_index
                )
                futures[future] = (agent_id, sample_index)

        specs: list[_CandidateSpec] = []
        completed = 0
        for future in self._completed_futures(
            futures, phase="h0_generation", generation=0
        ):
            agent_id, _sample_index = futures[future]
            try:
                draft = future.result()
            except Exception as exc:
                draft = ProgramDraft(
                    None, None, failure_stage="llm_exception", failure_reason=str(exc)
                )
            specs.append(
                _CandidateSpec(
                    agent_id=agent_id,
                    generation=0,
                    draft=draft,
                    source_operator="H0",
                )
            )
            completed += 1
            if (
                completed % self.config.progress_log_every == 0
                or completed == len(futures)
            ):
                self.recorder.emit(
                    "h0_generation_progress",
                    generation=0,
                    topic="progress",
                    round_index=round_index,
                    completed=completed,
                    total=len(futures),
                    pending=len(futures) - completed,
                )
                self.logger.info("  H0 generation progress %d/%d", completed, len(futures))
        return specs

    def _initialize(self) -> None:
        initialization_started = time.perf_counter()
        self.logger.info(
            "[Init] agents=%d position=%d initial_multiplier=%d",
            self.config.num_agents,
            self.config.position_size,
            self.config.initial_multiplier,
        )
        by_agent: dict[int, list[str]] = {i: [] for i in range(self.config.num_agents)}
        seen_by_agent: dict[int, set[str]] = {i: set() for i in range(self.config.num_agents)}
        generated_total = 0

        for round_index in range(self.config.fill_retries + 1):
            requests: dict[int, int] = {}
            for agent_id in range(self.config.num_agents):
                missing = self.config.position_size - len(by_agent[agent_id])
                if missing <= 0:
                    continue
                requests[agent_id] = (
                    self.config.position_size * self.config.initial_multiplier
                    if round_index == 0
                    else max(missing * self.config.initial_multiplier, missing)
                )
            if not requests:
                break
            self.recorder.emit(
                "initialization_round_started",
                generation=0,
                topic="stage",
                round_index=round_index,
                requests=requests,
            )
            specs = self._submit_h0_round(requests, round_index)
            generated_total += len(specs)
            individuals = self._evaluate_specs(specs, seen_by_agent)
            for individual in individuals:
                by_agent[individual.agent_id].append(individual.individual_id)
            self.recorder.emit(
                "initialization_round_finished",
                generation=0,
                topic="stage",
                round_index=round_index,
                generated=len(specs),
                valid=len(individuals),
                valid_by_agent={
                    str(agent_id): len(values) for agent_id, values in by_agent.items()
                },
            )

        all_valid = [item_id for values in by_agent.values() for item_id in values]
        global_pool = self._select_survivors(all_valid, max(self.config.position_size, len(all_valid)))
        if len(global_pool) < self.config.position_size:
            raise RuntimeError(
                "H0 could not produce enough distinct valid programs to fill one position. "
                "Inspect failures/failures.jsonl and the LLM response format."
            )

        self.agents = []
        for agent_id in range(self.config.num_agents):
            own = self._select_survivors(by_agent[agent_id], self.config.position_size)
            if len(own) < self.config.position_size:
                own_hashes = {self.archive[item_id].code_hash for item_id in own}
                for candidate_id in global_pool:
                    if self.archive[candidate_id].code_hash not in own_hashes:
                        own.append(candidate_id)
                        own_hashes.add(self.archive[candidate_id].code_hash)
                    if len(own) == self.config.position_size:
                        break
            pbest_id = self._best_id(own)
            self.agents.append(
                AgentState(
                    agent_id=agent_id,
                    position_ids=own,
                    pbest_id=pbest_id,
                    pbest_fitness=self.archive[pbest_id].fitness if pbest_id else None,
                )
            )
        self._update_global_best(0)
        self._save_generation(0)
        self._save_checkpoint(0)
        self._write_metrics(
            0,
            generated=generated_total,
            valid=len(all_valid),
            generated_by_path={"H0": generated_total},
            valid_by_path={"H0": len(all_valid)},
            survived_by_path={"H0": sum(len(agent.position_ids) for agent in self.agents)},
            generation_seconds=time.perf_counter() - initialization_started,
        )

    # --------------------------------------------------------------- velocity

    def _compute_velocity_and_cut(
        self, generation: int
    ) -> dict[tuple[int, str], list[CutItem]]:
        pbest_snapshot = {agent.agent_id: agent.pbest_id for agent in self.agents}
        gbest_snapshot = self.gbest_id
        cuts: dict[tuple[int, str], list[CutItem]] = {}

        for agent in self.agents:
            left_agent, right_agent = ring_neighbors(agent.agent_id, len(self.agents))
            current_hashes = {
                self.archive[item_id].code_hash for item_id in agent.position_ids
            }

            def novel_experience(item_id: str | None) -> str | None:
                if item_id is None or item_id not in self.archive:
                    return None
                return (
                    None
                    if self.archive[item_id].code_hash in current_hashes
                    else item_id
                )

            for guidance in ("H1", "H2"):
                rng = random.Random(self._seed_for(generation, agent.agent_id, guidance, "velocity"))
                random_values = (rng.random(), rng.random(), rng.random(), rng.random())
                if guidance == "H1":
                    personal_weight = self.config.h1_personal
                    global_weight = self.config.h1_global
                    raw_old_velocity = dict(agent.velocity_h1)
                    old_velocity = {
                        item_id: probability
                        for item_id, probability in agent.velocity_h1.items()
                        if item_id in self.archive
                        and self.archive[item_id].code_hash not in current_hashes
                    }
                else:
                    personal_weight = self.config.h2_personal
                    global_weight = self.config.h2_global
                    raw_old_velocity = dict(agent.velocity_h2)
                    old_velocity = {
                        item_id: probability
                        for item_id, probability in agent.velocity_h2.items()
                        if item_id in self.archive
                        and self.archive[item_id].code_hash not in current_hashes
                    }

                personal_experience = novel_experience(pbest_snapshot[agent.agent_id])
                global_experience = novel_experience(gbest_snapshot)
                left_experience = novel_experience(pbest_snapshot[left_agent])
                right_experience = novel_experience(pbest_snapshot[right_agent])

                velocity = update_velocity(
                    old_velocity,
                    agent.position_ids,
                    personal_experience,
                    global_experience,
                    left_experience,
                    right_experience,
                    omega=self.config.omega,
                    personal_weight=personal_weight,
                    global_weight=global_weight,
                    left_weight=self.config.neighbor_left,
                    right_weight=self.config.neighbor_right,
                    random_values=random_values,
                )
                if guidance == "H1":
                    agent.velocity_h1 = velocity
                else:
                    agent.velocity_h2 = velocity

                rp, rg, rl, rr = random_values
                self.recorder.emit(
                    "velocity_updated",
                    generation=generation,
                    agent_id=agent.agent_id,
                    topic="velocity",
                    guidance=guidance,
                    current_position_ids=agent.position_ids,
                    left_agent_id=left_agent,
                    right_agent_id=right_agent,
                    raw_experience_ids={
                        "personal": pbest_snapshot[agent.agent_id],
                        "global": gbest_snapshot,
                        "left_neighbor": pbest_snapshot[left_agent],
                        "right_neighbor": pbest_snapshot[right_agent],
                    },
                    eligible_experience_ids={
                        "personal": personal_experience,
                        "global": global_experience,
                        "left_neighbor": left_experience,
                        "right_neighbor": right_experience,
                    },
                    coefficients={
                        "omega": self.config.omega,
                        "personal": personal_weight,
                        "global": global_weight,
                        "left_neighbor": self.config.neighbor_left,
                        "right_neighbor": self.config.neighbor_right,
                    },
                    random_values={
                        "personal": rp,
                        "global": rg,
                        "left_neighbor": rl,
                        "right_neighbor": rr,
                    },
                    contribution_probabilities={
                        "personal": (
                            min(1.0, personal_weight * rp)
                            if personal_experience else 0.0
                        ),
                        "global": (
                            min(1.0, global_weight * rg)
                            if global_experience else 0.0
                        ),
                        "left_neighbor": (
                            min(1.0, self.config.neighbor_left * rl)
                            if left_experience else 0.0
                        ),
                        "right_neighbor": (
                            min(1.0, self.config.neighbor_right * rr)
                            if right_experience else 0.0
                        ),
                    },
                    raw_old_velocity=raw_old_velocity,
                    eligible_old_velocity=old_velocity,
                    updated_velocity=velocity,
                )

                current = [CutItem(item_id, 1.0, "current") for item_id in agent.position_ids]
                current_ids = set(agent.position_ids)
                possible_extras: list[CutItem] = []
                cut_decisions = [
                    {
                        "individual_id": item.individual_id,
                        "probability": item.probability,
                        "source": "current",
                        "included": True,
                        "reason": "current_position",
                    }
                    for item in current
                ]
                for item_id, probability in velocity.items():
                    if item_id not in self.archive:
                        cut_decisions.append({
                            "individual_id": item_id, "probability": probability,
                            "source": "velocity", "included": False,
                            "reason": "unknown_individual",
                        })
                    elif probability < self.config.alpha:
                        cut_decisions.append({
                            "individual_id": item_id, "probability": probability,
                            "source": "velocity", "included": False,
                            "reason": "below_alpha",
                        })
                    elif item_id in current_ids or self.archive[item_id].code_hash in current_hashes:
                        cut_decisions.append({
                            "individual_id": item_id, "probability": probability,
                            "source": "velocity", "included": False,
                            "reason": "already_in_current_position",
                        })
                    else:
                        possible_extras.append(CutItem(item_id, probability, "velocity"))
                possible_extras.sort(key=lambda item: (-item.probability, item.individual_id))
                extras: list[CutItem] = []
                extra_hashes = set(current_hashes)
                for item in possible_extras:
                    code_hash = self.archive[item.individual_id].code_hash
                    if code_hash in extra_hashes:
                        cut_decisions.append({
                            "individual_id": item.individual_id,
                            "probability": item.probability,
                            "source": "velocity", "included": False,
                            "reason": "duplicate_code_hash",
                        })
                        continue
                    if len(extras) >= self.config.max_cut_size - len(current):
                        cut_decisions.append({
                            "individual_id": item.individual_id,
                            "probability": item.probability,
                            "source": "velocity", "included": False,
                            "reason": "cut_capacity",
                        })
                        continue
                    extras.append(item)
                    extra_hashes.add(code_hash)
                    cut_decisions.append({
                        "individual_id": item.individual_id,
                        "probability": item.probability,
                        "source": "velocity", "included": True,
                        "reason": "probability_passed",
                    })
                cut = current + extras
                cuts[(agent.agent_id, guidance)] = cut
                self.recorder.emit(
                    "cut_created",
                    generation=generation,
                    agent_id=agent.agent_id,
                    topic="cut",
                    guidance=guidance,
                    alpha=self.config.alpha,
                    max_cut_size=self.config.max_cut_size,
                    cut_ids=[item.individual_id for item in cut],
                    cut_size=len(cut),
                    decisions=cut_decisions,
                )

                velocity_path = (
                    self.output_dir / "velocities" / f"generation_{generation:04d}"
                    / f"agent_{agent.agent_id:03d}_{guidance}.json"
                )
                cut_path = (
                    self.output_dir / "cuts" / f"generation_{generation:04d}"
                    / f"agent_{agent.agent_id:03d}_{guidance}.json"
                )
                self._write_json(velocity_path, velocity)
                self._write_json(cut_path, [item.to_dict() for item in cut])
        return cuts

    # --------------------------------------------------------------- offspring

    def _select_all_cuts(
        self, generation: int, cuts: dict[tuple[int, str], list[CutItem]]
    ) -> dict[tuple[int, str], list[str]]:
        assert self._llm_pool is not None
        futures: dict[Future, tuple[tuple[int, str], float]] = {}
        archive_snapshot = dict(self.archive)
        for key, cut in cuts.items():
            agent_id, guidance = key
            future = self._llm_pool.submit(
                self.generator.select_cut,
                agent_id,
                guidance,
                cut,
                archive_snapshot,
                self.config.selection_size,
            )
            futures[future] = (key, time.perf_counter())

        selected: dict[tuple[int, str], list[str]] = {}
        completed = 0
        for future in self._completed_futures(
            futures, phase="cut_selection", generation=generation
        ):
            key, selection_started = futures[future]
            cut = cuts[key]
            allowed = {item.individual_id for item in cut}
            selection_error = None
            try:
                ids = [item_id for item_id in future.result() if item_id in allowed]
            except Exception as exc:
                self.logger.debug("cut selection failed for %s: %s", key, exc)
                ids = []
                selection_error = f"{type(exc).__name__}: {exc}"
            ids = list(dict.fromkeys(ids))
            fallback_used = len(ids) < 2
            if len(ids) < 2:
                ids = [item.individual_id for item in cut[: self.config.selection_size]]
            selected[key] = ids
            agent_id, guidance = key
            completed += 1
            self.recorder.emit(
                "cut_selection_completed",
                generation=generation,
                agent_id=agent_id,
                topic="selection",
                guidance=guidance,
                cut_ids=[item.individual_id for item in cut],
                selected_ids=ids,
                fallback_used=fallback_used,
                error=selection_error,
                duration_seconds=time.perf_counter() - selection_started,
            )
            if (
                completed % self.config.progress_log_every == 0
                or completed == len(futures)
            ):
                self.recorder.emit(
                    "cut_selection_progress",
                    topic="progress",
                    completed=completed,
                    total=len(futures),
                    pending=len(futures) - completed,
                )
        return selected

    def _generate_children(
        self,
        generation: int,
        cuts: dict[tuple[int, str], list[CutItem]],
        selected: dict[tuple[int, str], list[str]],
    ) -> list[_CandidateSpec]:
        assert self._llm_pool is not None
        futures: dict[Future, tuple[_CandidateSpec, float]] = {}

        for agent in self.agents:
            for guidance in ("H1", "H2"):
                key = (agent.agent_id, guidance)
                selected_ids = selected[key]
                selected_individuals = [self.archive[item_id] for item_id in selected_ids]
                for variation in self.config.variation_operators:
                    rng = random.Random(
                        self._seed_for(generation, agent.agent_id, guidance, variation, "parents")
                    )
                    if variation in ("H3", "H4"):
                        parent_ids = rng.sample(selected_ids, 2)
                    else:
                        parent_ids = [selected_ids[0]]
                    parents = [self.archive[item_id] for item_id in parent_ids]
                    placeholder = _CandidateSpec(
                        agent_id=agent.agent_id,
                        generation=generation,
                        draft=ProgramDraft(None, None),
                        source_operator=variation,
                        guidance=guidance,
                        variation=variation,
                        parent_ids=parent_ids,
                        source_cut_ids=[item.individual_id for item in cuts[key]],
                    )
                    future = self._llm_pool.submit(
                        self.generator.generate_variation,
                        agent.agent_id,
                        guidance,
                        variation,
                        parents,
                        selected_individuals,
                    )
                    futures[future] = (placeholder, time.perf_counter())
                    self.recorder.emit(
                        "offspring_generation_submitted",
                        generation=generation,
                        agent_id=agent.agent_id,
                        topic="candidate",
                        detailed_only=True,
                        operator_path=f"{guidance}->{variation}",
                        parent_ids=parent_ids,
                        selected_ids=selected_ids,
                    )

        specs: list[_CandidateSpec] = []
        completed = 0
        for future in self._completed_futures(
            futures, phase="offspring_generation", generation=generation
        ):
            spec, generation_started = futures[future]
            try:
                spec.draft = future.result()
            except Exception as exc:
                spec.draft = ProgramDraft(
                    None, None, failure_stage="llm_exception", failure_reason=str(exc)
                )
            specs.append(spec)
            completed += 1
            self.recorder.emit(
                "offspring_generation_completed",
                generation=generation,
                agent_id=spec.agent_id,
                topic="candidate",
                operator_path=f"{spec.guidance}->{spec.variation}",
                parent_ids=spec.parent_ids or [],
                parse_success=bool(spec.draft.code),
                failure_stage=spec.draft.failure_stage,
                llm_attempts=spec.draft.llm_attempts,
                llm_seconds=spec.draft.llm_seconds,
                wall_seconds=time.perf_counter() - generation_started,
                prompt_hash=spec.draft.prompt_hash,
            )
            if (
                completed % self.config.progress_log_every == 0
                or completed == len(futures)
            ):
                self.recorder.emit(
                    "offspring_generation_progress",
                    generation=generation,
                    topic="progress",
                    completed=completed,
                    total=len(futures),
                    pending=len(futures) - completed,
                )
                self.logger.info("  offspring progress %d/%d", completed, len(futures))
        return specs

    def _evolve_generation(self, generation: int) -> None:
        generation_started = time.perf_counter()
        self.recorder.emit(
            "generation_started",
            generation=generation,
            topic="stage",
            gbest_id=self.gbest_id,
            agent_snapshots=[agent.to_dict() for agent in self.agents],
        )
        with self.recorder.stage("velocity_and_cut", generation=generation):
            cuts = self._compute_velocity_and_cut(generation)
        with self.recorder.stage("llm_cut_selection", generation=generation):
            selected = self._select_all_cuts(generation, cuts)
        with self.recorder.stage("llm_offspring_generation", generation=generation):
            specs = self._generate_children(generation, cuts, selected)

        seen_by_agent = {
            agent.agent_id: {
                self.archive[item_id].code_hash for item_id in agent.position_ids
            }
            for agent in self.agents
        }
        with self.recorder.stage("validation_and_evaluation", generation=generation):
            children = self._evaluate_specs(specs, seen_by_agent)
        children_by_agent: dict[int, list[str]] = {agent.agent_id: [] for agent in self.agents}
        for child in children:
            children_by_agent[child.agent_id].append(child.individual_id)

        survived_by_path: Counter = Counter()
        agent_metric_records: list[dict[str, Any]] = []
        for agent in self.agents:
            old_position_ids = list(agent.position_ids)
            old_pbest_id = agent.pbest_id
            old_pbest_fitness = agent.pbest_fitness
            # The position represents the current generation, while pbest is
            # stored separately as long-term memory. Prefer new children so a
            # historical pbest can leave X_i and influence later H1 velocities.
            child_ids = children_by_agent[agent.agent_id]
            candidate_ids = (
                child_ids
                if len(child_ids) >= self.config.position_size
                else child_ids + agent.position_ids
            )
            agent.position_ids = self._select_survivors(
                candidate_ids, self.config.position_size
            )
            child_id_set = set(child_ids)
            surviving_child_ids = [
                item_id for item_id in agent.position_ids if item_id in child_id_set
            ]
            for item_id in surviving_child_ids:
                survived_by_path[self.archive[item_id].operator_path] += 1
            generation_best = self._best_id(agent.position_ids)
            if generation_best is not None:
                fitness = self.archive[generation_best].fitness
                if agent.pbest_fitness is None or (
                    fitness is not None and fitness < agent.pbest_fitness
                ):
                    agent.pbest_id = generation_best
                    agent.pbest_fitness = fitness

            position_fitnesses = [
                self.archive[item_id].fitness
                for item_id in agent.position_ids
                if self.archive[item_id].fitness is not None
            ]
            transition = {
                "old_position_ids": old_position_ids,
                "generated_child_ids": child_ids,
                "candidate_pool_ids": candidate_ids,
                "new_position_ids": list(agent.position_ids),
                "surviving_child_ids": surviving_child_ids,
                "dropped_ids": [
                    item_id for item_id in candidate_ids if item_id not in agent.position_ids
                ],
                "old_pbest_id": old_pbest_id,
                "old_pbest_fitness": old_pbest_fitness,
                "new_pbest_id": agent.pbest_id,
                "new_pbest_fitness": agent.pbest_fitness,
                "pbest_changed": agent.pbest_id != old_pbest_id,
            }
            self.recorder.emit(
                "agent_position_updated",
                generation=generation,
                agent_id=agent.agent_id,
                topic="transition",
                **transition,
            )
            agent_metric_records.append({
                "generation": generation,
                "agent_id": agent.agent_id,
                "position_ids": list(agent.position_ids),
                "position_best": min(position_fitnesses) if position_fitnesses else None,
                "position_mean": (
                    sum(position_fitnesses) / len(position_fitnesses)
                    if position_fitnesses else None
                ),
                "position_std": (
                    statistics.pstdev(position_fitnesses)
                    if len(position_fitnesses) > 1 else 0.0
                ),
                "pbest_id": agent.pbest_id,
                "pbest_fitness": agent.pbest_fitness,
                "pbest_in_position": agent.pbest_id in agent.position_ids,
                "generated_children": len(child_ids),
                "surviving_children": len(surviving_child_ids),
                "velocity_h1_size": len(agent.velocity_h1),
                "velocity_h2_size": len(agent.velocity_h2),
                "cut_h1_size": len(cuts[(agent.agent_id, "H1")]),
                "cut_h2_size": len(cuts[(agent.agent_id, "H2")]),
            })

        self._update_global_best(generation)
        self._save_generation(generation)
        self._save_checkpoint(generation)
        generated_by_path = Counter(
            f"{spec.guidance}->{spec.variation}" for spec in specs
        )
        valid_by_path = Counter(child.operator_path for child in children)
        self._write_metrics(
            generation,
            generated=len(specs),
            valid=len(children),
            generated_by_path=dict(generated_by_path),
            valid_by_path=dict(valid_by_path),
            survived_by_path=dict(survived_by_path),
            generation_seconds=time.perf_counter() - generation_started,
            agent_metric_records=agent_metric_records,
            operator_best_by_path={
                path: min(
                    child.fitness
                    for child in children
                    if child.operator_path == path and child.fitness is not None
                )
                for path in valid_by_path
            },
        )
        self.recorder.emit(
            "generation_finished",
            generation=generation,
            topic="stage",
            duration_seconds=time.perf_counter() - generation_started,
            generated=len(specs),
            valid=len(children),
            survived_by_path=dict(survived_by_path),
            gbest_id=self.gbest_id,
            gbest_fitness=(self.archive[self.gbest_id].fitness if self.gbest_id else None),
        )

    # ------------------------------------------------------------- persistence

    def _save_generation(self, generation: int) -> None:
        payload = {
            "generation": generation,
            "gbest_id": self.gbest_id,
            "gbest_fitness": (
                self.archive[self.gbest_id].fitness if self.gbest_id else None
            ),
            "agents": [
                {
                    **agent.to_dict(),
                    "position": [self.archive[item_id].to_dict() for item_id in agent.position_ids],
                }
                for agent in self.agents
            ],
        }
        self._write_json(
            self.output_dir / "agents" / f"generation_{generation:04d}.json", payload
        )
        self._save_best()
        self.recorder.emit(
            "generation_snapshot_saved",
            generation=generation,
            topic="event",
            agent_count=len(self.agents),
            gbest_id=self.gbest_id,
        )

    def _save_best(self) -> None:
        if self.gbest_id is None:
            return
        best = self.archive[self.gbest_id]
        self._write_json(self.output_dir / "best" / "best.json", best.to_dict())
        best_path = self.output_dir / "best" / "best_heuristic.py"
        temporary = best_path.with_suffix(".py.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(best.code.rstrip() + "\n")
        os.replace(temporary, best_path)

    def _save_checkpoint(self, generation: int) -> None:
        payload = {
            "generation": generation,
            "individual_counter": self._individual_counter,
            "evaluation_count": self._evaluation_count,
            "cache_hits": self._cache_hits,
            "gbest_id": self.gbest_id,
            "agents": [agent.to_dict() for agent in self.agents],
            "archive": [individual.to_dict() for individual in self.archive.values()],
            "evaluation_cache": self._evaluation_cache,
            "config": self.config.to_dict(),
        }
        self._write_json(
            self.output_dir / "checkpoints" / f"generation_{generation:04d}.json",
            payload,
        )
        self.recorder.emit(
            "checkpoint_saved",
            generation=generation,
            topic="event",
            archive_size=len(self.archive),
            evaluation_cache_size=len(self._evaluation_cache),
        )

    def _load_checkpoint(self, path: str) -> int:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        self.archive = {
            item["individual_id"]: ProgramIndividual.from_dict(item)
            for item in payload["archive"]
        }
        self.agents = [AgentState.from_dict(item) for item in payload["agents"]]
        self.gbest_id = payload.get("gbest_id")
        self._individual_counter = int(payload.get("individual_counter", len(self.archive)))
        self._evaluation_count = int(payload.get("evaluation_count", 0))
        self._cache_hits = int(payload.get("cache_hits", 0))
        self._evaluation_cache = {}
        for key, value in payload.get("evaluation_cache", {}).items():
            if isinstance(value, dict):
                self._evaluation_cache[str(key)] = value
            else:
                self._evaluation_cache[str(key)] = {
                    "fitness": float(value),
                    "details": None,
                }
        self.recorder.emit(
            "checkpoint_loaded",
            generation=int(payload["generation"]),
            topic="event",
            checkpoint_path=path,
            archive_size=len(self.archive),
            agent_count=len(self.agents),
        )
        return int(payload["generation"])

    def _write_metrics(
        self,
        generation: int,
        generated: int,
        valid: int,
        generated_by_path: dict[str, int] | None = None,
        valid_by_path: dict[str, int] | None = None,
        survived_by_path: dict[str, int] | None = None,
        generation_seconds: float | None = None,
        agent_metric_records: list[dict[str, Any]] | None = None,
        operator_best_by_path: dict[str, float] | None = None,
    ) -> None:
        fitnesses = [
            self.archive[item_id].fitness
            for agent in self.agents
            for item_id in agent.position_ids
            if self.archive[item_id].fitness is not None
        ]
        best = self.archive[self.gbest_id].fitness if self.gbest_id else None
        generated_by_path = generated_by_path or {}
        valid_by_path = valid_by_path or {}
        survived_by_path = survived_by_path or {}
        failure_counts = dict(self._failure_counts.get(generation, Counter()))
        record = {
            "generation": generation,
            "gbest_id": self.gbest_id,
            "best_fitness": best,
            "population_mean": sum(fitnesses) / len(fitnesses) if fitnesses else None,
            "population_std": statistics.pstdev(fitnesses) if len(fitnesses) > 1 else 0.0,
            "population_min": min(fitnesses) if fitnesses else None,
            "population_max": max(fitnesses) if fitnesses else None,
            "generated_candidates": generated,
            "valid_candidates": valid,
            "valid_rate": valid / generated if generated else None,
            "surviving_candidates": sum(survived_by_path.values()),
            "surviving_children": (
                sum(survived_by_path.values()) if generation > 0 else 0
            ),
            "generated_by_path": generated_by_path,
            "valid_by_path": valid_by_path,
            "survived_by_path": survived_by_path,
            "failures_by_stage": failure_counts,
            "independent_evaluations": self._evaluation_count,
            "cache_hits": self._cache_hits,
            "archive_size": len(self.archive),
            "generation_seconds": generation_seconds,
        }
        self.recorder.record_generation_metrics(record)

        if agent_metric_records is None:
            agent_metric_records = []
            for agent in self.agents:
                values = [
                    self.archive[item_id].fitness
                    for item_id in agent.position_ids
                    if self.archive[item_id].fitness is not None
                ]
                agent_metric_records.append({
                    "generation": generation,
                    "agent_id": agent.agent_id,
                    "position_ids": list(agent.position_ids),
                    "position_best": min(values) if values else None,
                    "position_mean": sum(values) / len(values) if values else None,
                    "position_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "pbest_id": agent.pbest_id,
                    "pbest_fitness": agent.pbest_fitness,
                    "pbest_in_position": agent.pbest_id in agent.position_ids,
                    "generated_children": 0,
                    "surviving_children": 0,
                    "velocity_h1_size": len(agent.velocity_h1),
                    "velocity_h2_size": len(agent.velocity_h2),
                })
        self.recorder.record_agent_metrics(agent_metric_records)

        operator_records = []
        all_paths = sorted(
            set(generated_by_path) | set(valid_by_path) | set(survived_by_path)
        )
        for path in all_paths:
            path_generated = generated_by_path.get(path, 0)
            path_valid = valid_by_path.get(path, 0)
            path_survived = survived_by_path.get(path, 0)
            operator_records.append({
                "generation": generation,
                "operator_path": path,
                "generated": path_generated,
                "valid": path_valid,
                "invalid": max(0, path_generated - path_valid),
                "valid_rate": path_valid / path_generated if path_generated else None,
                "survived": path_survived,
                "survival_rate_from_valid": (
                    path_survived / path_valid if path_valid else None
                ),
                "best_fitness": (operator_best_by_path or {}).get(path),
                "recorded_failures": self._path_failure_counts.get(
                    generation, Counter()
                ).get(path, 0),
            })
        self.recorder.record_operator_metrics(operator_records)
        self.recorder.emit(
            "generation_metrics_recorded",
            generation=generation,
            topic="event",
            metrics=record,
        )
        self.logger.info(
            "--- Generation %d/%d best=%s valid=%d/%d evals=%d cache=%d",
            generation,
            self.config.generations,
            best,
            valid,
            generated,
            self._evaluation_count,
            self._cache_hits,
        )
        for agent in self.agents:
            self.logger.info(
                "  agent=%02d pbest=%s fitness=%s position=%s",
                agent.agent_id,
                agent.pbest_id,
                agent.pbest_fitness,
                ",".join(agent.position_ids),
            )

    # ------------------------------------------------------------------- run

    def run(self) -> dict[str, Any]:
        started = time.time()
        self.logger.info("=" * 68)
        self.logger.info("MA-EoH multi-agent heuristic evolution")
        self.logger.info(
            "agents=%d position=%d generations=%d llm_workers=%d evaluators=%d",
            self.config.num_agents,
            self.config.position_size,
            self.config.generations,
            self.config.num_llm_workers,
            self.config.num_evaluators,
        )
        self.logger.info("=" * 68)
        self.recorder.emit(
            "run_started",
            topic="event",
            config=self.config.to_dict(),
        )
        try:
            self._llm_pool = ThreadPoolExecutor(
                max_workers=self.config.num_llm_workers, thread_name_prefix="mae-llm"
            )
            self._eval_pool = ThreadPoolExecutor(
                max_workers=self.config.num_evaluators, thread_name_prefix="mae-eval"
            )
            if self.config.resume_from:
                with self.recorder.stage("checkpoint_restore"):
                    last_generation = self._load_checkpoint(self.config.resume_from)
                start_generation = last_generation + 1
                self.logger.info("Resumed from generation %d", last_generation)
            else:
                with self.recorder.stage("initialization", generation=0):
                    self._initialize()
                start_generation = 1

            for generation in range(start_generation, self.config.generations + 1):
                self._evolve_generation(generation)

            elapsed = time.time() - started
            best = self.archive[self.gbest_id] if self.gbest_id else None
            summary = {
                "best_id": best.individual_id if best else None,
                "best_fitness": best.fitness if best else None,
                "best_code_path": str(self.output_dir / "best" / "best_heuristic.py"),
                "archive_size": len(self.archive),
                "independent_evaluations": self._evaluation_count,
                "cache_hits": self._cache_hits,
                "elapsed_seconds": elapsed,
                "output_dir": str(self.output_dir),
            }
            self._write_json(self.output_dir / "metadata.json", summary)
            self.recorder.emit("run_finished", topic="event", summary=summary)
            self.logger.info(
                "Finished best=%s elapsed=%.1fs", summary["best_fitness"], elapsed
            )
            return summary
        except Exception as exc:
            self.recorder.emit(
                "run_failed",
                topic="error",
                exception_type=type(exc).__name__,
                exception=str(exc),
                elapsed_seconds=time.time() - started,
            )
            self.logger.exception("MA-EoH failed: %s", exc)
            raise
        finally:
            if self._llm_pool is not None:
                self._llm_pool.shutdown(wait=True, cancel_futures=False)
            if self._eval_pool is not None:
                self._eval_pool.shutdown(wait=True, cancel_futures=False)
            self._llm_pool = None
            self._eval_pool = None
            self._close_logger()
