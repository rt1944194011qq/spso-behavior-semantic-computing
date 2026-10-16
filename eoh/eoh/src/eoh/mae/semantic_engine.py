"""Single-program MA-EoH with a persistent semantic-difference velocity bank.

The original five-program engine remains in ``legacy_engine.py`` for
reproducing old experiments. This is the default engine exported by
``eoh.mae``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .legacy_engine import MAEEngine as LegacyMAEEngine, _CandidateSpec
from .experience import (
    birth_probability,
    decay_and_prune,
    prune_velocity,
    sample_cut,
    scheduled_guidance,
)
from .generator import ProgramDraft
from .models import ExperienceCutItem, ExperienceRecord, SemanticAgentState
from .topology import ring_neighbors


@dataclass
class _CandidateJob:
    spec: _CandidateSpec
    generate: Callable
    args: tuple


class MAEEngine(LegacyMAEEngine):
    """Asynchronous candidate pipelines with generation-frozen references."""

    CHECKPOINT_VERSION = 3
    _RESUME_RUNTIME_FIELDS = {
        "generations", "output_dir", "resume_from", "debug",
        "detailed_logging", "progress_log_every",
        "heartbeat_interval_seconds", "record_full_prompts",
        "record_full_responses", "num_llm_workers", "num_evaluators",
    }

    def _validate_resume_config(self, prior: dict[str, Any]) -> None:
        current = self.config.to_dict()
        mismatches = [
            key for key, value in current.items()
            if key not in self._RESUME_RUNTIME_FIELDS and prior.get(key) != value
        ]
        if mismatches:
            raise ValueError(f"resume config mismatch: {', '.join(mismatches)}")

    def _prepare_output(self) -> None:
        if not self.config.resume_from:
            super()._prepare_output()
            return
        checkpoint_path = Path(self.config.resume_from).resolve()
        if checkpoint_path.parent.parent != self.output_dir:
            raise ValueError("resume checkpoint must belong to the selected output_dir")
        with checkpoint_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != self.CHECKPOINT_VERSION:
            raise ValueError("older checkpoints cannot be resumed by semantic MA-EoH v3")
        self._validate_resume_config(payload.get("config", {}))
        if not (self.output_dir / "config.json").is_file():
            raise FileNotFoundError("resume run is missing config.json")
        # Resume only appends run records. The original configuration remains
        # intact even if the target generation or worker count changes.

    def __init__(self, config, problem, generator, validator=None) -> None:
        if config.position_size != 1:
            raise ValueError("semantic MA-EoH requires position_size=1")
        for method in ("describe_difference", "apply_experience"):
            if not callable(getattr(generator, method, None)):
                raise TypeError(f"generator must implement {method}()")
        super().__init__(config, problem, generator, validator)
        self.agents: list[SemanticAgentState] = []
        self._difference_cache: dict[str, str] = {}

    def _fitness_key(self, individual_id: str) -> tuple[float, str]:
        individual = self.archive[individual_id]
        fitness = individual.fitness
        if fitness is None:
            return math.inf, individual.code_hash
        rank = fitness if self.config.objective_direction == "min" else -fitness
        return rank, individual.code_hash

    def _initialize(self) -> None:
        started = time.perf_counter()
        by_agent: dict[int, list[str]] = {i: [] for i in range(self.config.num_agents)}
        seen: dict[int, set[str]] = {i: set() for i in by_agent}
        generated = 0
        for round_index in range(self.config.fill_retries + 1):
            missing = [agent_id for agent_id, ids in by_agent.items() if not ids]
            if not missing:
                break
            requests = {
                agent_id: self.config.initial_multiplier for agent_id in missing
            }
            if self.config.async_pipeline:
                jobs = [
                    _CandidateJob(
                        _CandidateSpec(agent_id, 0, ProgramDraft(None, None), "H0"),
                        self.generator.generate_h0,
                        (agent_id, round_index * 100000 + local_index),
                    )
                    for agent_id, count in requests.items()
                    for local_index in range(count)
                ]
                specs, programs = self._run_candidate_jobs(jobs, seen, generation=0)
            else:
                specs = self._submit_h0_round(requests, round_index)
                programs = self._evaluate_specs(specs, seen)
            generated += len(specs)
            for item in programs:
                by_agent[item.agent_id].append(item.individual_id)

        all_valid = [item_id for ids in by_agent.values() for item_id in ids]
        if not all_valid:
            raise RuntimeError("H0 produced no valid program; inspect failures/failures.jsonl")
        global_pool = self._select_survivors(all_valid, len(all_valid))
        occupied_hashes: set[str] = set()
        for agent_id in range(self.config.num_agents):
            candidates = self._select_survivors(by_agent[agent_id], len(by_agent[agent_id]))
            candidates += [item_id for item_id in global_pool if item_id not in candidates]
            preferred = next(
                (item_id for item_id in candidates
                 if self.archive[item_id].code_hash not in occupied_hashes),
                candidates[0],
            )
            occupied_hashes.add(self.archive[preferred].code_hash)
            fitness = self.archive[preferred].fitness
            assert fitness is not None
            self.agents.append(
                SemanticAgentState(agent_id, preferred, preferred, fitness)
            )
        self._update_global_best(0)
        self._save_generation(0)
        self._save_checkpoint(0)
        self._write_semantic_metrics(
            0,
            generated=generated,
            valid=len(all_valid),
            generated_by_path={"H0": generated},
            valid_by_path={"H0": len(all_valid)},
            survived_by_path={"H0": len(self.agents)},
            generation_seconds=time.perf_counter() - started,
        )

    def _frozen_references(self) -> dict[int, dict[str, str]]:
        if self.gbest_id is None:
            raise RuntimeError("global best is not initialized")
        frozen: dict[int, dict[str, str]] = {}
        for agent in self.agents:
            left, right = ring_neighbors(agent.agent_id, len(self.agents))
            frozen[agent.agent_id] = {
                "personal": agent.pbest_id,
                "global": self.gbest_id,
                "left": self.agents[left].pbest_id,
                "right": self.agents[right].pbest_id,
            }
        return frozen

    def _rank_fitness(self, value: float) -> float:
        return value if self.config.objective_direction == "min" else -value

    def _refresh_velocity(
        self, generation: int, references: dict[int, dict[str, str]]
    ) -> dict[int, int]:
        """Compare the current program first; decay old entries, then merge fresh ones."""
        assert self._llm_pool is not None
        comparisons: list[tuple[int, str, str, str, str, str]] = []
        pending: dict[str, Future] = {}
        for agent in self.agents:
            current = self.archive[agent.current_id]
            for role, reference_id in references[agent.agent_id].items():
                reference = self.archive[reference_id]
                if current.code_hash == reference.code_hash:
                    continue
                direction_type = (
                    "improve"
                    if self._rank_fitness(float(reference.fitness))
                    < self._rank_fitness(float(current.fitness))
                    else "diversify"
                )
                cache_key = (
                    f"{direction_type}:{current.code_hash}:{reference.code_hash}"
                )
                comparisons.append(
                    (agent.agent_id, role, current.individual_id, reference.individual_id,
                     cache_key, direction_type)
                )
                if cache_key not in self._difference_cache and cache_key not in pending:
                    pending[cache_key] = self._llm_pool.submit(
                        self.generator.describe_difference,
                        agent.agent_id, current, reference, direction_type,
                    )

        future_keys = {future: key for key, future in pending.items()}
        for future in self._completed_futures(
            future_keys, phase="experience_comparison", generation=generation
        ):
            key = future_keys[future]
            try:
                prompt = future.result()
            except Exception as exc:
                self.logger.debug("experience comparison failed: %s", exc)
                prompt = None
            if prompt and prompt.strip():
                self._difference_cache[key] = " ".join(prompt.strip().split())[:700]

        ranked_fitnesses = [
            self._rank_fitness(float(self.archive[item_id].fitness))
            for agent in self.agents
            for item_id in [agent.current_id, *references[agent.agent_id].values()]
        ]
        best_rank = min(ranked_fitnesses)
        positive_gaps = [value - best_rank for value in ranked_fitnesses if value > best_rank]
        scale = (
            statistics.median(positive_gaps)
            if positive_gaps else max(abs(best_rank) * 0.1, 1e-6)
        )
        added_by_agent: dict[int, int] = {}
        by_agent: dict[int, list[tuple[str, str, str, str, str]]] = {
            agent.agent_id: [] for agent in self.agents
        }
        for agent_id, role, source_id, target_id, cache_key, direction_type in comparisons:
            by_agent[agent_id].append(
                (role, source_id, target_id, cache_key, direction_type)
            )

        for agent in self.agents:
            before = len(agent.velocity)
            agent.velocity = decay_and_prune(
                agent.velocity,
                half_life=self.config.experience_half_life,
                drop_below=self.config.experience_drop_below,
                capacity=self.config.experience_capacity,
            )
            added = 0
            for role, source_id, target_id, cache_key, direction_type in by_agent[agent.agent_id]:
                difference_prompt = self._difference_cache.get(cache_key)
                if not difference_prompt:
                    self.recorder.emit(
                        "experience_comparison_failed",
                        generation=generation, agent_id=agent.agent_id,
                        topic="velocity", source_role=role,
                        source_id=source_id, target_id=target_id,
                    )
                    continue
                source = self.archive[source_id]
                target = self.archive[target_id]
                experience_id = hashlib.sha256(
                    f"{role}|{direction_type}|{source.code_hash}|{target.code_hash}".encode("utf-8")
                ).hexdigest()[:24]
                source_fitness = float(source.fitness)
                target_fitness = float(target.fitness)
                probability = birth_probability(
                    self._rank_fitness(source_fitness),
                    self._rank_fitness(target_fitness),
                    best_rank,
                    scale,
                    self.config.experience_birth_min,
                    self.config.experience_birth_max,
                    self.config.experience_target_weight,
                )
                previous = agent.velocity.get(experience_id)
                item = ExperienceRecord(
                    experience_id=experience_id,
                    source_role=role,
                    source_id=source_id,
                    target_id=target_id,
                    difference_prompt=difference_prompt,
                    source_fitness=source_fitness,
                    target_fitness=target_fitness,
                    probability=max(probability, previous.probability) if previous else probability,
                    created_generation=(previous.created_generation if previous else generation),
                    last_seen_generation=generation,
                    observations=(previous.observations + 1 if previous else 1),
                    direction_type=direction_type,
                )
                agent.velocity[experience_id] = item
                added += 1
                self._append_jsonl(
                    self.output_dir / "velocities" / "experience_events.jsonl",
                    {"generation": generation, "agent_id": agent.agent_id,
                     "event": "refreshed" if previous else "created", **item.to_dict()},
                )
            agent.velocity = prune_velocity(
                agent.velocity, self.config.experience_capacity
            )
            added_by_agent[agent.agent_id] = added
            self._write_json(
                self.output_dir / "velocities" / f"generation_{generation:04d}"
                / f"agent_{agent.agent_id:03d}.json",
                [item.to_dict() for item in agent.velocity.values()],
            )
            self.recorder.emit(
                "velocity_updated", generation=generation, agent_id=agent.agent_id,
                topic="velocity", old_size=before, new_size=len(agent.velocity),
                refreshed=added, comparisons=len(by_agent[agent.agent_id]),
            )
        return added_by_agent

    def _draw_cuts(
        self, generation: int, guidance_by_agent: dict[int, str]
    ) -> tuple[dict[int, list[ExperienceCutItem]], dict[int, dict[str, Any]]]:
        cuts = {}
        diagnostics = {}
        for agent in self.agents:
            guidance = guidance_by_agent[agent.agent_id]
            cut, info = sample_cut(
                agent.velocity,
                target_size=self.config.cut_target_size,
                neighbor_reserved_slots=self.config.neighbor_reserved_slots,
                generation=generation,
                agent_id=agent.agent_id,
                enabled_operators=(guidance,),
                h1_boost=self.config.h1_selection_boost,
                h2_boost=self.config.h2_selection_boost,
                rng=random.Random(self._seed_for(generation, agent.agent_id, "cut")),
            )
            cuts[agent.agent_id] = cut
            diagnostics[agent.agent_id] = info
            payload = {"generation": generation, "agent_id": agent.agent_id,
                       "guidance": guidance,
                       "cut": [item.to_dict() for item in cut], **info}
            self._write_json(
                self.output_dir / "cuts" / f"generation_{generation:04d}"
                / f"agent_{agent.agent_id:03d}.json", payload,
            )
            self.recorder.emit(
                "cut_created", generation=generation, agent_id=agent.agent_id,
                topic="cut", cut=payload["cut"], lambda_value=info["lambda"],
                eligible=info["eligible"], target=info["target"],
                reserved=info["reserved"], guidance=guidance,
            )
        return cuts, diagnostics

    def _finish_draft_futures(
        self,
        futures: dict[Future, _CandidateSpec],
        *,
        phase: str,
        generation: int,
    ) -> list[_CandidateSpec]:
        specs = []
        for future in self._completed_futures(
            futures, phase=phase, generation=generation
        ):
            spec = futures[future]
            try:
                spec.draft = future.result()
            except Exception as exc:
                spec.draft = ProgramDraft(
                    None, None, failure_stage="llm_exception", failure_reason=str(exc)
                )
            specs.append(spec)
        return specs

    def _run_candidate_jobs(
        self, jobs: list[_CandidateJob], seen_by_agent: dict[int, set[str]],
        *, generation: int, on_completed: Callable | None = None,
    ) -> tuple[list[_CandidateSpec], list[Any]]:
        """EoH-style bounded samplers: generate -> evaluate -> immediately refill.

        Sampler threads only generate/validate/evaluate/register candidates.
        Agent pools and operator transitions are owned by this calling thread.
        This avoids concurrent mutations of an agent's selection state.
        """
        specs: list[_CandidateSpec] = []
        programs = []
        pending: dict[Future, _CandidateJob] = {}
        started = time.perf_counter()
        completed = 0

        def execute(job: _CandidateJob, submitted: float):
            self.recorder.emit(
                "candidate_pipeline_started", generation=generation,
                agent_id=job.spec.agent_id, topic="progress",
                source_operator=job.spec.source_operator,
                sampler_queue_seconds=time.perf_counter() - submitted,
            )
            try:
                job.spec.draft = job.generate(*job.args)
            except Exception as exc:
                job.spec.draft = ProgramDraft(
                    None, None, failure_stage="llm_exception", failure_reason=str(exc)
                )
            result = self._evaluate_specs(
                [job.spec], seen_by_agent, report_progress=False
            )
            return result[0] if result else None

        with ThreadPoolExecutor(
            max_workers=self.config.num_llm_workers, thread_name_prefix="mae-sampler"
        ) as samplers:
            def submit(job: _CandidateJob) -> None:
                specs.append(job.spec)
                future = samplers.submit(execute, job, time.perf_counter())
                pending[future] = job

            for job in jobs:
                submit(job)
            while pending:
                done, _ = wait(
                    pending, timeout=self.config.heartbeat_interval_seconds,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    elapsed = time.perf_counter() - started
                    self.recorder.emit(
                        "phase_heartbeat", generation=generation, topic="progress",
                        phase="async_pipeline", pending=len(pending),
                        total=len(specs), completed=completed, elapsed_seconds=elapsed,
                    )
                    self.logger.info(
                        "  heartbeat phase=async_pipeline pending=%d completed=%d/%d elapsed=%.1fs",
                        len(pending), completed, len(specs), elapsed,
                    )
                for future in done:
                    job = pending.pop(future)
                    program = future.result()
                    completed += 1
                    if program is not None:
                        programs.append(program)
                    self.recorder.emit(
                        "candidate_pipeline_finished", generation=generation,
                        agent_id=job.spec.agent_id, topic="progress",
                        source_operator=job.spec.source_operator,
                        individual_id=program.individual_id if program else None,
                    )
                    if on_completed is not None:
                        for next_job in on_completed(job.spec, program):
                            submit(next_job)
                    if completed % self.config.progress_log_every == 0 or not pending:
                        self.logger.info(
                            "  async pipeline progress %d/%d pending=%d",
                            completed, len(specs), len(pending),
                        )
        return specs, programs

    def _base_jobs(
        self, generation: int, cuts: dict[int, list[ExperienceCutItem]],
        guidance_by_agent: dict[int, str],
    ) -> list[_CandidateJob]:
        jobs = []
        for agent in self.agents:
            current = self.archive[agent.current_id]
            guidance = guidance_by_agent[agent.agent_id]
            for cut_item in cuts[agent.agent_id]:
                experience = agent.velocity[cut_item.experience_id]
                exemplar = self.archive[experience.target_id]
                spec = _CandidateSpec(
                    agent_id=agent.agent_id, generation=generation,
                    draft=ProgramDraft(None, None), source_operator=guidance,
                    guidance=guidance, parent_ids=[agent.current_id],
                    source_cut_ids=[experience.experience_id],
                )
                jobs.append(_CandidateJob(
                    spec, self.generator.apply_experience,
                    (agent.agent_id, current, experience, exemplar, guidance),
                ))
            # Fewer than the configured minimum *real* experiences triggers
            # separate H0 exploration; it does not fabricate cut entries.
            for fallback_index in range(
                max(0, self.config.cut_min_size - len(cuts[agent.agent_id]))
            ):
                spec = _CandidateSpec(
                    agent_id=agent.agent_id, generation=generation,
                    draft=ProgramDraft(None, None), source_operator="H0",
                )
                jobs.append(_CandidateJob(
                    spec, self.generator.generate_h0,
                    (agent.agent_id, generation * 100000 + agent.agent_id * 100 + fallback_index),
                ))
        return jobs

    def _generate_base(self, generation, cuts, guidance_by_agent):
        assert self._llm_pool is not None
        futures = {
            self._llm_pool.submit(job.generate, *job.args): job.spec
            for job in self._base_jobs(generation, cuts, guidance_by_agent)
        }
        return self._finish_draft_futures(
            futures, phase="base_generation", generation=generation
        )

    def _ensure_base_pool(
        self, generation: int, pool_by_agent: dict[int, list[str]],
        seen_by_agent: dict[int, set[str]],
    ) -> tuple[list[_CandidateSpec], list[Any]]:
        """Try H0 repairs when only the original program survived the cut stage."""
        assert self._llm_pool is not None
        all_specs: list[_CandidateSpec] = []
        all_programs = []
        for retry in range(self.config.fill_retries + 1):
            missing = [agent for agent in self.agents
                       if len(pool_by_agent[agent.agent_id]) < 2]
            if not missing:
                break
            futures: dict[Future, _CandidateSpec] = {}
            for agent in missing:
                future = self._llm_pool.submit(
                    self.generator.generate_h0, agent.agent_id,
                    generation * 1000000 + agent.agent_id * 100 + retry,
                )
                futures[future] = _CandidateSpec(
                    agent_id=agent.agent_id, generation=generation,
                    draft=ProgramDraft(None, None), source_operator="H0",
                )
            specs = self._finish_draft_futures(
                futures, phase="base_pool_repair", generation=generation,
            )
            programs = self._evaluate_specs(specs, seen_by_agent)
            all_specs.extend(specs)
            all_programs.extend(programs)
            for program in programs:
                pool_by_agent[program.agent_id].append(program.individual_id)
        return all_specs, all_programs

    def _operator_jobs(
        self, generation: int, variation: str,
        pool_by_agent: dict[int, list[str]],
        cuts: dict[int, list[ExperienceCutItem]],
        guidance_by_agent: dict[int, str],
        agent_ids: set[int] | None = None,
    ) -> list[_CandidateJob]:
        """Make one draft per agent from the latest evaluated pool."""
        jobs = []
        for agent in self.agents:
            if agent_ids is not None and agent.agent_id not in agent_ids:
                continue
            pool_ids = pool_by_agent[agent.agent_id]
            ranked = sorted(pool_ids, key=self._fitness_key)
            if variation in {"H3", "H4"}:
                if len(ranked) < 2:
                    self.recorder.emit(
                        "operator_skipped_insufficient_parents",
                        generation=generation, agent_id=agent.agent_id,
                        topic="candidate", operator=variation,
                        pool_size=len(ranked),
                    )
                    continue
                parent_ids = ranked[:2] if variation == "H3" else ranked[-2:][::-1]
            elif variation in {"H5", "H6"}:
                rng = random.Random(
                    self._seed_for(generation, agent.agent_id, variation, "parent")
                )
                parent_ids = [rng.choice(ranked)]
            else:
                arity = (self.config.variation_parent_counts or {}).get(variation)
                if arity is None:
                    raise ValueError(f"operator {variation} needs a parent count")
                if len(pool_ids) < arity:
                    self.recorder.emit(
                        "operator_skipped_insufficient_parents",
                        generation=generation, agent_id=agent.agent_id,
                        topic="candidate", operator=variation,
                        pool_size=len(pool_ids), required=arity,
                    )
                    continue
                rng = random.Random(
                    self._seed_for(generation, agent.agent_id, variation, "parents")
                )
                parent_ids = rng.sample(ranked, arity)
            current_cut_ids = {
                item.experience_id for item in cuts[agent.agent_id]
            }
            source_ids = sorted({
                experience_id for parent_id in parent_ids
                for experience_id in self.archive[parent_id].source_cut_ids
                if experience_id in current_cut_ids
            })
            parents = [self.archive[parent_id] for parent_id in parent_ids]
            pool_programs = [self.archive[item_id] for item_id in pool_ids]
            guidance = guidance_by_agent[agent.agent_id]
            spec = _CandidateSpec(
                agent_id=agent.agent_id, generation=generation,
                draft=ProgramDraft(None, None), source_operator=variation,
                guidance=guidance, variation=variation,
                parent_ids=parent_ids, source_cut_ids=source_ids,
            )
            jobs.append(_CandidateJob(
                spec, self.generator.generate_variation,
                (agent.agent_id, guidance, variation, parents, pool_programs),
            ))
        return jobs

    def _generate_operator(self, generation, variation, pool_by_agent, cuts, guidance_by_agent):
        assert self._llm_pool is not None
        futures = {
            self._llm_pool.submit(job.generate, *job.args): job.spec
            for job in self._operator_jobs(
                generation, variation, pool_by_agent, cuts, guidance_by_agent
            )
        }
        return self._finish_draft_futures(
            futures, phase=f"{variation}_generation", generation=generation
        )

    def _admit_operator_candidates(
        self, generation: int, variation: str,
        specs: list[_CandidateSpec], programs: list[Any],
        pool_by_agent: dict[int, list[str]],
    ) -> Counter:
        """Replace the worst only on a strict fitness improvement."""
        by_agent = {program.agent_id: program for program in programs}
        admitted: Counter = Counter()
        for spec in specs:
            pool = pool_by_agent[spec.agent_id]
            worst_id = max(pool, key=self._fitness_key)
            child = by_agent.get(spec.agent_id)
            worst_fitness = float(self.archive[worst_id].fitness)
            accepted = (
                child is not None and
                self._rank_fitness(float(child.fitness))
                < self._rank_fitness(worst_fitness)
            )
            before = list(pool)
            if accepted:
                pool[pool.index(worst_id)] = child.individual_id
                admitted[child.operator_path] += 1
            decision = {
                "generation": generation, "agent_id": spec.agent_id,
                "operator": variation, "guidance": spec.guidance,
                "parent_ids": spec.parent_ids or [],
                "candidate_id": child.individual_id if child else None,
                "candidate_fitness": child.fitness if child else None,
                "worst_id": worst_id, "worst_fitness": worst_fitness,
                "admitted": accepted,
                "replaced_id": worst_id if accepted else None,
                "pool_before": before, "pool_after": list(pool),
            }
            self._append_jsonl(
                self.output_dir / "metrics" / "pool_decisions.jsonl", decision,
            )
            self.recorder.emit(
                "pool_candidate_compared", generation=generation,
                agent_id=spec.agent_id, topic="transition",
                **{key: value for key, value in decision.items()
                   if key not in {"generation", "agent_id"}},
            )
        return admitted

    def _evolve_async_pools(self, generation, cuts, guidance_by_agent, seen_by_agent):
        pool_by_agent = {
            agent.agent_id: [agent.current_id] for agent in self.agents
        }
        jobs = self._base_jobs(generation, cuts, guidance_by_agent)
        remaining_base = Counter(job.spec.agent_id for job in jobs)
        repair_attempts = Counter()
        operator_index = {agent.agent_id: -1 for agent in self.agents}
        stage_started: dict[tuple[int, str], float] = {}
        base_pool_sizes = {}
        admitted_by_path = Counter()
        admitted_by_agent = Counter()
        ordered = [name for name in ("H3", "H4", "H5", "H6")
                   if name in self.config.enabled_operators]
        ordered += [name for name in self.config.enabled_operators
                    if name not in {"H1", "H2", "H3", "H4", "H5", "H6"}]

        def advance(agent_id):
            while operator_index[agent_id] + 1 < len(ordered):
                operator_index[agent_id] += 1
                name = ordered[operator_index[agent_id]]
                next_jobs = self._operator_jobs(
                    generation, name, pool_by_agent, cuts, guidance_by_agent,
                    agent_ids={agent_id},
                )
                if next_jobs:
                    stage_started[agent_id, name] = time.perf_counter()
                    self.recorder.emit(
                        "stage_started", generation=generation, agent_id=agent_id,
                        topic="stage", stage=name, asynchronous=True,
                    )
                    return next_jobs
            self.recorder.emit(
                "agent_pipeline_finished", generation=generation,
                agent_id=agent_id, topic="progress",
            )
            return []

        def finish(spec, program):
            agent_id = spec.agent_id
            pool = pool_by_agent[agent_id]
            if spec.variation is None:
                if program is not None:
                    pool.append(program.individual_id)
                remaining_base[agent_id] -= 1
                if remaining_base[agent_id]:
                    return []
                if len(pool) < 2 and repair_attempts[agent_id] <= self.config.fill_retries:
                    retry = repair_attempts[agent_id]
                    repair_attempts[agent_id] += 1
                    remaining_base[agent_id] = 1
                    return [_CandidateJob(
                        _CandidateSpec(agent_id, generation, ProgramDraft(None, None), "H0"),
                        self.generator.generate_h0,
                        (agent_id, generation * 1000000 + agent_id * 100 + retry),
                    )]
                base_pool_sizes[agent_id] = len(pool)
                self._append_jsonl(
                    self.output_dir / "metrics" / "base_pools.jsonl",
                    {"generation": generation, "agent_id": agent_id,
                     "guidance": guidance_by_agent[agent_id],
                     "original_id": self.agents[agent_id].current_id,
                     "cut_size": len(cuts[agent_id]), "pool_ids": list(pool),
                     "pool_fitness": [self.archive[item_id].fitness for item_id in pool]},
                )
            else:
                admitted = self._admit_operator_candidates(
                    generation, spec.variation, [spec],
                    [program] if program is not None else [], pool_by_agent,
                )
                admitted_by_path.update(admitted)
                if program is not None and program.individual_id in pool:
                    admitted_by_agent[agent_id] += 1
                self.recorder.emit(
                    "stage_finished", generation=generation, agent_id=agent_id,
                    topic="stage", stage=spec.variation, asynchronous=True,
                    duration_seconds=time.perf_counter() - stage_started[agent_id, spec.variation],
                )
            return advance(agent_id)

        with self.recorder.stage("asynchronous_evolution", generation=generation):
            specs, programs = self._run_candidate_jobs(
                jobs, seen_by_agent, generation=generation, on_completed=finish,
            )
        base_specs = [spec for spec in specs if spec.variation is None]
        variation_specs = [spec for spec in specs if spec.variation is not None]
        base_programs = [program for program in programs if program.variation_operator is None]
        variation_programs = [program for program in programs if program.variation_operator is not None]
        return (base_specs, base_programs, variation_specs, variation_programs,
                base_pool_sizes, admitted_by_path, admitted_by_agent, pool_by_agent)

    def _evolve_generation(self, generation: int) -> None:
        started = time.perf_counter()
        snapshot = self._frozen_references()
        old_states = {agent.agent_id: agent.to_dict() for agent in self.agents}
        guidance_by_agent, p_h1 = scheduled_guidance(
            [agent.agent_id for agent in self.agents],
            generation=generation,
            total_generations=self.config.generations,
            start=self.config.h1_probability_start,
            end=self.config.h1_probability_end,
            enabled_operators=self.config.enabled_operators,
            rng=random.Random(self._seed_for(generation, "guidance")),
        )
        self.recorder.emit(
            "generation_started", generation=generation, topic="stage",
            gbest_id=self.gbest_id, agent_snapshots=list(old_states.values()),
            guidance_by_agent=guidance_by_agent, h1_probability=p_h1,
        )
        self.logger.info(
            "  guidance H1=%d H2=%d p_H1=%.4f",
            sum(value == "H1" for value in guidance_by_agent.values()),
            sum(value == "H2" for value in guidance_by_agent.values()), p_h1,
        )
        with self.recorder.stage("experience_update", generation=generation):
            added = self._refresh_velocity(generation, snapshot)
        with self.recorder.stage("bounded_cut", generation=generation):
            cuts, cut_info = self._draw_cuts(generation, guidance_by_agent)
        seen_by_agent = {
            agent.agent_id: {self.archive[agent.current_id].code_hash}
            for agent in self.agents
        }
        if self.config.async_pipeline:
            (base_specs, base_programs, variation_specs, variation_programs,
             base_pool_sizes, admitted_by_path, admitted_by_agent,
             pool_by_agent) = self._evolve_async_pools(
                generation, cuts, guidance_by_agent, seen_by_agent,
            )
        else:
            with self.recorder.stage("difference_application", generation=generation):
                base_specs = self._generate_base(generation, cuts, guidance_by_agent)
                base_programs = self._evaluate_specs(base_specs, seen_by_agent)
            # The evaluated current program participates without a redundant evaluation.
            pool_by_agent: dict[int, list[str]] = {
                agent.agent_id: [agent.current_id] for agent in self.agents
            }
            for child in base_programs:
                pool_by_agent[child.agent_id].append(child.individual_id)
            with self.recorder.stage("base_pool_repair", generation=generation):
                repair_specs, repair_programs = self._ensure_base_pool(
                    generation, pool_by_agent, seen_by_agent,
                )
            base_specs += repair_specs
            base_programs += repair_programs
            base_pool_sizes = {
                agent_id: len(pool) for agent_id, pool in pool_by_agent.items()
            }
            for agent in self.agents:
                pool = pool_by_agent[agent.agent_id]
                self._append_jsonl(
                    self.output_dir / "metrics" / "base_pools.jsonl",
                    {"generation": generation, "agent_id": agent.agent_id,
                     "guidance": guidance_by_agent[agent.agent_id],
                     "original_id": agent.current_id,
                     "cut_size": len(cuts[agent.agent_id]),
                     "pool_ids": list(pool),
                     "pool_fitness": [self.archive[item_id].fitness for item_id in pool]},
                )
            variation_specs: list[_CandidateSpec] = []
            variation_programs = []
            admitted_by_path: Counter = Counter()
            admitted_by_agent: Counter = Counter()
            ordered_operators = [
                name for name in ("H3", "H4", "H5", "H6")
                if name in self.config.enabled_operators
            ] + [
                name for name in self.config.enabled_operators
                if name not in {"H1", "H2", "H3", "H4", "H5", "H6"}
            ]
            with self.recorder.stage("variation_generation", generation=generation):
                for variation in ordered_operators:
                    with self.recorder.stage(variation, generation=generation):
                        specs = self._generate_operator(
                            generation, variation, pool_by_agent, cuts,
                            guidance_by_agent,
                        )
                        programs = self._evaluate_specs(specs, seen_by_agent)
                        admitted = self._admit_operator_candidates(
                            generation, variation, specs, programs, pool_by_agent,
                        )
                    variation_specs.extend(specs)
                    variation_programs.extend(programs)
                    admitted_by_path.update(admitted)
                    for child in programs:
                        if child.individual_id in pool_by_agent[child.agent_id]:
                            admitted_by_agent[child.agent_id] += 1

        all_specs = base_specs + variation_specs
        all_programs = base_programs + variation_programs
        for child in all_programs:
            for experience_id in child.source_cut_ids:
                self._append_jsonl(
                    self.output_dir / "lineage" / "experience_edges.jsonl",
                    {"generation": generation, "agent_id": child.agent_id,
                     "experience_id": experience_id,
                     "child_id": child.individual_id,
                     "operator_path": child.operator_path},
                )
        survived_by_path: Counter = Counter()
        retained_current = 0
        agent_metrics = []
        for agent in self.agents:
            old_current = agent.current_id
            old_pbest = agent.pbest_id
            winner = self._best_id(pool_by_agent[agent.agent_id])
            assert winner is not None  # The original evaluated current is in the pool.
            agent.current_id = winner
            if winner == old_current:
                retained_current += 1
            else:
                survived_by_path[self.archive[winner].operator_path] += 1
            if self._fitness_key(winner)[0] < self._fitness_key(agent.pbest_id)[0]:
                agent.pbest_id = winner
                agent.pbest_fitness = float(self.archive[winner].fitness)
            current_fitness = float(self.archive[agent.current_id].fitness)
            role_by_id = {
                key: item.source_role for key, item in agent.velocity.items()
            }
            usage_by_role: dict[str, Counter] = {
                role: Counter() for role in ("personal", "global", "left", "right")
            }
            for cut_item in cuts[agent.agent_id]:
                usage_by_role[cut_item.source_role]["cut_selected"] += 1
            for spec in base_specs:
                if spec.agent_id == agent.agent_id:
                    for experience_id in spec.source_cut_ids or []:
                        role = role_by_id.get(experience_id)
                        if role:
                            usage_by_role[role]["base_attempted"] += 1
            for child in base_programs:
                if child.agent_id == agent.agent_id:
                    for experience_id in child.source_cut_ids:
                        role = role_by_id.get(experience_id)
                        if role:
                            usage_by_role[role]["base_valid"] += 1
            for child in variation_programs:
                if child.agent_id == agent.agent_id:
                    for experience_id in child.source_cut_ids:
                        role = role_by_id.get(experience_id)
                        if role:
                            usage_by_role[role]["variation_valid_attributed"] += 1
            if winner != old_current:
                current_cut_ids = {
                    item.experience_id for item in cuts[agent.agent_id]
                }
                for experience_id in self.archive[winner].source_cut_ids:
                    role = role_by_id.get(experience_id)
                    if role and experience_id in current_cut_ids:
                        usage_by_role[role]["winner_attributed"] += 1
            for role, counts in usage_by_role.items():
                self._append_jsonl(
                    self.output_dir / "metrics" / "experience_source_metrics.jsonl",
                    {"generation": generation, "agent_id": agent.agent_id,
                     "source_role": role,
                     **{key: counts[key] for key in (
                         "cut_selected", "base_attempted", "base_valid",
                         "variation_valid_attributed", "winner_attributed",
                     )}},
                )
            self.recorder.emit(
                "agent_position_updated", generation=generation,
                agent_id=agent.agent_id, topic="transition",
                old_current_id=old_current, new_current_id=agent.current_id,
                old_pbest_id=old_pbest, new_pbest_id=agent.pbest_id,
                winner_id=winner, candidate_ids=pool_by_agent[agent.agent_id],
                current_retained=winner == old_current,
                fallback_used=False,
            )
            agent_metrics.append({
                "generation": generation, "agent_id": agent.agent_id,
                "guidance": guidance_by_agent[agent.agent_id],
                "h1_selection_probability": p_h1,
                "current_id": agent.current_id,
                "current_fitness": current_fitness,
                "pbest_id": agent.pbest_id,
                "pbest_fitness": agent.pbest_fitness,
                "current_equals_pbest": agent.current_id == agent.pbest_id,
                "velocity_size": len(agent.velocity),
                "new_experiences": added[agent.agent_id],
                "cut_size": len(cuts[agent.agent_id]),
                "cut_lambda": cut_info[agent.agent_id]["lambda"],
                "valid_base": sum(child.agent_id == agent.agent_id
                                  for child in base_programs),
                "valid_children": sum(child.agent_id == agent.agent_id
                                      for child in all_programs),
                "base_pool_size": base_pool_sizes[agent.agent_id],
                "final_pool_size": len(pool_by_agent[agent.agent_id]),
                "admitted_variations": admitted_by_agent[agent.agent_id],
                "current_retained": winner == old_current,
                "winner_operator": (
                    "CURRENT" if winner == old_current
                    else self.archive[winner].operator_path
                ),
                "cut_sources": dict(Counter(
                    item.source_role for item in cuts[agent.agent_id]
                )),
            })
        # No agent observes another agent's updated pbest until the next round.
        self._update_global_best(generation)
        self._save_generation(generation)
        self._save_checkpoint(generation)
        generated_by_path = Counter(
            f"{spec.guidance}->{spec.variation}"
            if spec.guidance and spec.variation else spec.source_operator
            for spec in all_specs
        )
        valid_by_path = Counter(item.operator_path for item in all_programs)
        self._write_semantic_metrics(
            generation, generated=len(all_specs), valid=len(all_programs),
            generated_by_path=dict(generated_by_path),
            valid_by_path=dict(valid_by_path),
            survived_by_path=dict(survived_by_path),
            admitted_by_path=dict(admitted_by_path),
            retained_current=retained_current,
            generation_seconds=time.perf_counter() - started,
            agent_metric_records=agent_metrics,
        )
        self.recorder.emit(
            "generation_finished", generation=generation, topic="stage",
            generated=len(all_specs), valid=len(all_programs),
            gbest_id=self.gbest_id,
            gbest_fitness=self.archive[self.gbest_id].fitness,
        )

    def _save_generation(self, generation: int) -> None:
        payload = {
            "schema_version": self.CHECKPOINT_VERSION,
            "generation": generation,
            "gbest_id": self.gbest_id,
            "gbest_fitness": (
                self.archive[self.gbest_id].fitness if self.gbest_id else None
            ),
            "agents": [
                {
                    **agent.to_dict(),
                    "current": self.archive[agent.current_id].to_dict(),
                    "pbest": self.archive[agent.pbest_id].to_dict(),
                    "position": [self.archive[agent.current_id].to_dict()],
                }
                for agent in self.agents
            ],
        }
        self._write_json(
            self.output_dir / "agents" / f"generation_{generation:04d}.json",
            payload,
        )
        self._save_best()
        self.recorder.emit(
            "generation_snapshot_saved", generation=generation,
            topic="event", agent_count=len(self.agents), gbest_id=self.gbest_id,
        )

    def _save_checkpoint(self, generation: int) -> None:
        payload = {
            "schema_version": self.CHECKPOINT_VERSION,
            "generation": generation,
            "individual_counter": self._individual_counter,
            "evaluation_count": self._evaluation_count,
            "cache_hits": self._cache_hits,
            "gbest_id": self.gbest_id,
            "agents": [agent.to_dict() for agent in self.agents],
            "archive": [item.to_dict() for item in self.archive.values()],
            "evaluation_cache": self._evaluation_cache,
            "difference_cache": self._difference_cache,
            "config": self.config.to_dict(),
        }
        self._write_json(
            self.output_dir / "checkpoints" / f"generation_{generation:04d}.json",
            payload,
        )
        self.recorder.emit(
            "checkpoint_saved", generation=generation, topic="event",
            archive_size=len(self.archive),
            evaluation_cache_size=len(self._evaluation_cache),
            velocity_size=sum(len(agent.velocity) for agent in self.agents),
        )

    def _load_checkpoint(self, path: str) -> int:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != self.CHECKPOINT_VERSION:
            raise ValueError("older checkpoints cannot be resumed by semantic MA-EoH v3")
        self._validate_resume_config(payload.get("config", {}))
        from .models import ProgramIndividual

        self.archive = {
            item["individual_id"]: ProgramIndividual.from_dict(item)
            for item in payload["archive"]
        }
        self.agents = [SemanticAgentState.from_dict(item) for item in payload["agents"]]
        self.gbest_id = payload["gbest_id"]
        self._individual_counter = int(payload["individual_counter"])
        self._evaluation_count = int(payload.get("evaluation_count", 0))
        self._cache_hits = int(payload.get("cache_hits", 0))
        self._evaluation_cache = dict(payload.get("evaluation_cache", {}))
        self._difference_cache = dict(payload.get("difference_cache", {}))
        self.recorder.emit(
            "checkpoint_loaded", generation=int(payload["generation"]),
            topic="event", checkpoint_path=path,
            archive_size=len(self.archive), agent_count=len(self.agents),
        )
        return int(payload["generation"])

    def _write_semantic_metrics(
        self,
        generation: int,
        *,
        generated: int,
        valid: int,
        generated_by_path: dict[str, int],
        valid_by_path: dict[str, int],
        survived_by_path: dict[str, int],
        admitted_by_path: dict[str, int] | None = None,
        retained_current: int = 0,
        generation_seconds: float,
        agent_metric_records: list[dict[str, Any]] | None = None,
    ) -> None:
        current_fitnesses = [
            float(self.archive[agent.current_id].fitness) for agent in self.agents
        ]
        best_fitness = self.archive[self.gbest_id].fitness if self.gbest_id else None
        failure_counts = dict(self._failure_counts.get(generation, Counter()))
        record = {
            "generation": generation,
            "gbest_id": self.gbest_id,
            "best_fitness": best_fitness,
            "population_mean": statistics.mean(current_fitnesses),
            "population_std": statistics.pstdev(current_fitnesses)
            if len(current_fitnesses) > 1 else 0.0,
            "population_min": min(current_fitnesses),
            "population_max": max(current_fitnesses),
            "generated_candidates": generated,
            "valid_candidates": valid,
            "valid_rate": valid / generated if generated else None,
            "surviving_candidates": len(self.agents),
            "surviving_children": sum(survived_by_path.values()) if generation else 0,
            "retained_current": retained_current,
            "generated_by_path": generated_by_path,
            "valid_by_path": valid_by_path,
            "survived_by_path": survived_by_path,
            "admitted_by_path": admitted_by_path or {},
            "failures_by_stage": failure_counts,
            "independent_evaluations": self._evaluation_count,
            "cache_hits": self._cache_hits,
            "archive_size": len(self.archive),
            "generation_seconds": generation_seconds,
            "velocity_entries": sum(len(agent.velocity) for agent in self.agents),
            "h1_agents": sum(
                item.get("guidance") == "H1" for item in (agent_metric_records or [])
            ),
            "h2_agents": sum(
                item.get("guidance") == "H2" for item in (agent_metric_records or [])
            ),
            "mean_cut_size": (
                statistics.mean(item["cut_size"] for item in agent_metric_records)
                if agent_metric_records else 0.0
            ),
        }
        self.recorder.record_generation_metrics(record)
        if agent_metric_records is None:
            agent_metric_records = [
                {
                    "generation": generation, "agent_id": agent.agent_id,
                    "current_id": agent.current_id,
                    "current_fitness": self.archive[agent.current_id].fitness,
                    "pbest_id": agent.pbest_id,
                    "pbest_fitness": agent.pbest_fitness,
                    "current_equals_pbest": agent.current_id == agent.pbest_id,
                    "velocity_size": len(agent.velocity),
                    "new_experiences": 0, "cut_size": 0,
                    "cut_lambda": 0.0, "valid_base": 0,
                    "valid_children": 0, "winner_operator": "H0",
                }
                for agent in self.agents
            ]
        self.recorder.record_agent_metrics(agent_metric_records)

        operator_records = []
        for path in sorted(
            set(generated_by_path) | set(valid_by_path) |
            set(survived_by_path) | set(admitted_by_path or {})
        ):
            path_generated = generated_by_path.get(path, 0)
            path_valid = valid_by_path.get(path, 0)
            path_survived = survived_by_path.get(path, 0)
            path_admitted = (admitted_by_path or {}).get(path, 0)
            operator_records.append({
                "generation": generation, "operator_path": path,
                "generated": path_generated, "valid": path_valid,
                "invalid": max(0, path_generated - path_valid),
                "valid_rate": path_valid / path_generated if path_generated else None,
                "admitted": path_admitted,
                "admission_rate_from_valid": (
                    path_admitted / path_valid if path_valid else None
                ),
                "survived": path_survived,
                "survival_rate_from_valid": (
                    path_survived / path_valid if path_valid else None
                ),
                "best_fitness": (
                    min if self.config.objective_direction == "min" else max
                )(
                    (item.fitness for item in self.archive.values()
                     if item.generation == generation and item.operator_path == path
                     and item.fitness is not None),
                    default=None,
                ),
                "recorded_failures": self._path_failure_counts.get(
                    generation, Counter()
                ).get(path, 0),
            })
        self.recorder.record_operator_metrics(operator_records)
        self.recorder.emit(
            "generation_metrics_recorded", generation=generation,
            topic="event", metrics=record,
        )
        self.logger.info(
            "--- Generation %d/%d best=%s valid=%d/%d evals=%d cache=%d",
            generation, self.config.generations, best_fitness,
            valid, generated, self._evaluation_count, self._cache_hits,
        )
        for agent in self.agents:
            self.logger.info(
                "  agent=%02d current=%s fitness=%s pbest=%s pbest_fitness=%s memory=%d",
                agent.agent_id, agent.current_id,
                self.archive[agent.current_id].fitness,
                agent.pbest_id, agent.pbest_fitness, len(agent.velocity),
            )
