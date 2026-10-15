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
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from .legacy_engine import MAEEngine as LegacyMAEEngine, _CandidateSpec
from .experience import (
    birth_probability,
    decay_and_prune,
    prune_velocity,
    sample_cut,
)
from .generator import ProgramDraft
from .models import ExperienceCutItem, ExperienceRecord, SemanticAgentState
from .topology import ring_neighbors


class MAEEngine(LegacyMAEEngine):
    """Synchronous agents, one current program each, persistent experience."""

    CHECKPOINT_VERSION = 2
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
            raise ValueError("v1 checkpoints cannot be resumed by semantic MA-EoH v2")
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
            specs = self._submit_h0_round(requests, round_index)
            generated += len(specs)
            for item in self._evaluate_specs(specs, seen):
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
        self, generation: int
    ) -> tuple[dict[int, list[ExperienceCutItem]], dict[int, dict[str, Any]]]:
        cuts = {}
        diagnostics = {}
        for agent in self.agents:
            cut, info = sample_cut(
                agent.velocity,
                target_size=self.config.cut_target_size,
                neighbor_reserved_slots=self.config.neighbor_reserved_slots,
                generation=generation,
                agent_id=agent.agent_id,
                enabled_operators=self.config.enabled_operators,
                h1_boost=self.config.h1_selection_boost,
                h2_boost=self.config.h2_selection_boost,
                rng=random.Random(self._seed_for(generation, agent.agent_id, "cut")),
            )
            cuts[agent.agent_id] = cut
            diagnostics[agent.agent_id] = info
            payload = {"generation": generation, "agent_id": agent.agent_id,
                       "cut": [item.to_dict() for item in cut], **info}
            self._write_json(
                self.output_dir / "cuts" / f"generation_{generation:04d}"
                / f"agent_{agent.agent_id:03d}.json", payload,
            )
            self.recorder.emit(
                "cut_created", generation=generation, agent_id=agent.agent_id,
                topic="cut", cut=payload["cut"], lambda_value=info["lambda"],
                eligible=info["eligible"], target=info["target"],
                reserved=info["reserved"],
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

    def _generate_base(
        self, generation: int, cuts: dict[int, list[ExperienceCutItem]]
    ) -> list[_CandidateSpec]:
        assert self._llm_pool is not None
        futures: dict[Future, _CandidateSpec] = {}
        active_guidance = "+".join(
            operator for operator in self.config.enabled_operators
            if operator in {"H1", "H2"}
        ) or "none"
        for agent in self.agents:
            current = self.archive[agent.current_id]
            for cut_item in cuts[agent.agent_id]:
                experience = agent.velocity[cut_item.experience_id]
                exemplar = self.archive[experience.target_id]
                future = self._llm_pool.submit(
                    self.generator.apply_experience,
                    agent.agent_id, current, experience, exemplar,
                )
                futures[future] = _CandidateSpec(
                    agent_id=agent.agent_id, generation=generation,
                    draft=ProgramDraft(None, None), source_operator="DIFF",
                    guidance=active_guidance, parent_ids=[agent.current_id],
                    source_cut_ids=[experience.experience_id],
                )
            # Fewer than the configured minimum *real* experiences triggers
            # separate H0 exploration; it does not fabricate cut entries.
            for fallback_index in range(
                max(0, self.config.cut_min_size - len(cuts[agent.agent_id]))
            ):
                future = self._llm_pool.submit(
                    self.generator.generate_h0,
                    agent.agent_id,
                    generation * 100000 + agent.agent_id * 100 + fallback_index,
                )
                futures[future] = _CandidateSpec(
                    agent_id=agent.agent_id, generation=generation,
                    draft=ProgramDraft(None, None), source_operator="H0",
                )
        return self._finish_draft_futures(
            futures, phase="base_generation", generation=generation
        )

    def _generate_variations(
        self,
        generation: int,
        cuts: dict[int, list[ExperienceCutItem]],
        base_by_agent: dict[int, list[str]],
    ) -> list[_CandidateSpec]:
        assert self._llm_pool is not None
        futures: dict[Future, _CandidateSpec] = {}
        operators = [
            item for item in self.config.enabled_operators
            if item not in {"H1", "H2"}
        ]
        parent_counts = {"H3": 2, "H4": 2, "H5": 1}
        parent_counts.update(self.config.variation_parent_counts or {})
        guidance = "+".join(
            item for item in self.config.enabled_operators if item in {"H1", "H2"}
        ) or "none"

        for agent in self.agents:
            base_ids = base_by_agent[agent.agent_id]
            if not base_ids:
                continue
            base_programs = [self.archive[item_id] for item_id in base_ids]
            # Keep the requested N attempts per enabled variation even when
            # some first-stage drafts were invalid; parent reuse is explicit.
            n = len(cuts[agent.agent_id])
            for variation in operators:
                arity = parent_counts.get(variation)
                if arity is None:
                    raise ValueError(
                        f"operator {variation} needs variation_parent_counts entry"
                    )
                for sample_index in range(n):
                    rng = random.Random(
                        self._seed_for(generation, agent.agent_id, variation, sample_index)
                    )
                    pool = list(base_ids)
                    if len(pool) < arity:
                        for candidate_id in (agent.current_id, agent.pbest_id):
                            if self.archive[candidate_id].code_hash not in {
                                self.archive[item_id].code_hash for item_id in pool
                            }:
                                pool.append(candidate_id)
                    parent_ids = (
                        rng.sample(pool, arity)
                        if len(pool) >= arity
                        else [rng.choice(pool) for _ in range(arity)]
                    )
                    parents = [self.archive[item_id] for item_id in parent_ids]
                    current_cut_ids = {
                        item.experience_id for item in cuts[agent.agent_id]
                    }
                    source_ids = sorted({
                        experience_id for parent in parents
                        for experience_id in parent.source_cut_ids
                        if experience_id in current_cut_ids
                    })
                    future = self._llm_pool.submit(
                        self.generator.generate_variation,
                        agent.agent_id, guidance, variation, parents, base_programs,
                    )
                    futures[future] = _CandidateSpec(
                        agent_id=agent.agent_id, generation=generation,
                        draft=ProgramDraft(None, None), source_operator=variation,
                        guidance=guidance, variation=variation,
                        parent_ids=parent_ids, source_cut_ids=source_ids,
                    )
        return self._finish_draft_futures(
            futures, phase="variation_generation", generation=generation
        )

    def _evolve_generation(self, generation: int) -> None:
        started = time.perf_counter()
        snapshot = self._frozen_references()
        old_states = {agent.agent_id: agent.to_dict() for agent in self.agents}
        self.recorder.emit(
            "generation_started", generation=generation, topic="stage",
            gbest_id=self.gbest_id, agent_snapshots=list(old_states.values()),
        )
        with self.recorder.stage("experience_update", generation=generation):
            added = self._refresh_velocity(generation, snapshot)
        with self.recorder.stage("bounded_cut", generation=generation):
            cuts, cut_info = self._draw_cuts(generation)
        seen_by_agent = {
            agent.agent_id: {self.archive[agent.current_id].code_hash}
            for agent in self.agents
        }
        with self.recorder.stage("difference_application", generation=generation):
            base_specs = self._generate_base(generation, cuts)
            base_programs = self._evaluate_specs(base_specs, seen_by_agent)
        base_by_agent: dict[int, list[str]] = {
            agent.agent_id: [] for agent in self.agents
        }
        for child in base_programs:
            base_by_agent[child.agent_id].append(child.individual_id)
        with self.recorder.stage("variation_generation", generation=generation):
            variation_specs = self._generate_variations(
                generation, cuts, base_by_agent
            )
            variation_programs = self._evaluate_specs(
                variation_specs, seen_by_agent
            )

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
        by_agent: dict[int, list[str]] = {
            agent.agent_id: [] for agent in self.agents
        }
        for child in all_programs:
            by_agent[child.agent_id].append(child.individual_id)
        survived_by_path: Counter = Counter()
        agent_metrics = []
        for agent in self.agents:
            old_current = agent.current_id
            old_pbest = agent.pbest_id
            winner = self._best_id(by_agent[agent.agent_id])
            if winner is not None:
                agent.current_id = winner
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
                        usage_by_role[role_by_id[experience_id]]["base_attempted"] += 1
            for child in base_programs:
                if child.agent_id == agent.agent_id:
                    for experience_id in child.source_cut_ids:
                        usage_by_role[role_by_id[experience_id]]["base_valid"] += 1
            for child in variation_programs:
                if child.agent_id == agent.agent_id:
                    for experience_id in child.source_cut_ids:
                        usage_by_role[role_by_id[experience_id]]["variation_valid_attributed"] += 1
            if winner:
                for experience_id in self.archive[winner].source_cut_ids:
                    usage_by_role[role_by_id[experience_id]]["winner_attributed"] += 1
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
                winner_id=winner, candidate_ids=by_agent[agent.agent_id],
                fallback_used=winner is None,
            )
            agent_metrics.append({
                "generation": generation, "agent_id": agent.agent_id,
                "current_id": agent.current_id,
                "current_fitness": current_fitness,
                "pbest_id": agent.pbest_id,
                "pbest_fitness": agent.pbest_fitness,
                "current_equals_pbest": agent.current_id == agent.pbest_id,
                "velocity_size": len(agent.velocity),
                "new_experiences": added[agent.agent_id],
                "cut_size": len(cuts[agent.agent_id]),
                "cut_lambda": cut_info[agent.agent_id]["lambda"],
                "valid_base": len(base_by_agent[agent.agent_id]),
                "valid_children": len(by_agent[agent.agent_id]),
                "winner_operator": (
                    self.archive[winner].operator_path if winner else None
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
            raise ValueError("v1 checkpoints cannot be resumed by semantic MA-EoH v2")
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
            "surviving_candidates": sum(survived_by_path.values()),
            "surviving_children": sum(survived_by_path.values()) if generation else 0,
            "generated_by_path": generated_by_path,
            "valid_by_path": valid_by_path,
            "survived_by_path": survived_by_path,
            "failures_by_stage": failure_counts,
            "independent_evaluations": self._evaluation_count,
            "cache_hits": self._cache_hits,
            "archive_size": len(self.archive),
            "generation_seconds": generation_seconds,
            "velocity_entries": sum(len(agent.velocity) for agent in self.agents),
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
            set(generated_by_path) | set(valid_by_path) | set(survived_by_path)
        ):
            path_generated = generated_by_path.get(path, 0)
            path_valid = valid_by_path.get(path, 0)
            path_survived = survived_by_path.get(path, 0)
            operator_records.append({
                "generation": generation, "operator_path": path,
                "generated": path_generated, "valid": path_valid,
                "invalid": max(0, path_generated - path_valid),
                "valid_rate": path_valid / path_generated if path_generated else None,
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
