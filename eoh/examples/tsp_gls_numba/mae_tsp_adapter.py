from __future__ import annotations

import hashlib
import re
import threading
import time

import numpy as np

from eoh.mae.generator import ProgramDraft
from eoh.mae.models import CutItem, ExperienceRecord, ProgramIndividual
from eoh.mae.validation import (
    ProgramValidator,
    make_tsp_smoke_test,
    structural_fingerprint,
)
from prob import TSPGLS, solve_instance
from heuristic_runtime import HeuristicExecutionError


class TracedTSPGLS(TSPGLS):
    """TSPGLS variant returning per-instance details to the MA-EoH logger."""

    def evaluate_program(self, program_str, callable_func):
        try:
            callable_func, runtime = self._prepare_heuristic(callable_func)
            return self._evaluate_prepared(callable_func, runtime)
        except HeuristicExecutionError as exc:
            return {"fitness": None, "details": {"failure": exc.details}}

    def _evaluate_prepared(self, callable_func, runtime):
        instances = []
        gaps = []
        evaluation_started = time.perf_counter()
        for index in range(self.n_inst_eva):
            started = time.perf_counter()
            gap, iterations = solve_instance(
                self.opt_costs[index],
                self.instances[index],
                self.time_limit,
                self.ite_max,
                self.perturbation_moves,
                callable_func,
            )
            gap = float(gap)
            opt_cost = float(self.opt_costs[index])
            estimated_length = opt_cost * (1.0 + gap / 100.0)
            gaps.append(gap)
            instances.append({
                "instance_index": index,
                "node_count": int(len(self.instances[index])),
                "reference_length": opt_cost,
                "estimated_length": estimated_length,
                "gap_percent": gap,
                "iterations": int(iterations),
                "iteration_limit": int(self.ite_max),
                "stopped_by": "iteration_limit" if iterations >= self.ite_max else "time_limit",
                "elapsed_seconds": time.perf_counter() - started,
                "valid": bool(np.isfinite(gap) and gap < 1e9),
            })
        return {
            "fitness": float(np.mean(gaps)),
            "details": {
                "heuristic_runtime": runtime,
                "instances": instances,
                "instance_count": self.n_inst_eva,
                "time_limit": self.time_limit,
                "iteration_limit": self.ite_max,
                "perturbation_moves": self.perturbation_moves,
                "total_elapsed_seconds": time.perf_counter() - evaluation_started,
            },
        }


def build_tsp_validator(
    problem, *, require_symmetric=True, require_nonnegative=True,
    allow_edge_n_used_update=True,
):
    return ProgramValidator(
        problem.template_program,
        smoke_test=make_tsp_smoke_test(
            require_symmetric=require_symmetric,
            require_nonnegative=require_nonnegative,
            allow_edge_n_used_update=allow_edge_n_used_update,
        ),
    )


class OfflineTSPGenerator:
    """Deterministic generator for smoke tests; it never calls an external LLM."""

    def __init__(self, seed: int = 2024) -> None:
        self.seed = seed
        self._counter = 0
        self._lock = threading.Lock()

    def _next(self, *parts: object) -> tuple[int, float]:
        with self._lock:
            self._counter += 1
            index = self._counter
        key = "|".join(str(item) for item in (self.seed, index, *parts))
        number = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
        scale = 0.25 + (number % 176) / 100.0
        return index, scale

    @staticmethod
    def _code(scale: float, mode: int) -> str:
        if mode % 3 == 0:
            delta = f"edge_distance[u, v] / (1.0 + {scale:.4f} * edge_n_used[u, v])"
        elif mode % 3 == 1:
            delta = f"{scale:.4f} * edge_distance[u, v] / (1.0 + edge_n_used[u, v])"
        else:
            delta = (
                f"edge_distance[u, v] * (1.0 + {scale:.4f} / "
                "(2.0 + edge_n_used[u, v]))"
            )
        return f'''def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    augmented = edge_distance.copy()
    n = len(local_opt_tour)
    for index in range(n):
        u = int(local_opt_tour[index])
        v = int(local_opt_tour[(index + 1) % n])
        delta = {delta}
        augmented[u, v] += delta
        augmented[v, u] += delta
    return augmented
'''

    def generate_h0(self, agent_id: int, sample_index: int) -> ProgramDraft:
        index, scale = self._next("H0", agent_id, sample_index)
        return ProgramDraft(
            description=(
                f"Offline H0 candidate {index} penalizes current-tour edges with scale {scale:.4f}."
            ),
            code=self._code(scale, index),
            prompt_hash=f"offline-h0-{index}",
        )

    def describe_difference(
        self,
        agent_id: int,
        source: ProgramIndividual,
        target: ProgramIndividual,
        direction_type: str = "improve",
    ) -> str:
        return (
            f"Adapt the edge-penalty strength and update pattern of {source.individual_id} "
            f"toward a {direction_type} alternative from {target.individual_id}."
        )

    def apply_experience(
        self,
        agent_id: int,
        current: ProgramIndividual,
        experience: ExperienceRecord,
        exemplar: ProgramIndividual,
        guidance: str,
    ) -> ProgramDraft:
        index, scale = self._next(
            guidance, agent_id, current.code_hash, experience.experience_id
        )
        return ProgramDraft(
            description=(
                f"Offline {guidance} difference candidate {index} uses {experience.source_role} "
                f"experience with penalty scale {scale:.4f}."
            ),
            code=self._code(scale, index),
            prompt_hash=f"offline-{guidance}-diff-{index}",
        )

    def select_cut(
        self,
        agent_id: int,
        guidance: str,
        cut: list[CutItem],
        individuals: dict[str, ProgramIndividual],
        selection_size: int,
    ) -> list[str]:
        ranked = sorted(
            cut,
            key=lambda item: (
                individuals[item.individual_id].fitness
                if individuals[item.individual_id].fitness is not None
                else float("inf"),
                -item.probability,
                item.individual_id,
            ),
        )
        return [item.individual_id for item in ranked[:selection_size]]

    def generate_variation(
        self,
        agent_id: int,
        guidance: str,
        variation: str,
        parents: list[ProgramIndividual],
        selected: list[ProgramIndividual],
    ) -> ProgramDraft:
        index, scale = self._next(guidance, variation, agent_id, *(p.code_hash for p in parents))
        if variation == "H5":
            code = re.sub(r"\d+\.\d{4}", f"{scale:.4f}", parents[0].code, count=1)
        elif variation in {"H4", "H6"}:
            parent_structures = {structural_fingerprint(parent.code) for parent in parents}
            code = None
            for mode in range(3):
                proposal = self._code(scale, mode)
                if structural_fingerprint(proposal) not in parent_structures:
                    code = proposal
                    break
            if code is None:
                code = self._code(scale, index)
        else:
            code = self._code(scale, index)
        return ProgramDraft(
            description=(
                f"Offline {guidance}->{variation} candidate {index} applies scale {scale:.4f}."
            ),
            code=code,
            prompt_hash=f"offline-{guidance}-{variation}-{index}",
        )
