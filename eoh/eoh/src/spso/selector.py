"""LLM-backed semantic module selection with strict JSON validation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .models import ModuleChoice, ParticlePosition
from .registry import ModuleRegistry

logger = logging.getLogger("spso")


@dataclass(frozen=True)
class SelectionResult:
    position: ParticlePosition
    strategy: str = "B"
    source: str = "fallback"
    raw_response: str | None = None


class SemanticSelector:
    """Select one validated module per slot from cut sets.

    The LLM is only a semantic selector.  It cannot provide executable code or
    introduce a module outside the registry.
    """

    def __init__(
        self,
        registry: ModuleRegistry,
        llm=None,
        max_retries: int = 2,
        module_code_provider: Callable[[str, str, Mapping[str, Any] | None], str] | None = None,
        position_code_provider: Callable[[ParticlePosition], str] | None = None,
    ):
        self.registry = registry
        self.llm = llm
        self.max_retries = max(0, int(max_retries))
        self.module_code_provider = module_code_provider
        self.position_code_provider = position_code_provider

    def _cut_payload(self, cut_sets: Mapping[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
        """Build the code-bearing candidate context sent to the LLM."""
        payload: dict[str, list[dict[str, Any]]] = {}
        for slot_id, module_ids in cut_sets.items():
            entries = []
            for module_id in module_ids:
                spec = self.registry.module(slot_id, module_id)
                parameters = {
                    parameter.name: {
                        "kind": parameter.kind,
                        "default": parameter.default,
                        "choices": list(parameter.choices),
                        "minimum": parameter.minimum,
                        "maximum": parameter.maximum,
                    }
                    for parameter in spec.parameters
                }
                params = spec.default_params()
                entry = {
                    "module": module_id,
                    "description": spec.description,
                    "parameters": parameters,
                    "default_params": params,
                }
                if self.module_code_provider is not None:
                    entry["code"] = self.module_code_provider(slot_id, module_id, params)
                entries.append(entry)
            payload[slot_id] = entries
        return payload

    def _program_context(self, name: str, position: ParticlePosition) -> str:
        if self.position_code_provider is None:
            return f"{name}: code provider unavailable; use the module snippets above."
        try:
            return f"{name}:\n```python\n{self.position_code_provider(position)}\n```"
        except Exception as exc:
            logger.debug("could not compile %s reference: %s", name, exc)
            return f"{name}: code compilation unavailable; use the module snippets above."

    def _prompt(
        self,
        cut_sets: Mapping[str, list[str]],
        current: ParticlePosition,
        pbest: ParticlePosition,
        gbest: ParticlePosition,
        objectives: Mapping[str, float | None],
    ) -> str:
        cut_payload = self._cut_payload(cut_sets)
        return (
            "You are selecting modules for a typed heuristic program.\n"
            "Return JSON only, with this schema: "
            '{"sequence":[{"slot":"...","module":"...","params":{}}],"strategy":"A|B|C"}.\n'
            "Choose exactly one module from each slot's candidates. Parameters must be valid. "
            "A reverses one semantic direction, B preserves the selected structure, C performs a local semantic change.\n"
            "The public function contract is exactly "
            "update_edge_distance(edge_distance, local_opt_tour, edge_n_used) -> matrix.\n"
            "The candidate code below is reference code for internal compiler snippets. "
            "Do not write new executable code and do not change the public signature.\n"
            f"Cut candidates with descriptions, parameters, and code:\n{json.dumps(cut_payload, ensure_ascii=False, indent=2)}\n"
            f"Current: {json.dumps(current.to_dict(), ensure_ascii=False)}\n"
            f"pbest: {json.dumps(pbest.to_dict(), ensure_ascii=False)}\n"
            f"gbest: {json.dumps(gbest.to_dict(), ensure_ascii=False)}\n"
            f"Objectives: {json.dumps(dict(objectives), ensure_ascii=False)}\n"
            f"{self._program_context('Current program', current)}\n"
            f"{self._program_context('pbest program', pbest)}\n"
            f"{self._program_context('gbest program', gbest)}\n"
        )

    @staticmethod
    def _extract_json(response: str | None) -> dict[str, Any] | None:
        if not response:
            return None
        text = response.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
        candidate = fenced.group(1) if fenced else text
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return None
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None

    def _validate_response(
        self,
        data: Mapping[str, Any],
        cut_sets: Mapping[str, list[str]],
    ) -> SelectionResult:
        raw_sequence = data.get("sequence")
        if not isinstance(raw_sequence, list):
            raise ValueError("response.sequence must be a list")
        parsed = {}
        for item in raw_sequence:
            if not isinstance(item, Mapping):
                raise ValueError("each sequence item must be an object")
            slot = str(item.get("slot"))
            module = str(item.get("module"))
            if slot not in cut_sets or module not in cut_sets[slot]:
                raise ValueError(f"LLM selected module outside cut set: {slot}/{module}")
            parsed[slot] = self.registry.validate_choice(
                ModuleChoice.create(slot, module, item.get("params", {}))
            )
        if set(parsed) != set(self.registry.slot_ids):
            raise ValueError("response must select exactly one module per slot")
        strategy = str(data.get("strategy", "B")).upper()
        if strategy not in {"A", "B", "C"}:
            raise ValueError("strategy must be A, B, or C")
        return SelectionResult(
            position=ParticlePosition(tuple(parsed[slot] for slot in self.registry.slot_ids)),
            strategy=strategy,
            source="llm",
        )

    def _fallback(self, cut_sets: Mapping[str, list[str]], velocity) -> SelectionResult:
        choices = []
        for slot in self.registry.slot_ids:
            module = cut_sets[slot][0]
            choices.append(ModuleChoice.create(slot, module))
        return SelectionResult(ParticlePosition(tuple(choices)), "B", "fallback")

    def select(self, cut_sets, current, pbest, gbest, objectives, velocity) -> SelectionResult:
        if self.llm is None:
            return self._fallback(cut_sets, velocity)
        prompt = self._prompt(cut_sets, current, pbest, gbest, objectives)
        last_response = None
        for attempt in range(self.max_retries + 1):
            try:
                last_response = self.llm.get_response(prompt)
                data = self._extract_json(last_response)
                if data is not None:
                    result = self._validate_response(data, cut_sets)
                    return SelectionResult(result.position, result.strategy, result.source, last_response)
            except Exception as exc:
                logger.debug("semantic selection attempt %d failed: %s", attempt + 1, exc)
        fallback = self._fallback(cut_sets, velocity)
        return SelectionResult(fallback.position, fallback.strategy, "fallback", last_response)
