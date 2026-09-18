"""LLM module-library construction and EoH-style p0--p4 operations.

The adapter owns the typed proposal protocol.  The LLM can propose a sentence
and a recipe, but it cannot register or execute arbitrary Python.  Registration
and compilation happen later in deterministic coordinator order.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from spso.models import ModuleChoice, ParticlePosition, SetPosition
from spso.registry import ModuleRegistry, ModuleSpec, ParameterSpec


def _json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    candidate = match.group(1) if match else text
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


@dataclass
class LLMCallStats:
    calls: int = 0
    retries: int = 0
    failures: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "retries": self.retries,
            "failures": self.failures,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class ModuleProposal:
    slot: str
    description: str
    recipe: dict[str, Any]
    parameters: tuple[ParameterSpec, ...]
    source_operator: str
    relation: str | None = None
    reference_module: str | None = None
    module_id: str | None = None
    prompt_ids: list[str] = field(default_factory=list)
    description_response: str | None = None
    implementation_response: str | None = None


@dataclass
class ChildProposal:
    operator: str
    position: ParticlePosition | None
    set_position: SetPosition | None
    new_modules: list[ModuleProposal] = field(default_factory=list)
    base_position: ParticlePosition | None = None
    prompt_ids: list[str] = field(default_factory=list)
    failure_reason: str | None = None


class LLMProtocol:
    """Audit wrapper around the EoH ``get_response(prompt)`` interface."""

    def __init__(self, llm=None, *, output_dir=None):
        self.llm = llm
        self.output_dir = Path(output_dir) if output_dir else None
        self.stats: dict[str, LLMCallStats] = {}
        self.calls: list[dict[str, Any]] = []
        self._counter = 0
        self._lock = threading.RLock()

    def call(self, prompt: str, *, generation: int = 0, operator: str = "unknown") -> tuple[str, str]:
        if self.llm is None:
            raise RuntimeError("LLM is required for online module evolution")
        key = f"generation_{generation}/{operator}"
        with self._lock:
            stats = self.stats.setdefault(key, LLMCallStats())
            stats.calls += 1
            self._counter += 1
            call_id = f"llm-{self._counter:06d}"
        started = time.perf_counter()
        response = None
        error = None
        try:
            response = self.llm.get_response(prompt)
            if not response:
                raise ValueError("empty LLM response")
            return str(response), call_id
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                stats.failures += 1
            raise
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                stats.elapsed_seconds += elapsed
                self.calls.append({
                "call_id": call_id,
                "generation": generation,
                "operator": operator,
                "elapsed_seconds": elapsed,
                "error": error,
                "prompt": prompt,
                "response": response,
                })

    def retry(self, prompt: str, validator, *, generation: int, operator: str, max_retries: int):
        last_error = None
        for attempt in range(max_retries + 1):
            if attempt:
                with self._lock:
                    self.stats.setdefault(f"generation_{generation}/{operator}", LLMCallStats()).retries += 1
            try:
                response, call_id = self.call(prompt, generation=generation, operator=operator)
                value = validator(response)
                if value is not None:
                    return value, response, call_id
                last_error = ValueError("response failed schema validation")
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"LLM protocol failed for {operator}: {last_error}")

    def flush(self):
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "llm_calls.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in self.calls),
            encoding="utf-8",
        )
        (self.output_dir / "llm_metrics.json").write_text(
            json.dumps({key: value.to_dict() for key, value in self.stats.items()}, indent=2),
            encoding="utf-8",
        )


class TSPModuleLibraryBuilder:
    """Build exactly ``modules_per_slot`` accepted modules per slot."""

    def __init__(self, prototype: ModuleRegistry, llm, *, modules_per_slot=10,
                 max_retries=3, output_dir=None, use_descriptions=True):
        self.prototype = prototype
        self.llm = LLMProtocol(llm, output_dir=output_dir)
        self.modules_per_slot = int(modules_per_slot)
        self.max_retries = int(max_retries)
        self.output_dir = Path(output_dir) if output_dir else None
        self.use_descriptions = bool(use_descriptions)
        self.rejected: list[dict[str, Any]] = []

    def _library_prompt(self, slot_id: str) -> str:
        slot = self.prototype.slot(slot_id)
        examples = []
        for module in slot.modules:
            item = {"module": module.module_id, "recipe": dict(module.recipe), "parameters": [p.to_dict() for p in module.parameters]}
            if self.use_descriptions:
                item["description"] = module.description
            examples.append(item)
        return json.dumps({"task": "TSP_GLS_TYPED_MODULE", "slot": slot_id,
                           "slot_description": slot.description,
                           "allowed_examples": examples,
                           "required": "one sentence description followed by a typed implementation"},
                          ensure_ascii=False, indent=2)

    def _description(self, slot_id: str, index: int, generation=0, accepted_signatures=()):
        novelty = ""
        if accepted_signatures:
            novelty = (
                "\nAlready accepted normalized implementations for this slot:\n"
                f"{json.dumps(list(accepted_signatures), ensure_ascii=False, indent=2)}\n"
                "Design a different module by changing the recipe kind when useful, "
                "or by changing bounded parameter values.\n"
            )
        prompt = (
            "MODULE_DESCRIPTION phase. Return JSON only: "
            '{"description":"one concise sentence"}.\n'
            f"Create a new executable TSP GLS module for slot {slot_id}, candidate {index}.\n"
            f"{self._library_prompt(slot_id)}{novelty}"
        )
        def validate(response):
            data = _json_object(response)
            text = data.get("description") if data else None
            if not isinstance(text, str) or not text.strip() or len(text.strip()) > 240:
                return None
            return text.strip()
        return self.llm.retry(prompt, validate, generation=generation, operator="library_description", max_retries=self.max_retries)

    def _allowed_kinds(self, slot_id: str) -> dict[str, ModuleSpec]:
        return {str(module.recipe.get("kind")): module for module in self.prototype.slot(slot_id).modules}

    @staticmethod
    def _fixed_values(template: ModuleSpec, supplied: Mapping[str, Any]) -> dict[str, Any]:
        """Validate values against adapter-owned schema, ignoring no fields."""
        supplied = dict(supplied)
        required = {parameter.name for parameter in template.parameters}
        if set(supplied) - required:
            raise ValueError("implementation contains parameters not used by the recipe")
        values = template.default_params()
        values.update(supplied)
        return {
            parameter.name: parameter.validate(values.get(parameter.name, parameter.default))
            for parameter in template.parameters
        }

    def _implementation(self, slot_id: str, index: int, description: str, generation=0, accepted_signatures=()):
        novelty = ""
        if accepted_signatures:
            novelty = (
                "\nAvoid these already accepted normalized implementations:\n"
                f"{json.dumps(list(accepted_signatures), ensure_ascii=False, indent=2)}\n"
                "Return a recipe whose kind and parameter values produce a new implementation.\n"
            )
        prompt = (
            "MODULE_IMPLEMENTATION phase. The previous response is the design sentence. "
            "Return JSON only with recipe={kind,family,polarity,parameters} and optional "
            "parameter_schema. Use only the typed recipe kinds in the examples; do not emit Python.\n"
            f"description={description}\n{self._library_prompt(slot_id)}{novelty}"
        )
        allowed = self._allowed_kinds(slot_id)
        def validate(response):
            data = _json_object(response)
            recipe = data.get("recipe") if data else None
            if not isinstance(recipe, Mapping) or str(recipe.get("kind")) not in allowed:
                return None
            base = allowed[str(recipe["kind"])]
            params = dict(recipe.get("parameters", {}))
            try:
                # The adapter owns the required names, types and bounds. The
                # LLM can only supply values inside this fixed schema.
                values = self._fixed_values(base, params)
            except Exception:
                return None
            specs = tuple(replace(parameter, default=values[parameter.name]) for parameter in base.parameters)
            clean_recipe = {
                "kind": str(recipe["kind"]),
                # Semantic relations are adapter metadata, never LLM claims.
                "family": base.semantic_family,
                "polarity": base.polarity,
            }
            return clean_recipe, specs, values
        return self.llm.retry(prompt, validate, generation=generation, operator="library_implementation", max_retries=self.max_retries)

    def build(self) -> ModuleRegistry:
        if self.modules_per_slot < 1:
            raise ValueError("modules_per_slot must be positive")
        slots = []
        for slot in self.prototype.slots:
            accepted: list[ModuleSpec] = []
            signatures = set()
            for index in range(1, self.modules_per_slot + 1):
                accepted_one = False
                # Building ten unique modules per slot is harder than a normal
                # schema retry: valid LLM answers can still be duplicates.
                # Give uniqueness retries a larger budget without relaxing the
                # adapter-owned recipe schema.
                for attempt in range(max(self.max_retries + 1, self.modules_per_slot * 2)):
                    try:
                        description, desc_response, desc_id = self._description(
                            slot.slot_id, index, accepted_signatures=sorted(signatures)
                        )
                        (recipe, parameters, values), impl_response, impl_id = self._implementation(
                            slot.slot_id, index, description, accepted_signatures=sorted(signatures)
                        )
                        signature = json.dumps({
                            "slot": slot.slot_id,
                            "kind": recipe["kind"],
                            "parameters": values,
                        }, sort_keys=True, default=str)
                        if signature in signatures:
                            raise ValueError("duplicate normalized implementation")
                        signatures.add(signature)
                        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
                        module = ModuleSpec(
                            module_id=f"llm_{slot.slot_id}_{digest}",
                            description=description,
                            parameters=parameters,
                            tags=("llm_generated",),
                            recipe=recipe,
                            generation=0,
                            source_operator="p0",
                            semantic_family=recipe["family"],
                            polarity=recipe["polarity"],
                        )
                        accepted.append(module)
                        accepted_one = True
                        break
                    except Exception as exc:
                        self.rejected.append({"slot": slot.slot_id, "index": index, "attempt": attempt + 1, "reason": str(exc)})
                if not accepted_one:
                    raise RuntimeError(f"could not obtain {self.modules_per_slot} valid modules for slot {slot.slot_id}")
            slots.append(type(slot)(slot.slot_id, slot.description, tuple(accepted), slot.min_cut, slot.max_cut, slot.capacity))
        registry = ModuleRegistry(tuple(slots))
        # A frozen initial library must support the requested semantic
        # operators. Otherwise a later p1/p2 failure would be caused by the
        # library design rather than by an LLM proposal.
        inverse = {"decay": "boost", "boost": "decay", "long": "short", "short": "long", "compress": "amplify", "amplify": "compress", "dense": "sparse", "sparse": "dense"}
        same_family = any(
            len([module for module in slot.modules if module.semantic_family == family]) >= 2
            for slot in registry.slots
            for family in {module.semantic_family for module in slot.modules}
        )
        inverse_pair = any(
            any(other.polarity == inverse.get(module.polarity) for other in slot.modules if other.module_id != module.module_id)
            for slot in registry.slots for module in slot.modules
        )
        if not same_family:
            raise RuntimeError("generated library has no same-family alternative for p1")
        if not inverse_pair:
            raise RuntimeError("generated library has no trusted inverse polarity pair for p2")
        self._save(registry)
        return registry

    def _save(self, registry: ModuleRegistry):
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "module_library.json").write_text(json.dumps(registry.library_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "rejected_proposals.json").write_text(json.dumps(self.rejected, ensure_ascii=False, indent=2), encoding="utf-8")
        self.llm.flush()


class TSPModuleEvolution:
    """Generate legal p0--p4 proposals against a frozen library snapshot."""

    def __init__(self, registry: ModuleRegistry, compiler, llm=None, *, use_descriptions=True,
                 max_retries=2, output_dir=None, reuse_existing_modules=False,
                 reuse_existing_probability=0.0):
        self.registry = registry
        self.compiler = compiler
        self.llm = LLMProtocol(llm, output_dir=output_dir)
        self.use_descriptions = bool(use_descriptions)
        self.max_retries = int(max_retries)
        self.reuse_existing_modules = bool(reuse_existing_modules)
        self.reuse_existing_probability = float(reuse_existing_probability)
        if not 0.0 <= self.reuse_existing_probability <= 1.0:
            raise ValueError("reuse_existing_probability must be in [0, 1]")
        self.reuse_stats = {
            "eligible": 0,
            "attempted": 0,
            "llm_selected": 0,
            "deterministic_fallback": 0,
            "new_module": 0,
        }
        self._reuse_stats_lock = threading.Lock()

    def _module_payload(self, slot_id: str, ids: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
        output = []
        for module_id in ids:
            spec = self.registry.module(slot_id, module_id)
            item = {"module": module_id, "parameters": [p.to_dict() for p in spec.parameters], "default_params": spec.default_params(), "code": self.compiler.module_code(slot_id, module_id, spec.default_params())}
            if self.use_descriptions:
                item["description"] = spec.description
            output.append(item)
        return output

    def _cut_prompt(self, cut_sets: Mapping[str, list[str]]) -> str:
        return json.dumps({slot: self._module_payload(slot, ids) for slot, ids in cut_sets.items()}, ensure_ascii=False, indent=2)

    @staticmethod
    def _opposite_polarity(value: str) -> str | None:
        return {
            "decay": "boost", "boost": "decay", "long": "short", "short": "long",
            "compress": "amplify", "amplify": "compress", "dense": "sparse", "sparse": "dense",
        }.get(value)

    def _candidate_slots(self, operator: str, base: ParticlePosition) -> list[str]:
        candidates = []
        for slot_id in self.registry.slot_ids:
            reference = self.registry.module(slot_id, base.as_mapping()[slot_id].module)
            modules = list(self.registry.slot(slot_id).modules)
            if operator == "p2":
                opposite = self._opposite_polarity(reference.polarity)
                if opposite and any(module.polarity == opposite for module in modules):
                    candidates.append(slot_id)
            elif operator == "p1":
                if any(module.module_id != reference.module_id and module.semantic_family == reference.semantic_family for module in modules):
                    candidates.append(slot_id)
            elif operator == "p3":
                if any(module.module_id != reference.module_id for module in modules) or reference.parameters:
                    candidates.append(slot_id)
        return candidates

    def _reuse_candidates(self, operator: str, base: ParticlePosition,
                          cut_sets: Mapping[str, list[str]], slot_id: str) -> list[str]:
        """Return existing cut-set modules that satisfy an operator relation."""
        reference = self.registry.module(slot_id, base.as_mapping()[slot_id].module)
        opposite = self._opposite_polarity(reference.polarity)
        candidates = []
        for module_id in cut_sets.get(slot_id, ()):
            if module_id == reference.module_id:
                continue
            candidate = self.registry.module(slot_id, module_id)
            if operator == "p1" and candidate.semantic_family != reference.semantic_family:
                continue
            if operator == "p2" and candidate.polarity != opposite:
                continue
            # p3 only requires a different existing executable module.
            candidates.append(module_id)
        return candidates

    def _reuse_decision(self, operator: str, generation: int, parent_id: str) -> bool:
        """Make a reproducible reuse/new decision independent of thread order."""
        if not self.reuse_existing_modules or self.reuse_existing_probability <= 0.0:
            return False
        token = hashlib.sha256(
            f"reuse|{generation}|{parent_id}|{operator}".encode("utf-8")
        ).digest()
        value = int.from_bytes(token[:8], "big") / float(1 << 64)
        return value < self.reuse_existing_probability

    def _reuse_existing_child(
        self, operator: str, parent: ParticlePosition, base: ParticlePosition,
        cut_sets: Mapping[str, list[str]], *, generation: int, parent_id: str,
    ) -> ChildProposal | None:
        """Prefer one existing typed module before asking for a new module.

        The reuse prompt uses one LLM sequence-selection call.  If that call
        is malformed or unavailable, a deterministic valid module is selected
        from the same cut set, so reuse never adds a retry or a failed child.
        Returning ``None`` means this parent/operator takes the original
        description -> implementation path.
        """
        if operator not in {"p1", "p2", "p3"}:
            return None

        eligible_slots = [
            slot_id for slot_id in self.registry.slot_ids
            if self._reuse_candidates(operator, base, cut_sets, slot_id)
        ]
        if not eligible_slots:
            return None
        with self._reuse_stats_lock:
            self.reuse_stats["eligible"] += 1
        if not self._reuse_decision(operator, generation, parent_id):
            return None

        with self._reuse_stats_lock:
            self.reuse_stats["attempted"] += 1
        slot_id = eligible_slots[
            (generation + sum(ord(ch) for ch in parent_id)) % len(eligible_slots)
        ]
        candidates = self._reuse_candidates(operator, base, cut_sets, slot_id)
        allowed = {
            slot: [base.as_mapping()[slot].module]
            for slot in self.registry.slot_ids
        }
        allowed[slot_id] = list(candidates)
        reference = self.registry.module(slot_id, base.as_mapping()[slot_id].module)
        relation = (
            "the same semantic family" if operator == "p1"
            else "the opposite polarity" if operator == "p2"
            else "a different executable recipe"
        )
        prompt = (
            "REUSE_EXISTING_SEQUENCE phase. Return JSON only as "
            '{"sequence":[{"slot":"...","module":"...","params":{}}]}. '
            "Reuse an existing module; do not invent a module, description, recipe, "
            "or Python code. Change exactly one module slot and preserve all other "
            f"module IDs and parameters. For slot {slot_id}, choose an existing "
            f"candidate with {relation} relative to {reference.module_id}. "
            "Use the candidate module's default parameters.\n"
            f"REFERENCE_SEQUENCE:\n{self._position_prompt('BASE_SEQUENCE', base)}\n"
            f"CUT_SET_MODULES_WITH_EXECUTABLE_CODE:\n{self._cut_prompt(allowed)}"
        )
        try:
            position, calls = self._retry_sequence(
                prompt, allowed, generation, f"{operator}_reuse", max_retries=0
            )
            selected = position.as_mapping()[slot_id]
            if selected.module not in candidates:
                raise RuntimeError("reuse selector chose a non-candidate module")
            mapping = base.as_mapping().copy()
            mapping[slot_id] = ModuleChoice.create(
                slot_id, selected.module,
                self.registry.module(slot_id, selected.module).default_params(),
            )
            position = self.registry.validate_position(
                ParticlePosition(tuple(mapping[slot] for slot in self.registry.slot_ids))
            )
            with self._reuse_stats_lock:
                self.reuse_stats["llm_selected"] += 1
            return ChildProposal(
                operator=operator,
                position=position,
                set_position=None,
                base_position=base,
                prompt_ids=[f"reuse-existing:{operator}", *calls],
            )
        except Exception:
            # No additional LLM retry: preserve runtime and guarantee a legal
            # reused child whenever the cut set contains a valid candidate.
            digest = hashlib.sha256(
                f"reuse-fallback|{generation}|{parent_id}|{operator}|{slot_id}".encode("utf-8")
            ).digest()
            selected_id = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
            mapping = base.as_mapping().copy()
            mapping[slot_id] = ModuleChoice.create(
                slot_id, selected_id,
                self.registry.module(slot_id, selected_id).default_params(),
            )
            position = self.registry.validate_position(
                ParticlePosition(tuple(mapping[slot] for slot in self.registry.slot_ids))
            )
            with self._reuse_stats_lock:
                self.reuse_stats["deterministic_fallback"] += 1
            return ChildProposal(
                operator=operator,
                position=position,
                set_position=None,
                base_position=base,
                prompt_ids=[f"deterministic-reuse:{operator}"],
            )

    def _position_prompt(self, label: str, position: ParticlePosition) -> str:
        items = []
        for choice in position.choices:
            spec = self.registry.module(choice.slot, choice.module)
            item = {"slot": choice.slot, "module": choice.module, "params": choice.params_dict(), "code": self.compiler.module_code(choice.slot, choice.module, choice.params_dict())}
            if self.use_descriptions:
                item["description"] = spec.description
            items.append(item)
        return f"{label}:\n" + json.dumps(items, ensure_ascii=False, indent=2)

    def _parse_sequence(self, response: str, allowed: Mapping[str, list[str]]) -> ParticlePosition | None:
        data = _json_object(response)
        raw = data.get("sequence") if data else None
        if not isinstance(raw, list):
            return None
        parsed = {}
        for item in raw:
            if not isinstance(item, Mapping):
                return None
            slot, module = str(item.get("slot")), str(item.get("module"))
            if slot not in allowed or module not in allowed[slot] or slot in parsed:
                return None
            try:
                parsed[slot] = self.registry.validate_choice(ModuleChoice.create(slot, module, item.get("params", {})))
            except Exception:
                return None
        if set(parsed) != set(self.registry.slot_ids):
            return None
        return self.registry.validate_position(ParticlePosition(tuple(parsed[slot] for slot in self.registry.slot_ids)))

    def generate_sequence(self, cut_sets: Mapping[str, list[str]], *, generation=0, operator="p0", extra="") -> tuple[ParticlePosition, list[str]]:
        prompt = (
            "SEQUENCE_SELECTION phase. Return JSON only as {\"sequence\":[{\"slot\":\"...\",\"module\":\"...\",\"params\":{}}]}. "
            "Choose exactly one module from each cut set and keep parameters valid. Do not write Python.\n"
            f"CUT_SET_MODULES_WITH_EXECUTABLE_CODE:\n{self._cut_prompt(cut_sets)}\n{extra}"
        )
        allowed = {slot: list(ids) for slot, ids in cut_sets.items()}
        return self._retry_sequence(prompt, allowed, generation, operator)

    def _retry_sequence(self, prompt, allowed, generation, operator, max_retries=None):
        def validate(response):
            return self._parse_sequence(response, allowed)
        retries = self.max_retries if max_retries is None else int(max_retries)
        position, _response, call_id = self.llm.retry(prompt, validate, generation=generation, operator=operator, max_retries=retries)
        return position, [call_id]

    def _description_and_recipe(self, slot_id: str, operator: str, reference: ModuleSpec, context: str, generation: int) -> tuple[ModuleProposal, list[str]]:
        target_modules = self._module_payload(slot_id, [module.module_id for module in self.registry.slot(slot_id).modules])
        required = "same semantic family" if operator == "p1" else (f"opposite polarity {self._opposite_polarity(reference.polarity)}" if operator == "p2" else "different executable recipe")
        description_prompt = (
            f"{operator.upper()}_DESCRIPTION phase. Return JSON only: {{\"description\":\"one sentence\"}}.\n"
            f"TARGET_SLOT={slot_id}; REQUIRED_RELATION={required}. The reference module is {reference.module_id}.\n"
            f"TARGET_SLOT_CANDIDATES:\n{json.dumps(target_modules, ensure_ascii=False, indent=2)}\n{context}"
        )
        def valid_desc(response):
            data = _json_object(response)
            value = data.get("description") if data else None
            return value.strip() if isinstance(value, str) and value.strip() and len(value.strip()) <= 240 else None
        description, desc_response, desc_id = self.llm.retry(description_prompt, valid_desc, generation=generation, operator=f"{operator}_description", max_retries=self.max_retries)
        impl_prompt = (
            f"{operator.upper()}_IMPLEMENTATION phase. Return JSON only with recipe={{kind,family,polarity,parameters}} and parameter_schema. "
            "Use only the typed recipe kinds listed in the candidate context; emit no Python. "
            "Choose an implementation (kind and parameter values) not already present in TARGET_SLOT_CANDIDATES.\n"
            f"TARGET_SLOT={slot_id}; REQUIRED_RELATION={required}; Design sentence: {description}\n"
            f"TARGET_SLOT_CANDIDATES:\n{json.dumps(target_modules, ensure_ascii=False, indent=2)}\n{context}"
        )
        allowed = {str(m.recipe.get("kind")): m for m in self.registry.slot(slot_id).modules}
        def valid_impl(response):
            data = _json_object(response)
            recipe = data.get("recipe") if data else None
            if not isinstance(recipe, Mapping) or str(recipe.get("kind")) not in allowed:
                return None
            template = allowed[str(recipe["kind"])]
            if operator == "p1" and template.semantic_family != reference.semantic_family:
                return None
            if operator == "p2" and template.polarity != self._opposite_polarity(reference.polarity):
                return None
            # Parameter names, kinds and bounds are owned by the adapter.
            # The LLM can only provide values for this fixed recipe schema.
            specs = tuple(template.parameters)
            values = template.default_params()
            supplied = dict(recipe.get("parameters", {}))
            if set(supplied) - {p.name for p in specs}:
                return None
            values.update(supplied)
            try:
                values = {p.name: p.validate(values.get(p.name, p.default)) for p in specs}
            except Exception:
                return None
            if any(
                str(existing.recipe.get("kind")) == str(recipe["kind"])
                and existing.default_params() == values
                for existing in self.registry.slot(slot_id).modules
            ):
                return None
            specs = tuple(replace(parameter, default=values[parameter.name]) for parameter in specs)
            clean = {"kind": str(recipe["kind"]), "family": template.semantic_family, "polarity": template.polarity}
            return clean, specs, values
        (recipe, specs, values), impl_response, impl_id = self.llm.retry(impl_prompt, valid_impl, generation=generation, operator=f"{operator}_implementation", max_retries=self.max_retries)
        proposal = ModuleProposal(slot_id, description, recipe, specs, operator, reference_module=reference.module_id, prompt_ids=[desc_id, impl_id], description_response=desc_response, implementation_response=impl_response)
        proposal.recipe["parameters"] = values
        return proposal, [desc_id, impl_id]

    def initial_sequences(self, count: int, *, seed=0, require_tunable=True) -> list[ParticlePosition]:
        rng = random.Random(seed)
        output = []
        seen = set()
        for index in range(count):
            cut_sets = {slot.slot_id: [m.module_id for m in slot.modules] for slot in self.registry.slots}
            position, _ = self.generate_sequence(cut_sets, generation=0, operator="p0", extra="INITIAL_SEQUENCE; generate a legal diverse sequence.")
            if require_tunable and not any(choice.params_dict() for choice in position.choices):
                # This is a validation failure, not a random substitution.
                raise RuntimeError("p0 sequence has no tunable parameter while p4 is enabled")
            key = position.canonical(self.registry.slot_ids)
            if key in seen:
                # Ask the LLM again with a deterministic diversity hint.
                position, _ = self.generate_sequence(cut_sets, generation=0, operator="p0", extra=f"INITIAL_SEQUENCE retry {index}; choose a different legal sequence.")
                key = position.canonical(self.registry.slot_ids)
            if key in seen:
                raise RuntimeError("LLM failed to generate distinct p0 positions")
            seen.add(key)
            output.append(position)
        return output

    def base_sequence(self, cut_sets, *, generation, parent_id):
        return self.generate_sequence(cut_sets, generation=generation, operator="p0", extra=f"BASE_SEQUENCE for parent {parent_id}; do not inspect the old parent sequence.")

    @staticmethod
    def _parameter_candidates(spec: ParameterSpec, current: Any) -> list[Any]:
        """Return a small deterministic set of valid alternative values."""
        values = []

        def add(value):
            try:
                value = spec.validate(value)
            except Exception:
                return
            if value != current and value not in values:
                values.append(value)

        if spec.kind == "choice":
            for value in spec.choices:
                add(value)
        elif spec.kind == "int":
            add(current - 1)
            add(current + 1)
            if spec.minimum is not None:
                add(int(spec.minimum))
            if spec.maximum is not None:
                add(int(spec.maximum))
            if spec.minimum is not None and spec.maximum is not None:
                add(int(round((spec.minimum + spec.maximum) / 2)))
        elif spec.kind == "float":
            add(float(current) - 0.1)
            add(float(current) + 0.1)
            if spec.minimum is not None:
                add(float(spec.minimum))
            if spec.maximum is not None:
                add(float(spec.maximum))
            if spec.minimum is not None and spec.maximum is not None:
                add((float(spec.minimum) + float(spec.maximum)) / 2.0)
        return values

    def _fallback_new_module(
        self, slot_id: str, operator: str, reference: ModuleSpec, generation: int
    ) -> ModuleProposal | None:
        """Build a legal new typed module when the LLM protocol is invalid.

        The LLM remains the primary proposal source.  This adapter-owned
        fallback keeps an operator batch productive when a provider returns
        malformed JSON or repeats an existing recipe.
        """
        opposite = self._opposite_polarity(reference.polarity)
        templates = []
        for template in self.registry.slot(slot_id).modules:
            if operator == "p1" and template.semantic_family != reference.semantic_family:
                continue
            if operator == "p2" and template.polarity != opposite:
                continue
            templates.append(template)
        templates.sort(key=lambda item: (item.module_id == reference.module_id, item.module_id))

        existing = {
            (str(item.recipe.get("kind")), tuple(sorted(item.default_params().items())))
            for item in self.registry.slot(slot_id).modules
        }
        for template in templates:
            defaults = template.default_params()
            variants = [defaults]
            for parameter in template.parameters:
                current = defaults.get(parameter.name, parameter.default)
                for value in self._parameter_candidates(parameter, current):
                    candidate = dict(defaults)
                    candidate[parameter.name] = value
                    variants.append(candidate)
            for values in variants:
                signature = (
                    str(template.recipe.get("kind")),
                    tuple(sorted(values.items())),
                )
                if signature in existing:
                    continue
                specs = tuple(
                    replace(parameter, default=values[parameter.name])
                    for parameter in template.parameters
                )
                recipe = {
                    "kind": str(template.recipe.get("kind")),
                    "family": template.semantic_family,
                    "polarity": template.polarity,
                    "parameters": dict(values),
                }
                description = (
                    f"Deterministic {operator} fallback using {recipe['kind']} "
                    "with a validated alternative parameterization."
                )
                return ModuleProposal(
                    slot=slot_id,
                    description=description,
                    recipe=recipe,
                    parameters=specs,
                    source_operator=operator,
                    relation=("similar" if operator == "p1" else "opposite" if operator == "p2" else "revised"),
                    reference_module=reference.module_id,
                )
        return None

    def _fallback_parameter_position(self, base: ParticlePosition) -> ParticlePosition | None:
        """Change one bounded parameter while preserving every module ID."""
        mapping = base.as_mapping()
        for slot_id in self.registry.slot_ids:
            choice = mapping[slot_id]
            spec = self.registry.module(slot_id, choice.module)
            current = choice.params_dict()
            for parameter in spec.parameters:
                for value in self._parameter_candidates(
                    parameter, current.get(parameter.name, parameter.default)
                ):
                    params = dict(current)
                    params[parameter.name] = value
                    candidate = dict(mapping)
                    candidate[slot_id] = ModuleChoice.create(slot_id, choice.module, params)
                    try:
                        position = self.registry.validate_position(
                            ParticlePosition(tuple(candidate[slot] for slot in self.registry.slot_ids))
                        )
                        if self.compiler.compile(position)[0] != self.compiler.compile(base)[0]:
                            return position
                    except Exception:
                        continue
        return None

    def _fallback_existing_position(
        self, operator: str, base: ParticlePosition, cut_sets: Mapping[str, list[str]]
    ) -> ParticlePosition | None:
        """Last-resort legal sequence change using the current cut sets."""
        for slot_id in self.registry.slot_ids:
            reference = self.registry.module(slot_id, base.as_mapping()[slot_id].module)
            opposite = self._opposite_polarity(reference.polarity)
            for module_id in cut_sets.get(slot_id, ()):
                if module_id == reference.module_id:
                    continue
                candidate = self.registry.module(slot_id, module_id)
                if operator == "p1" and candidate.semantic_family != reference.semantic_family:
                    continue
                if operator == "p2" and candidate.polarity != opposite:
                    continue
                mapping = base.as_mapping().copy()
                mapping[slot_id] = ModuleChoice.create(slot_id, module_id, candidate.default_params())
                return self.registry.validate_position(
                    ParticlePosition(tuple(mapping[slot] for slot in self.registry.slot_ids))
                )
        return None

    def propose(self, operator: str, parent: ParticlePosition, base: ParticlePosition, cut_sets: Mapping[str, list[str]], *, generation: int, parent_id: str) -> ChildProposal:
        operator = operator.lower()
        if operator == "p0":
            position, calls = self.base_sequence(cut_sets, generation=generation, parent_id=parent_id)
            return ChildProposal(operator, position, None, base_position=position, prompt_ids=calls)
        if operator not in {"p1", "p2", "p3", "p4"}:
            raise ValueError(f"unknown heuristic operator {operator}")
        reused = self._reuse_existing_child(
            operator, parent, base, cut_sets,
            generation=generation, parent_id=parent_id,
        )
        if reused is not None:
            return reused
        if operator == "p4":
            allowed = {slot: list(ids) for slot, ids in cut_sets.items()}
            extra = self._position_prompt("BASE_SEQUENCE", base)
            prompt = (
                "P4_PARAMETER_ONLY phase. Return JSON with exactly the same module IDs and descriptions as BASE_SEQUENCE, "
                "but change at least one bounded parameter. Do not introduce modules or Python.\n"
                f"CUT_SET_MODULES_WITH_EXECUTABLE_CODE:\n{self._cut_prompt(cut_sets)}\n" + extra
            )
            try:
                position, calls = self._retry_sequence(prompt, allowed, generation, "p4")
                if all(position.as_mapping()[slot].params == base.as_mapping()[slot].params for slot in self.registry.slot_ids):
                    raise RuntimeError("p4 did not change a parameter")
                if any(position.as_mapping()[slot].module != base.as_mapping()[slot].module for slot in self.registry.slot_ids):
                    raise RuntimeError("p4 changed a module ID")
                before = self.compiler.compile(base)[0]
                after = self.compiler.compile(position)[0]
                if before == after:
                    raise RuntimeError("p4 parameter change did not change compiled source")
            except Exception:
                position = self._fallback_parameter_position(base)
                if position is None:
                    raise RuntimeError("p4 has no valid bounded parameter variation")
                calls = ["adapter-fallback:p4"]
            return ChildProposal(operator, position, None, base_position=base, prompt_ids=calls)

        reference = base
        context = self._position_prompt("BASE_SEQUENCE", base)
        if operator in {"p1", "p2"}:
            context += "\n" + self._position_prompt("OLD_PARENT_SEQUENCE", parent)
        slots = self._candidate_slots(operator, base)
        if not slots:
            raise RuntimeError(f"no adapter-supported slot can satisfy {operator}")
        slot_id = slots[(generation + sum(ord(ch) for ch in parent_id)) % len(slots)]
        ref_choice = reference.as_mapping()[slot_id]
        ref_spec = self.registry.module(slot_id, ref_choice.module)
        try:
            proposal, calls = self._description_and_recipe(slot_id, operator, ref_spec, context, generation)
        except Exception as error:
            proposal = self._fallback_new_module(slot_id, operator, ref_spec, generation)
            calls = [f"adapter-fallback:{operator}"]
            if proposal is None:
                position = self._fallback_existing_position(operator, base, cut_sets)
                if position is None:
                    raise RuntimeError(f"{operator} could not construct a legal fallback") from error
                return ChildProposal(operator, position, None, base_position=base, prompt_ids=calls)
        if operator == "p1":
            if proposal.recipe.get("family") != ref_spec.semantic_family:
                raise RuntimeError("p1 proposal is not in the declared similar semantic family")
        elif operator == "p2":
            if proposal.recipe.get("polarity") != self._opposite_polarity(ref_spec.polarity):
                raise RuntimeError("p2 proposal does not declare the inverse semantic direction")
        reference_signature = {"kind": ref_spec.recipe.get("kind"), "parameters": ref_choice.params_dict()}
        proposed_signature = {"kind": proposal.recipe.get("kind"), "parameters": proposal.recipe.get("parameters", {})}
        if proposed_signature == reference_signature:
            raise RuntimeError(f"{operator} did not change the executable implementation")
        proposal.module_id = f"llm_{slot_id}_{hashlib.sha256(json.dumps({'kind': proposal.recipe['kind'], 'parameters': [p.to_dict() for p in proposal.parameters]}, sort_keys=True, default=str).encode()).hexdigest()[:12]}"
        with self._reuse_stats_lock:
            self.reuse_stats["new_module"] += 1
        mapping = base.as_mapping().copy()
        mapping[slot_id] = ModuleChoice.create(slot_id, proposal.module_id, proposal.recipe.get("parameters", {}))
        return ChildProposal(operator, ParticlePosition(tuple(mapping[slot] for slot in self.registry.slot_ids)), None, [proposal], base, calls)


class DeterministicFakeLLM:
    """Small offline LLM for tests; it follows the two-step protocol."""

    def __init__(self, prototype: ModuleRegistry):
        self.prototype = prototype
        self.counter = 0
        self.sequence_counter = 0

    @staticmethod
    def _schema_map(raw: Any) -> dict[str, dict[str, Any]]:
        """Normalize the adapter's serialized parameter schema.

        ``TSPModuleEvolution._module_payload`` deliberately emits the same
        list-of-``ParameterSpec`` objects that the real LLM sees.  The fake
        must consume that wire format too; treating it as a mapping was an
        offline-only bug and could hide protocol errors in smoke tests.
        """
        if isinstance(raw, list):
            return {
                str(item["name"]): dict(item)
                for item in raw
                if isinstance(item, Mapping) and isinstance(item.get("name"), str)
            }
        if isinstance(raw, Mapping):
            return {
                str(name): dict(rule) if isinstance(rule, Mapping) else {}
                for name, rule in raw.items()
            }
        return {}

    @staticmethod
    def _json_after(prompt: str, marker: str):
        start = prompt.find(marker)
        if start < 0:
            return None
        # The marker is immediately followed by one JSON value.  Scanning for
        # a later ``[``/``{`` can accidentally return a nested list (and made
        # the fake interpret a dict payload as a list when code contained
        # punctuation).  Decode only the first value after the marker.
        tail = prompt[start + len(marker):].lstrip()
        if not tail or tail[0] not in "[{":
            return None
        try:
            return json.JSONDecoder().raw_decode(tail)[0]
        except json.JSONDecodeError:
            return None

    def get_response(self, prompt: str) -> str:
        self.counter += 1
        if "MODULE_DESCRIPTION" in prompt or "P1_DESCRIPTION" in prompt or "P2_DESCRIPTION" in prompt or "P3_DESCRIPTION" in prompt:
            return json.dumps({"description": f"deterministic typed module proposal {self.counter}."})
        if "MODULE_IMPLEMENTATION" in prompt or "P1_IMPLEMENTATION" in prompt or "P2_IMPLEMENTATION" in prompt or "P3_IMPLEMENTATION" in prompt:
            slot = re.search(r"TARGET_SLOT=([EFHTW])|\"slot\"\s*:\s*\"([EFHTW])\"", prompt)
            slot_id = (slot.group(1) or slot.group(2)) if slot else "E"
            candidates = self.prototype.slot(slot_id).modules
            relation = "P2_IMPLEMENTATION" in prompt
            required = re.search(r"opposite polarity ([A-Za-z_]+)", prompt)
            if relation and required:
                choices = [module for module in candidates if module.polarity == required.group(1)]
                template = choices[self.counter % len(choices)] if choices else candidates[self.counter % len(candidates)]
            else:
                template = candidates[(self.counter + 1) % len(candidates)]
            # For p2, preserve the trusted inverse-polarity choice even when
            # that recipe has no tunable parameters.  Replacing it with an
            # arbitrary parameterized template can silently destroy the
            # opposite-direction guarantee.  p1/p3 may use a parameterized
            # alternative when available, because their relation is not
            # polarity-based.
            if not template.parameters and not relation:
                parameterized = [module for module in candidates if module.parameters]
                if parameterized:
                    template = parameterized[self.counter % len(parameterized)]
            params = template.default_params()
            for name, value in list(params.items()):
                if isinstance(value, int):
                    spec = next(item for item in template.parameters if item.name == name)
                    params[name] = min(int(spec.maximum), value + max(1, self.counter % 20))
                elif isinstance(value, float):
                    spec = next(item for item in template.parameters if item.name == name)
                    params[name] = min(float(spec.maximum), value + 0.1 * max(1, self.counter % 20))
            return json.dumps({"recipe": {"kind": template.recipe["kind"], "family": template.semantic_family, "polarity": template.polarity, "parameters": params}, "parameter_schema": [p.to_dict() for p in template.parameters]})
        if "P4_PARAMETER_ONLY" in prompt:
            sequence = self._json_after(prompt, "BASE_SEQUENCE:") or []
            candidates = self._json_after(prompt, "CUT_SET_MODULES_WITH_EXECUTABLE_CODE:") or {}
            schemas = {
                item.get("module"): self._schema_map(item.get("parameters", {}))
                for entries in candidates.values() for item in entries
            }
            changed = False
            for item in sequence:
                params = dict(item.get("params", {}))
                schema = schemas.get(item.get("module"), {})
                for name, value in list(params.items()):
                    rule = schema.get(name, {})
                    minimum, maximum = rule.get("minimum"), rule.get("maximum")
                    if isinstance(value, int) and not isinstance(value, bool):
                        candidate = value + 1
                        if maximum is not None and candidate > maximum:
                            candidate = value - 1
                        if candidate != value:
                            params[name] = candidate
                            changed = True
                    elif isinstance(value, float):
                        candidate = value + 0.1
                        if maximum is not None and candidate > maximum:
                            candidate = value - 0.1
                        if minimum is not None and candidate < minimum:
                            candidate = value
                        if candidate != value:
                            params[name] = candidate
                            changed = True
                item["params"] = params
            if not changed and sequence:
                # The fake deliberately emits an invalid response here; the
                # protocol test can verify p4 rejects it rather than changing
                # a module ID or silently falling back.
                return json.dumps({"sequence": sequence})
            return json.dumps({"sequence": sequence})
        # Extract candidate module IDs from the final JSON payload and choose
        # a deterministic, parameterized legal combination.
        payload = self._json_after(prompt, "CUT_SET_MODULES_WITH_EXECUTABLE_CODE:") or {}
        self.sequence_counter += 1
        sequence_index = self.sequence_counter - 1
        sequence = []
        for slot_index, (slot_id, entries) in enumerate(payload.items()):
            if not entries:
                continue
            entry = entries[(sequence_index + slot_index) % len(entries)]
            params = dict(entry.get("default_params", {}))
            for parameter_index, (name, rule) in enumerate(self._schema_map(entry.get("parameters", {})).items()):
                if name not in params:
                    continue
                value = params[name]
                minimum, maximum = rule.get("minimum"), rule.get("maximum")
                if isinstance(value, int) and not isinstance(value, bool):
                    if minimum is not None and maximum is not None and int(maximum) >= int(minimum):
                        span = int(maximum) - int(minimum) + 1
                        value = int(minimum) + ((sequence_index + slot_index + parameter_index) % span)
                    else:
                        value = value + max(0, sequence_index % 5)
                elif isinstance(value, float):
                    if minimum is not None and maximum is not None and float(maximum) >= float(minimum):
                        # Eleven deterministic points provide a second
                        # dimension of diversity after the ten module choices
                        # repeat, while remaining inside the adapter bounds.
                        fraction = ((sequence_index + slot_index + parameter_index) % 11) / 10.0
                        value = float(minimum) + (float(maximum) - float(minimum)) * fraction
                    else:
                        value = value + 0.1 * (sequence_index % 5)
                params[name] = value
            sequence.append({"slot": slot_id, "module": entry["module"], "params": params})
        return json.dumps({"sequence": sequence})
