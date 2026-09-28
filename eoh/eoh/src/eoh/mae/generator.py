from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Protocol

from .models import CutItem, ProgramIndividual

logger = logging.getLogger("eoh.mae")


@dataclass
class ProgramDraft:
    description: str | None
    code: str | None
    prompt_hash: str | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None
    llm_seconds: float | None = None
    llm_attempts: int = 0
    raw_response: str | None = field(default=None, repr=False)


class ProgramGenerator(Protocol):
    def generate_h0(self, agent_id: int, sample_index: int) -> ProgramDraft: ...

    def select_cut(
        self,
        agent_id: int,
        guidance: str,
        cut: list[CutItem],
        individuals: dict[str, ProgramIndividual],
        selection_size: int,
    ) -> list[str]: ...

    def generate_variation(
        self,
        agent_id: int,
        guidance: str,
        variation: str,
        parents: list[ProgramIndividual],
        selected: list[ProgramIndividual],
    ) -> ProgramDraft: ...


class LLMProgramGenerator:
    """Prompt and response adapter for H0-H5 complete-program generation."""

    def __init__(
        self,
        llm,
        task_description: str,
        template_program: str,
        *,
        retries: int = 3,
        require_description: bool = True,
        trace_path: str | Path | None = None,
        record_full_prompts: bool = True,
        record_full_responses: bool = True,
    ) -> None:
        self.llm = llm
        self.task_description = task_description
        self.template_program = template_program.strip()
        self.retries = max(1, retries)
        self.require_description = require_description
        self.trace_path = Path(trace_path) if trace_path else None
        self.record_full_prompts = record_full_prompts
        self.record_full_responses = record_full_responses
        self._trace_lock = threading.Lock()

    def _trace(self, call_type: str, prompt: str, response: str | None, attempt: int) -> None:
        if self.trace_path is None:
            return
        record = {
            "call_type": call_type,
            "prompt_hash": self._prompt_hash(prompt),
            "attempt": attempt,
            "prompt_chars": len(prompt),
            "response_chars": len(response) if response else 0,
            "prompt": prompt if self.record_full_prompts else None,
            "response": response if self.record_full_responses else None,
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_lock, self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_code(response: str) -> str | None:
        fenced = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        candidates = fenced or []
        if not candidates:
            start = re.search(r"^(?:import |from |def |class |@)", response, re.MULTILINE)
            if start:
                raw = response[start.start():].strip()
                lines = raw.splitlines()
                for end in range(len(lines), 0, -1):
                    candidate = "\n".join(lines[:end]).strip()
                    try:
                        ast.parse(candidate)
                        candidates = [candidate]
                        break
                    except SyntaxError:
                        continue
        for candidate in candidates:
            cleaned = re.sub(r"^\s*\{[^{}]+\}\s*", "", candidate, count=1).strip()
            try:
                ast.parse(cleaned)
                return cleaned
            except SyntaxError:
                continue
        return None

    @staticmethod
    def _extract_description(response: str, code: str | None) -> str | None:
        prefix = response
        if "```" in response:
            prefix = response.split("```", 1)[0]
        elif code:
            index = response.find(code[:80])
            if index >= 0:
                prefix = response[:index]
        matches = re.findall(r"\{([^{}]{8,500})\}", prefix, re.DOTALL)
        if matches:
            return " ".join(matches[0].split())
        plain = " ".join(prefix.strip().split())
        return plain[:500] if len(plain) >= 8 else None

    def _call_code(self, prompt: str, call_type: str) -> ProgramDraft:
        prompt_hash = self._prompt_hash(prompt)
        last_response = None
        started = time.perf_counter()
        for attempt in range(1, self.retries + 1):
            try:
                response = self.llm.get_response(prompt)
            except Exception as exc:
                logger.debug("LLM attempt %d failed: %s", attempt, exc)
                response = None
            self._trace(call_type, prompt, response, attempt)
            if not response:
                continue
            last_response = response
            code = self._extract_code(response)
            description = self._extract_description(response, code)
            if code and (description or not self.require_description):
                return ProgramDraft(
                    description=description or "Description disabled.",
                    code=code,
                    prompt_hash=prompt_hash,
                    llm_seconds=time.perf_counter() - started,
                    llm_attempts=attempt,
                    raw_response=response,
                )
        return ProgramDraft(
            description=None,
            code=None,
            prompt_hash=prompt_hash,
            failure_stage="llm_parse",
            failure_reason="LLM did not return a parsable description and program",
            llm_seconds=time.perf_counter() - started,
            llm_attempts=self.retries,
            raw_response=last_response,
        )

    def _contract(self) -> str:
        return (
            f"Task:\n{self.task_description}\n\n"
            "The result must implement exactly this template and preserve its function name, "
            f"arguments, and return contract:\n{self.template_program}\n\n"
            "Return exactly two parts: first one concise algorithm description inside braces; "
            "then the complete Python code. Do not return tests or explanatory prose after the code."
        )

    def generate_h0(self, agent_id: int, sample_index: int) -> ProgramDraft:
        prompt = (
            f"You are initializing agent {agent_id}, sample {sample_index}, in a multi-agent "
            "heuristic evolution system. Design a valid and distinct GLS heuristic without "
            "copying a supplied parent. Prefer simple vectorized numerical operations and make "
            "a fresh output matrix instead of modifying inputs.\n\n"
            + self._contract()
        )
        return self._call_code(prompt, "H0")

    @staticmethod
    def _cut_block(
        cut: list[CutItem], individuals: dict[str, ProgramIndividual]
    ) -> str:
        blocks = []
        for rank, item in enumerate(cut, 1):
            individual = individuals[item.individual_id]
            blocks.append(
                f"Candidate {rank}\n"
                f"id: {individual.individual_id}\n"
                f"probability: {item.probability:.6f}\n"
                f"source: {item.source}\n"
                f"fitness: {individual.fitness}\n"
                f"description: {individual.description}\n"
                f"code:\n{individual.code}"
            )
        return "\n\n".join(blocks)

    def select_cut(
        self,
        agent_id: int,
        guidance: str,
        cut: list[CutItem],
        individuals: dict[str, ProgramIndividual],
        selection_size: int,
    ) -> list[str]:
        goal = (
            "prefer useful private historical experience"
            if guidance == "H1"
            else "prefer useful globally shared experience while retaining diversity"
        )
        allowed = [item.individual_id for item in cut]
        prompt = (
            f"Agent {agent_id} is applying {guidance}. {goal}. Rank at most "
            f"{min(selection_size, len(cut))} candidates from the cut set below. Consider "
            "probability, algorithm description, code behavior, fitness, and diversity. "
            "Return only JSON in the form {\"selected_ids\":[\"id1\",\"id2\"]}.\n\n"
            + self._cut_block(cut, individuals)
        )
        for attempt in range(1, self.retries + 1):
            try:
                response = self.llm.get_response(prompt)
            except Exception:
                response = None
            if not response:
                self._trace(f"{guidance}-select", prompt, response, attempt)
                continue
            self._trace(f"{guidance}-select", prompt, response, attempt)
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                continue
            try:
                parsed = json.loads(match.group(0))
                result = []
                for item_id in parsed.get("selected_ids", []):
                    if item_id in allowed and item_id not in result:
                        result.append(item_id)
                if len(result) >= 2:
                    return result[:selection_size]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return []

    @staticmethod
    def _parent_block(parents: list[ProgramIndividual]) -> str:
        return "\n\n".join(
            f"Parent {index}\n"
            f"id: {parent.individual_id}\n"
            f"description: {parent.description}\n"
            f"fitness: {parent.fitness}\n"
            f"code:\n{parent.code}"
            for index, parent in enumerate(parents, 1)
        )

    def generate_variation(
        self,
        agent_id: int,
        guidance: str,
        variation: str,
        parents: list[ProgramIndividual],
        selected: list[ProgramIndividual],
    ) -> ProgramDraft:
        if variation == "H3":
            instruction = (
                "Find the useful common behavior of the two parents and create a new program "
                "that is semantically similar to both, but is not a verbatim copy."
            )
        elif variation == "H4":
            instruction = (
                "Use the two parents only as contrastive references. Create a structurally or "
                "semantically different exploration program while preserving the task contract."
            )
        elif variation == "H5":
            instruction = (
                "Treat the supplied parent as the most valuable candidate. Perform a STRICT "
                "parameter-only mutation: copy the complete parent program and change only one "
                "or more existing numeric literals. Do not add or remove imports, statements, "
                "function calls, variables, branches, loops, array operations, or return logic. "
                "Do not rewrite the algorithm. The parameter-insensitive AST must remain identical "
                "to the parent; only numeric values may differ. Preserve the exact function name, "
                "arguments, and return contract."
            )
        else:
            return ProgramDraft(None, None, failure_stage="unknown_operator", failure_reason=variation)

        selected_descriptions = "\n".join(
            f"- {item.individual_id}: {item.description}" for item in selected
        )
        prompt = (
            f"Agent {agent_id} is using operator path {guidance}->{variation}.\n"
            f"{instruction}\n\nSelected cut-set descriptions:\n{selected_descriptions}\n\n"
            f"Parents:\n{self._parent_block(parents)}\n\n"
            + self._contract()
        )
        return self._call_code(prompt, f"{guidance}->{variation}")
