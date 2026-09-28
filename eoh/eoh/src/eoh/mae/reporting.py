from __future__ import annotations

import csv
import json
import math
import os
import platform
import socket
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class ExperimentRecorder:
    """Thread-safe structured telemetry for experiments and debugging.

    Every event is written to telemetry/events.jsonl. Important event families
    are also copied into smaller topic files, which makes later paper analysis
    possible without parsing the human-readable run log.
    """

    _TOPIC_FILES = {
        "candidate": "candidate_lifecycle.jsonl",
        "evaluation": "evaluations.jsonl",
        "selection": "selections.jsonl",
        "velocity": "velocity_updates.jsonl",
        "cut": "cut_decisions.jsonl",
        "transition": "agent_transitions.jsonl",
        "progress": "progress.jsonl",
        "stage": "stage_timings.jsonl",
        "best": "best_updates.jsonl",
        "error": "errors.jsonl",
    }

    def __init__(self, output_dir: Path, *, detailed: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.telemetry_dir = self.output_dir / "telemetry"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.detailed = detailed
        self._started_wall = time.time()
        self._started_mono = time.perf_counter()
        self._sequence = 0
        self._lock = threading.RLock()

    def _base_record(
        self,
        event: str,
        *,
        generation: int | None,
        agent_id: int | None,
        topic: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._sequence += 1
        return {
            "sequence": self._sequence,
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime()
            ),
            "elapsed_seconds": time.perf_counter() - self._started_mono,
            "event": event,
            "topic": topic,
            "generation": generation,
            "agent_id": agent_id,
            "thread": threading.current_thread().name,
            **_json_safe(payload),
        }

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def emit(
        self,
        event: str,
        *,
        generation: int | None = None,
        agent_id: int | None = None,
        topic: str = "event",
        detailed_only: bool = False,
        **payload: Any,
    ) -> dict[str, Any] | None:
        if detailed_only and not self.detailed:
            return None
        with self._lock:
            record = self._base_record(
                event,
                generation=generation,
                agent_id=agent_id,
                topic=topic,
                payload=payload,
            )
            self._append(self.telemetry_dir / "events.jsonl", record)
            topic_file = self._TOPIC_FILES.get(topic)
            if topic_file:
                self._append(self.telemetry_dir / topic_file, record)
            return record

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        generation: int | None = None,
        agent_id: int | None = None,
        **payload: Any,
    ) -> Iterator[None]:
        started = time.perf_counter()
        self.emit(
            "stage_started",
            generation=generation,
            agent_id=agent_id,
            topic="stage",
            stage=name,
            **payload,
        )
        try:
            yield
        except Exception as exc:
            self.emit(
                "stage_failed",
                generation=generation,
                agent_id=agent_id,
                topic="stage",
                stage=name,
                duration_seconds=time.perf_counter() - started,
                exception_type=type(exc).__name__,
                exception=str(exc),
            )
            raise
        else:
            self.emit(
                "stage_finished",
                generation=generation,
                agent_id=agent_id,
                topic="stage",
                stage=name,
                duration_seconds=time.perf_counter() - started,
            )

    def record_environment(self, config: dict[str, Any], problem: Any) -> None:
        self.emit(
            "run_environment",
            topic="event",
            python=sys.version,
            platform=platform.platform(),
            hostname=socket.gethostname(),
            pid=os.getpid(),
            cpu_count=os.cpu_count(),
            config=config,
            problem_class=f"{type(problem).__module__}.{type(problem).__qualname__}",
            problem_timeout=getattr(problem, "timeout", None),
        )

    def record_generation_metrics(self, record: dict[str, Any]) -> None:
        safe = _json_safe(record)
        with self._lock:
            self._append(self.output_dir / "metrics" / "generation_metrics.jsonl", safe)
            csv_path = self.output_dir / "metrics" / "generation_metrics.csv"
            columns = [
                "generation", "best_fitness", "population_mean", "population_std",
                "population_min", "population_max", "generated_candidates",
                "valid_candidates", "valid_rate", "surviving_candidates",
                "surviving_children",
                "independent_evaluations", "cache_hits", "archive_size",
                "generation_seconds", "generated_by_path", "valid_by_path",
                "survived_by_path", "failures_by_stage",
            ]
            row = {
                key: (
                    json.dumps(safe.get(key, {}), ensure_ascii=False, sort_keys=True)
                    if isinstance(safe.get(key), (dict, list))
                    else safe.get(key)
                )
                for key in columns
            }
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    def record_agent_metrics(self, records: list[dict[str, Any]]) -> None:
        path = self.output_dir / "metrics" / "agent_metrics.jsonl"
        with self._lock:
            for record in records:
                self._append(path, _json_safe(record))

    def record_operator_metrics(self, records: list[dict[str, Any]]) -> None:
        path = self.output_dir / "metrics" / "operator_metrics.jsonl"
        with self._lock:
            for record in records:
                self._append(path, _json_safe(record))
