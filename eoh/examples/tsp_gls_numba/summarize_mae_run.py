"""Build paper-friendly intermediate CSV/JSON summaries from a MA-EoH run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), (dict, list)) else row.get(key)
                )
                for key in fields
            })


def aggregate_operators(
    records: list[dict[str, Any]], objective_direction: str
) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "generated": 0, "valid": 0, "survived": 0,
        "recorded_failures": 0, "best_values": [],
    })
    for record in records:
        item = totals[str(record.get("operator_path"))]
        for key in ("generated", "valid", "survived", "recorded_failures"):
            item[key] += int(record.get(key) or 0)
        if record.get("best_fitness") is not None:
            item["best_values"].append(float(record["best_fitness"]))

    rows = []
    for operator_path, values in sorted(totals.items()):
        generated = values["generated"]
        valid = values["valid"]
        best_values = values.pop("best_values")
        rows.append({
            "operator_path": operator_path,
            **values,
            "valid_rate": valid / generated if generated else None,
            "survival_rate_from_valid": values["survived"] / valid if valid else None,
            "best_fitness": (
                (min(best_values) if objective_direction == "min" else max(best_values))
                if best_values else None
            ),
        })
    return rows


def aggregate_sources(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "cut_selected", "base_attempted", "base_valid",
        "variation_valid_attributed", "winner_attributed",
    )
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {key: 0 for key in keys})
    for record in records:
        item = totals[str(record.get("source_role"))]
        for key in keys:
            item[key] += int(record.get(key) or 0)
    return [
        {"source_role": role, **values}
        for role, values in sorted(totals.items())
    ]


def aggregate_stages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    durations: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.get("event") == "stage_finished" and record.get("stage"):
            durations[str(record["stage"])].append(float(record["duration_seconds"]))
    return [
        {
            "stage": stage,
            "count": len(values),
            "total_seconds": sum(values),
            "mean_seconds": statistics.fmean(values),
            "max_seconds": max(values),
        }
        for stage, values in sorted(durations.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "reports").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(run_dir / "config.json") or {}
    metadata = read_json(run_dir / "metadata.json")
    generations = read_jsonl(run_dir / "metrics" / "generation_metrics.jsonl")
    agents = read_jsonl(run_dir / "metrics" / "agent_metrics.jsonl")
    operators = read_jsonl(run_dir / "metrics" / "operator_metrics.jsonl")
    sources = read_jsonl(run_dir / "metrics" / "experience_source_metrics.jsonl")
    stages = read_jsonl(run_dir / "telemetry" / "stage_timings.jsonl")
    events = read_jsonl(run_dir / "telemetry" / "events.jsonl")
    errors = read_jsonl(run_dir / "telemetry" / "errors.jsonl")
    environment = next(
        (record for record in events if record.get("event") == "run_environment"),
        None,
    )
    failed = next(
        (record for record in reversed(errors) if record.get("event") == "run_failed"),
        None,
    )

    operator_summary = aggregate_operators(
        operators, str(config.get("objective_direction", "min"))
    )
    source_summary = aggregate_sources(sources)
    stage_summary = aggregate_stages(stages)
    write_csv(output_dir / "generations.csv", generations)
    write_csv(output_dir / "agents.csv", agents)
    write_csv(output_dir / "operators_by_generation.csv", operators)
    write_csv(output_dir / "operator_summary.csv", operator_summary)
    write_csv(output_dir / "experience_sources_by_generation.csv", sources)
    write_csv(output_dir / "experience_source_summary.csv", source_summary)
    write_csv(output_dir / "stage_summary.csv", stage_summary)

    latest = generations[-1] if generations else None
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "status": "completed" if metadata else "failed" if failed else "running_or_stopped",
        "completed_generations": int(latest.get("generation") or 0) if latest else 0,
        "generation_metric_rows": len(generations),
        "latest_generation": latest.get("generation") if latest else None,
        "best_fitness": latest.get("best_fitness") if latest else None,
        "independent_evaluations": (
            latest.get("independent_evaluations") if latest else None
        ),
        "cache_hits": latest.get("cache_hits") if latest else None,
        "latest_valid_rate": latest.get("valid_rate") if latest else None,
        "config": config,
        "dataset": {
            "path": environment.get("problem_dataset_path") if environment else None,
            "sha256": environment.get("problem_dataset_sha256") if environment else None,
            "instance_count": environment.get("problem_instance_count") if environment else None,
        },
        "final_metadata": metadata,
        "failure": failed,
        "operator_summary": operator_summary,
        "experience_source_summary": source_summary,
        "stage_summary": stage_summary,
    }
    (output_dir / "intermediate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
