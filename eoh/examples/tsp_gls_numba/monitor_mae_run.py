"""Periodically snapshot a running MA-EoH experiment into structured JSONL."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def read_last_jsonl(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            size = min(end, 256 * 1024)
            handle.seek(end - size)
            lines = handle.read(size).splitlines()
        for raw in reversed(lines):
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
    except OSError:
        pass
    return None


def line_count(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def linux_process_metrics(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    status_path = Path(f"/proc/{pid}/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                result["resident_memory_kib"] = int(line.split()[1])
            elif line.startswith("VmSize:"):
                result["virtual_memory_kib"] = int(line.split()[1])
            elif line.startswith("Threads:"):
                result["threads"] = int(line.split()[1])
    except (FileNotFoundError, OSError, ValueError):
        pass
    try:
        result["system_load_average"] = list(os.getloadavg())
    except (AttributeError, OSError):
        pass
    return result


def directory_size(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except OSError:
        pass
    return total


def build_snapshot(run_dir: Path, pid: int | None) -> dict[str, Any]:
    alive = process_alive(pid)
    metrics = read_last_jsonl(run_dir / "metrics" / "generation_metrics.jsonl")
    event = read_last_jsonl(run_dir / "telemetry" / "events.jsonl")
    metadata = read_json(run_dir / "metadata.json")
    error = read_last_jsonl(run_dir / "telemetry" / "errors.jsonl")
    checkpoint_paths = sorted((run_dir / "checkpoints").glob("generation_*.json"))

    if metadata:
        status = "completed"
    elif error and error.get("event") == "run_failed":
        status = "failed"
    elif alive:
        status = "running"
    else:
        status = "stopped_without_summary"

    disk = shutil.disk_usage(run_dir.resolve().anchor or run_dir)
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "pid": pid,
        "process_alive": alive,
        "generation": metrics.get("generation") if metrics else None,
        "best_fitness": metrics.get("best_fitness") if metrics else None,
        "population_mean": metrics.get("population_mean") if metrics else None,
        "population_std": metrics.get("population_std") if metrics else None,
        "generated_candidates": metrics.get("generated_candidates") if metrics else None,
        "valid_candidates": metrics.get("valid_candidates") if metrics else None,
        "valid_rate": metrics.get("valid_rate") if metrics else None,
        "independent_evaluations": (
            metrics.get("independent_evaluations") if metrics else None
        ),
        "cache_hits": metrics.get("cache_hits") if metrics else None,
        "archive_size": metrics.get("archive_size") if metrics else None,
        "velocity_entries": metrics.get("velocity_entries") if metrics else None,
        "mean_cut_size": metrics.get("mean_cut_size") if metrics else None,
        "generation_seconds": metrics.get("generation_seconds") if metrics else None,
        "checkpoint_count": len(checkpoint_paths),
        "latest_checkpoint": str(checkpoint_paths[-1]) if checkpoint_paths else None,
        "failure_count": line_count(run_dir / "failures" / "failures.jsonl"),
        "last_event": event.get("event") if event else None,
        "last_event_generation": event.get("generation") if event else None,
        "run_size_bytes": directory_size(run_dir),
        "disk_free_bytes": disk.free,
        "process": linux_process_metrics(pid) if alive and pid is not None else {},
        "final_metadata": metadata,
    }


def append_snapshot(path: Path, latest_path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
    temporary = latest_path.with_suffix(latest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, latest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="training process id; optional for completed runs and one-off snapshots",
    )
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    run_dir = args.run_dir.resolve()
    output = run_dir / "monitor" / "intermediate_snapshots.jsonl"
    latest = run_dir / "monitor" / "latest_snapshot.json"
    while True:
        snapshot = build_snapshot(run_dir, args.pid)
        append_snapshot(output, latest, snapshot)
        print(
            f"{snapshot['timestamp']} status={snapshot['status']} "
            f"generation={snapshot['generation']} best={snapshot['best_fitness']}",
            flush=True,
        )
        if args.once or snapshot["status"] != "running":
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
