from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path


def tail_jsonl(path: Path, count: int) -> list[dict]:
    if not path.is_file():
        return []
    records = deque(maxlen=count)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the latest state of a running or completed MA-EoH experiment."
    )
    parser.add_argument("run_dir", help="Path to results_mae/<run_name>")
    parser.add_argument("--tail", type=int, default=8, help="Number of recent events to show")
    args = parser.parse_args()

    root = Path(args.run_dir).resolve()
    events = tail_jsonl(root / "telemetry" / "events.jsonl", max(1, args.tail))
    progress = tail_jsonl(root / "telemetry" / "progress.jsonl", 1)
    generations = tail_jsonl(root / "metrics" / "generation_metrics.jsonl", 1)
    failures = tail_jsonl(root / "failures" / "failures.jsonl", 100000)

    if not root.is_dir():
        raise FileNotFoundError(root)
    latest = events[-1] if events else None
    completed = (root / "metadata.json").is_file()
    failed = bool(latest and latest.get("event") == "run_failed")
    status = "FAILED" if failed else "FINISHED" if completed else "RUNNING/INCOMPLETE"

    print(f"run: {root}")
    print(f"status: {status}")
    if latest:
        print(
            "latest: "
            f"{latest.get('timestamp')} event={latest.get('event')} "
            f"generation={latest.get('generation')} "
            f"elapsed={latest.get('elapsed_seconds', 0):.1f}s"
        )
    if progress:
        item = progress[-1]
        print(
            "progress: "
            f"event={item.get('event')} phase={item.get('phase')} "
            f"completed={item.get('completed')} total={item.get('total')} "
            f"pending={item.get('pending')}"
        )
    if generations:
        item = generations[-1]
        print(
            "generation metrics: "
            f"generation={item.get('generation')} best={item.get('best_fitness')} "
            f"valid={item.get('valid_candidates')}/{item.get('generated_candidates')} "
            f"evals={item.get('independent_evaluations')} cache={item.get('cache_hits')}"
        )
    if failures:
        by_stage = Counter(item.get("failure_stage", "unknown") for item in failures)
        by_path = Counter(item.get("operator_path", "unknown") for item in failures)
        print(f"failures: total={len(failures)} by_stage={dict(by_stage)}")
        print(f"failure paths: {dict(by_path)}")

    print("recent events:")
    for item in events:
        print(
            f"  #{item.get('sequence')} {item.get('timestamp')} "
            f"g={item.get('generation')} a={item.get('agent_id')} "
            f"{item.get('event')}"
        )


if __name__ == "__main__":
    main()
