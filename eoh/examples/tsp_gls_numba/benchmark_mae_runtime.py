"""Click Run to measure complete evaluations; no LLM or training is started.

Edit the constants below to repeat the same checks on the Linux server.
Every case runs in an isolated process with a hard timeout. Results are
saved after each case, including early Numba/performance rejections.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "../../eoh/src"))

from eoh.eoh.evolution import _eval_with_timeout
from heuristic_runtime import prepare_heuristic
from mae_tsp_adapter import TracedTSPGLS
from prob import TSPGLS

# --------------------- Editable benchmark configuration ---------------------
DATASET_64 = HERE / "TrainingData/custom_TSP200_seed2026_solved.pkl"  # first 64 x TSP200
DATASET_200 = HERE / "TrainingData/custom_TSP200_seed2026_solved.pkl"
REFERENCE_CODE = HERE / "author_run_bests/run3/heuristic.py"
EOH_TRAINED_BEST = PROJECT / "tsp_baselines/frozen/EOH_USER_BEST.py"
SLOW_RUN_ARCHIVE = Path("D:/mae_tsp200_online_seed2026_20261015_211404_926753/archive/programs.jsonl")
FULL_200_CHECK = False          # one complete 64-instance pass is the default
ITE_MAX = 1000
TIME_LIMIT = 60.0
CASE_TIMEOUT = 900
CALL_BUDGET = 0.0
OUTPUT_DIR = PROJECT / "reports" / ("mae_async_numba_" + datetime.now().strftime("%Y%m%d_%H%M%S"))


def microbenchmark():
    """Same numeric loop and inputs, Python versus native code, no GLS work."""
    source = '''def update_edge_distance(distance, tour, history):
    result = distance.copy()
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            result[i, j] += 0.02 * distance[i, j] / (1.0 + history[i, j])
    return result
'''
    modules = [types.ModuleType("micro_python"), types.ModuleType("micro_numba")]
    for module in modules:
        exec(source, module.__dict__)
    rng = np.random.default_rng(2026)
    coords = rng.random((200, 2))
    distance = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    tour, history = np.arange(200, dtype=np.int64), np.zeros_like(distance)
    compiled, runtime = prepare_heuristic(
        modules[1].update_edge_distance, distance, max_call_seconds=0,
    )
    np.testing.assert_array_equal(
        modules[0].update_edge_distance(distance, tour, history),
        compiled(distance, tour, history),
    )
    timings = {}
    for name, function, repeats in (
        ("python", modules[0].update_edge_distance, 20), ("numba", compiled, 500),
    ):
        started = time.perf_counter()
        for _ in range(repeats):
            function(distance, tour, history)
        timings[name + "_seconds_per_call"] = (time.perf_counter() - started) / repeats
    timings["speedup"] = timings["python_seconds_per_call"] / timings["numba_seconds_per_call"]
    timings["output_equal"] = True
    timings["runtime"] = runtime
    return timings


def evaluate_case(label, code, dataset, count, *, traced, numba, iterations=ITE_MAX):
    klass = TracedTSPGLS if traced else TSPGLS
    problem = klass(
        n_inst_eva=count, datafile=dataset, time_limit=TIME_LIMIT, ite_max=iterations,
        timeout=CASE_TIMEOUT, numba_heuristic=numba,
        max_heuristic_call_seconds=CALL_BUDGET if numba else 0,
        require_numba=False,
    )
    started = time.perf_counter()
    raw = _eval_with_timeout(problem, code, CASE_TIMEOUT)
    elapsed = time.perf_counter() - started
    details = raw.get("details", {}) if isinstance(raw, dict) else {}
    fitness = raw.get("fitness") if isinstance(raw, dict) else raw
    instances = details.get("instances", [])
    return {
        "label": label, "dataset": str(dataset), "requested_instances": count,
        "node_count": len(problem.instances[0]), "iteration_limit": iterations,
        "time_limit": TIME_LIMIT, "execution": "numba" if numba else "python",
        "heuristic_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "wall_seconds": elapsed, "fitness": fitness, "valid": fitness is not None,
        "completed_instances": len(instances) if traced else count if fitness is not None else 0,
        "iteration_limit_reached": sum(x["iterations"] == iterations for x in instances),
        "details": details,
    }


def save(report):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    lines = ["# MAE asynchronous / Numba runtime verification", "",
             "All evaluations below are measured on this local machine, not inferred server times.",
             "The archived EoH training used 200 instances and a 5400 s total timeout; it has no per-code 64-instance timing.",
             "Wall time includes subprocess startup and candidate compilation; all cases use the same GLS core.", "",
             "| Case | Instances x cities | Iterations cap | Wall seconds | Fitness (%) | Status |",
             "|---|---:|---:|---:|---:|---|"]
    for case in report["cases"]:
        failure = case["details"].get("failure", {})
        status = "complete" if case["valid"] else failure.get("stage", "timeout/error")
        lines.append(f'| {case["label"]} | {case["requested_instances"]} x {case["node_count"]} '
                     f'| {case["iteration_limit"]} | {case["wall_seconds"]:.3f} '
                     f'| {case["fitness"]} | {status} |')
    micro = report.get("callback_microbenchmark")
    if micro:
        lines += ["", f'Numeric-loop microbenchmark (not end-to-end GLS): {micro["speedup"]:.2f}x, equal output.',
                  f'Python {micro["python_seconds_per_call"]:.6f}s/call; Numba {micro["numba_seconds_per_call"]:.6f}s/call.']
    lines += ["", "The asynchronous ordering regression uses events: a fast agent reaches H6 before a deliberately blocked agent finishes H3.",
              "A numerical callback budget changes the admissible search space. Use the same runtime contract for fair method comparisons."]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    report = {"python": sys.version, "platform": platform.platform(),
              "cpu_count": os.cpu_count(), "cases": []}
    code = REFERENCE_CODE.read_text(encoding="utf-8")
    cases = []
    if EOH_TRAINED_BEST.is_file():
        cases.append(("trained_eoh_best_python_64", EOH_TRAINED_BEST.read_text(encoding="utf-8"),
                      DATASET_64, 64, False, False, ITE_MAX))
    cases += [
        ("reference_eoh_python_64", code, DATASET_64, 64, False, False, ITE_MAX),
        ("same_reference_mae_numba_64", code, DATASET_64, 64, True, True, ITE_MAX),
    ]
    if FULL_200_CHECK:
        cases.append(("same_reference_mae_numba_full200", code, DATASET_200, 200, True, True, ITE_MAX))
    if SLOW_RUN_ARCHIVE.is_file():
        targets = {"ind-00000141", "ind-00000148"}
        with SLOW_RUN_ARCHIVE.open(encoding="utf-8") as handle:
            for line in handle:
                program = json.loads(line)
                if program["individual_id"] in targets:
                    cases.append((program["individual_id"] + "_optional_numba_one_instance",
                                  program["code"], DATASET_200, 1, True, True, 50))
    for label, source, dataset, count, traced, numba, iterations in cases:
        print(f"START {label}: {count} instances", flush=True)
        case = evaluate_case(label, source, dataset, count, traced=traced,
                             numba=numba, iterations=iterations)
        report["cases"].append(case)
        save(report)
        failure = case["details"].get("failure", {})
        print(f"END {label}: {case['wall_seconds']:.3f}s, fitness={case['fitness']}, "
              f"status={failure.get('stage', 'complete' if case['valid'] else 'timeout/error')}", flush=True)
    report["callback_microbenchmark"] = microbenchmark()
    save(report)
    print(f"Results: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
