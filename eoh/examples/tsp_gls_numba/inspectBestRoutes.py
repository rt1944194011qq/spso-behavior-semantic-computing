"""Inspect a saved heuristic without an LLM; edit constants, then run this file."""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "eoh" / "src"))

from eoh.eoh.evolution import _eval_with_timeout
from spso_task.reporting import TSPGLSWithDetails, print_route_report

BEST_FILE = HERE / "results_spso" / "run_001" / "samples" / "best.json"
TRAIN_INSTANCES = 3
GLS_TIME_LIMIT = 60.0
GLS_ITE_MAX = 1000
PERTURBATION_MOVES = 1
TASK_TIMEOUT = 240


def main():
    best = json.loads(BEST_FILE.read_text(encoding="utf-8"))
    task = TSPGLSWithDetails(n_inst_eva=TRAIN_INSTANCES, time_limit=GLS_TIME_LIMIT,
                           ite_max=GLS_ITE_MAX, perturbation_moves=PERTURBATION_MOVES,
                           timeout=TASK_TIMEOUT)
    result = _eval_with_timeout(task, best['code'], TASK_TIMEOUT)
    if not isinstance(result, dict):
        raise RuntimeError("Route evaluation failed or timed out; no verified result produced")
    result.update({"stored_objective": best['objective'], "reevaluation": True,
                   "source_file": str(BEST_FILE),
                   "code_sha256": hashlib.sha256(best['code'].encode()).hexdigest(),
                   "settings": {"train_instances": TRAIN_INSTANCES, "time_limit": GLS_TIME_LIMIT,
                                "ite_max": GLS_ITE_MAX, "perturbation_moves": PERTURBATION_MOVES}})
    output = BEST_FILE.parent / f"best_routes_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Stored fitness: {best['objective']}; fresh evaluation follows", flush=True)
    print_route_report(result)
    print(f"Route evidence: {output}")


if __name__ == '__main__':
    main()
