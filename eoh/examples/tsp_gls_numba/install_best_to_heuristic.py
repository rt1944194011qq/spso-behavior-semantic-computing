"""Install the best S-PSO program into the original EoH test entrypoint.

Edit ``BEST_FILE`` below, run this script once, then execute the unchanged
``evaluation/runEval.py`` exactly as in the original EoH workflow.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent

# ── test configuration ───────────────────────────────────────────────────────
BEST_FILE = HERE / "results_spso" / "run_001" / "samples" / "best.json"
TARGET_HEURISTIC = HERE / "evaluation" / "heuristic.py"
BACKUP_FILE = BEST_FILE.parent / "heuristic_baseline.py"


def main():
    if not BEST_FILE.is_file():
        raise FileNotFoundError(f"best file not found: {BEST_FILE}")
    if not TARGET_HEURISTIC.is_file():
        raise FileNotFoundError(f"target heuristic file not found: {TARGET_HEURISTIC}")

    record = json.loads(BEST_FILE.read_text(encoding="utf-8"))
    source = record.get("code")
    if not isinstance(source, str) or "def update_edge_distance" not in source:
        raise ValueError("best.json does not contain a valid update_edge_distance program")

    # Validate syntax and the required public entrypoint before changing the
    # original EoH evaluation file.
    namespace = {}
    compile(source, str(BEST_FILE), "exec")
    exec(compile(source, str(BEST_FILE), "exec"), namespace)
    if not callable(namespace.get("update_edge_distance")):
        raise ValueError("best program does not define update_edge_distance")

    if not BACKUP_FILE.exists():
        shutil.copyfile(TARGET_HEURISTIC, BACKUP_FILE)

    TARGET_HEURISTIC.write_text(source.rstrip() + "\n", encoding="utf-8")
    print(f"Installed best program: {BEST_FILE}")
    print(f"Target: {TARGET_HEURISTIC}")
    print(f"Baseline backup: {BACKUP_FILE}")
    print("Now run evaluation/runEval.py to evaluate this heuristic on TSPLIB.")


if __name__ == "__main__":
    main()
