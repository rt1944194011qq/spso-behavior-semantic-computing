"""Evaluate the best S-PSO source on the held-out TSPLIB protocol."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "evaluation"))
from runEval import BKS, TIME_LIMIT, ITE_MAX, _TSPLIB_DIR, instance_size, read_tsplib  # noqa: E402
from prob import solve_instance  # noqa: E402


def load_code(path: Path):
    source = path.read_text(encoding="utf-8")
    namespace = {"np": np}
    exec(source, namespace)
    return namespace["update_edge_distance"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", default=str(HERE / "results_spso" / "samples" / "best.json"))
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    import json

    record = json.loads(Path(args.best).read_text(encoding="utf-8"))
    source_path = Path(args.best).with_name("best_generated.py")
    source_path.write_text(record["code"], encoding="utf-8")
    update_fn = load_code(source_path)
    results = []
    for name in sorted(BKS, key=instance_size):
        matrix, scale = read_tsplib(os.path.join(_TSPLIB_DIR, f"{name}.tsp"))
        gap, iterations = solve_instance(
            BKS[name] / scale, matrix, TIME_LIMIT, ITE_MAX, 1, update_fn
        )
        results.append((name, matrix.shape[0], gap, iterations))
        print(f"{name:<10s} n={matrix.shape[0]:>3d} gap={gap:8.4f}% iters={iterations:>4d}/{ITE_MAX}")
    print(f"average gap: {np.mean([item[2] for item in results]):.4f}%")


if __name__ == "__main__":
    main()
