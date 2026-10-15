"""Validate a solved four-field TSP training dataset and write a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "TrainingData" / "custom_TSP200_seed2026_solved.pkl"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path) -> dict:
    with path.open("rb") as handle:
        data = pickle.load(handle)

    required = ("coordinate", "optimal_tour", "distance_matrix", "cost")
    if set(data) != set(required):
        raise ValueError(
            f"expected exactly {list(required)}, found {sorted(data)}"
        )
    lengths = {key: len(data[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"field lengths differ: {lengths}")

    node_counts: Counter[int] = Counter()
    costs: list[float] = []
    route_cost_errors: list[float] = []
    symmetry_errors: list[float] = []
    coordinate_distance_errors: list[float] = []

    for index, (coords_raw, tour_raw, matrix_raw, cost_raw) in enumerate(zip(
        data["coordinate"], data["optimal_tour"],
        data["distance_matrix"], data["cost"],
    )):
        coords = np.asarray(coords_raw, dtype=float)
        matrix = np.asarray(matrix_raw, dtype=float)
        tour = np.asarray(tour_raw, dtype=np.int64).reshape(-1)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"instance {index}: coordinate shape is {coords.shape}")
        node_count = int(coords.shape[0])
        if matrix.shape != (node_count, node_count):
            raise ValueError(f"instance {index}: matrix shape is {matrix.shape}")
        if tour.shape != (node_count,):
            raise ValueError(f"instance {index}: tour shape is {tour.shape}")
        if not np.array_equal(np.sort(tour), np.arange(node_count)):
            raise ValueError(f"instance {index}: tour is not a node permutation")
        if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"instance {index}: coordinates or distances are non-finite")
        if np.any(matrix < 0):
            raise ValueError(f"instance {index}: distance matrix contains negatives")

        cost = float(cost_raw)
        if not np.isfinite(cost) or cost <= 0:
            raise ValueError(f"instance {index}: invalid solved cost {cost_raw!r}")
        route_cost = float(matrix[tour, np.roll(tour, -1)].sum())
        route_error = abs(route_cost - cost)
        tolerance = 1e-9 * max(1.0, abs(cost))
        if route_error > tolerance:
            raise ValueError(
                f"instance {index}: stored cost {cost} != route cost {route_cost}"
            )

        symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
        diagonal_error = float(np.max(np.abs(np.diag(matrix))))
        if symmetry_error > 1e-10 or diagonal_error > 1e-10:
            raise ValueError(
                f"instance {index}: invalid distance matrix "
                f"(symmetry={symmetry_error}, diagonal={diagonal_error})"
            )
        diff = coords[:, None, :] - coords[None, :, :]
        reconstructed = np.sqrt(np.sum(diff * diff, axis=-1))
        coordinate_error = float(np.max(np.abs(reconstructed - matrix)))
        if coordinate_error > 1e-10:
            raise ValueError(
                f"instance {index}: coordinates and distance matrix differ by "
                f"{coordinate_error}"
            )

        node_counts[node_count] += 1
        costs.append(cost)
        route_cost_errors.append(route_error)
        symmetry_errors.append(symmetry_error)
        coordinate_distance_errors.append(coordinate_error)

    return {
        "dataset_path": str(path.resolve()),
        "sha256": file_sha256(path),
        "file_size_bytes": path.stat().st_size,
        "instance_count": len(costs),
        "field_lengths": lengths,
        "node_count_distribution": dict(sorted(node_counts.items())),
        "solved_instances": len(costs),
        "cost": {
            "min": min(costs),
            "mean": statistics.fmean(costs),
            "median": statistics.median(costs),
            "max": max(costs),
        },
        "max_route_cost_error": max(route_cost_errors, default=0.0),
        "max_symmetry_error": max(symmetry_errors, default=0.0),
        "max_coordinate_distance_error": max(
            coordinate_distance_errors, default=0.0
        ),
        "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = validate(args.dataset.resolve())
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"manifest: {args.output.resolve()}")
    print(rendered)


if __name__ == "__main__":
    main()
