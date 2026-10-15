"""Generate reproducible, non-uniform Euclidean TSP instance datasets.

This script deliberately does not solve TSP instances.  It writes exactly
the four top-level fields used by ``TSPAEL64.pkl`` and leaves
``optimal_tour`` and ``cost`` as ``None`` until a trusted exact/approximate
solver is run in a later step.  The generated files therefore must not be
used as MAE/EoH training data yet.

Examples
--------
Generate the recommended 600-instance training set and 100-instance
validation set:

    python generate_custom_tsp.py --split both

Generate a small smoke dataset:

    python generate_custom_tsp.py --split train --train-count 12
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SIZES = (50, 75, 100, 125, 150, 200)
DEFAULT_DISTRIBUTION_WEIGHTS = {
    "uniform": 0.25,
    "cluster": 0.30,
    "ring": 0.15,
    "grid_jitter": 0.15,
    "corridor": 0.15,
}


def reflect_to_unit(values: np.ndarray) -> np.ndarray:
    """Reflect real-valued coordinates into [0, 1] without edge piling."""
    folded = np.mod(values, 2.0)
    return np.where(folded <= 1.0, folded, 2.0 - folded)


def generate_uniform(rng: np.random.Generator, n_nodes: int) -> tuple[np.ndarray, dict[str, Any]]:
    return rng.random((n_nodes, 2)), {}


def generate_cluster(
    rng: np.random.Generator, n_nodes: int
) -> tuple[np.ndarray, dict[str, Any]]:
    n_clusters = int(rng.integers(2, min(8, n_nodes) + 1))
    centers = rng.uniform(0.10, 0.90, size=(n_clusters, 2))
    scales = rng.uniform(0.025, 0.085, size=n_clusters)
    labels = np.arange(n_nodes) % n_clusters
    rng.shuffle(labels)
    coords = centers[labels] + rng.normal(
        0.0, scales[labels, None], size=(n_nodes, 2)
    )
    return reflect_to_unit(coords), {
        "n_clusters": n_clusters,
        "cluster_scales": [float(value) for value in scales],
    }


def generate_ring(
    rng: np.random.Generator, n_nodes: int
) -> tuple[np.ndarray, dict[str, Any]]:
    n_rings = int(rng.choice([1, 1, 1, 2, 3]))
    centers = rng.uniform(0.38, 0.62, size=(n_rings, 2))
    radii = np.sort(rng.uniform(0.14, 0.42, size=n_rings))
    labels = np.arange(n_nodes) % n_rings
    rng.shuffle(labels)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_nodes)
    radial_noise = rng.normal(0.0, rng.uniform(0.008, 0.035), size=n_nodes)
    radius = np.maximum(0.02, radii[labels] + radial_noise)
    coords = centers[labels] + radius[:, None] * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    return reflect_to_unit(coords), {
        "n_rings": n_rings,
        "radii": [float(value) for value in radii],
    }


def generate_grid_jitter(
    rng: np.random.Generator, n_nodes: int
) -> tuple[np.ndarray, dict[str, Any]]:
    side = int(np.ceil(np.sqrt(n_nodes)))
    axis = (np.arange(side, dtype=float) + 0.5) / side
    mesh_x, mesh_y = np.meshgrid(axis, axis, indexing="xy")
    lattice = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))[:n_nodes]
    rng.shuffle(lattice, axis=0)
    jitter = rng.normal(0.0, 0.22 / side, size=(n_nodes, 2))
    return reflect_to_unit(lattice + jitter), {"grid_side": side}


def generate_corridor(
    rng: np.random.Generator, n_nodes: int
) -> tuple[np.ndarray, dict[str, Any]]:
    center = rng.uniform(0.30, 0.70, size=2)
    angle = float(rng.uniform(0.0, np.pi))
    longitudinal = rng.uniform(-0.52, 0.52, size=n_nodes)
    width = float(rng.uniform(0.025, 0.085))
    transverse = rng.normal(0.0, width, size=n_nodes)
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-np.sin(angle), np.cos(angle)])
    coords = center + longitudinal[:, None] * direction + transverse[:, None] * normal
    return reflect_to_unit(coords), {
        "angle_radians": angle,
        "corridor_width": width,
    }


GENERATORS = {
    "uniform": generate_uniform,
    "cluster": generate_cluster,
    "ring": generate_ring,
    "grid_jitter": generate_grid_jitter,
    "corridor": generate_corridor,
}


def distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(difference * difference, axis=-1))
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float64, copy=False)


def allocate_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Allocate exact integer counts using largest remainder rounding."""
    if total < len(weights):
        raise ValueError("total must be at least the number of distributions")
    names = list(weights)
    raw = np.asarray([weights[name] for name in names], dtype=float) * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    return {name: int(count) for name, count in zip(names, counts)}


def balanced_values(values: tuple[int, ...], total: int) -> list[int]:
    """Repeat values as evenly as possible and shuffle them deterministically."""
    result = [values[index % len(values)] for index in range(total)]
    return result


def make_specs(
    split: str,
    count: int,
    sizes: tuple[int, ...],
    seed: int,
    distribution_weights: dict[str, float],
) -> list[dict[str, Any]]:
    distribution_counts = allocate_counts(count, distribution_weights)
    distributions = [
        name for name, amount in distribution_counts.items() for _ in range(amount)
    ]
    size_values = balanced_values(sizes, count)
    schedule_rng = np.random.default_rng(seed)
    schedule_rng.shuffle(distributions)
    schedule_rng.shuffle(size_values)
    specs = []
    for index, (distribution, n_nodes) in enumerate(zip(distributions, size_values)):
        # Keep per-instance seeds readable while retaining independent streams.
        instance_seed = int(seed * 1_000_000 + index)
        specs.append(
            {
                "split": split,
                "index": index,
                "distribution": distribution,
                "n_nodes": int(n_nodes),
                "seed": instance_seed,
            }
        )
    return specs


def generate_dataset(
    split: str,
    count: int,
    sizes: tuple[int, ...],
    seed: int,
    distribution_weights: dict[str, float],
) -> dict[str, Any]:
    specs = make_specs(split, count, sizes, seed, distribution_weights)
    coordinates: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    for spec in specs:
        rng = np.random.default_rng(spec["seed"])
        coordinates_for_instance, parameters = GENERATORS[spec["distribution"]](
            rng, spec["n_nodes"]
        )
        coordinates_for_instance = coordinates_for_instance.astype(np.float64, copy=False)
        coordinates.append(coordinates_for_instance)
        matrices.append(distance_matrix(coordinates_for_instance))
        metadata.append(
            {
                "instance_id": (
                    f"{split}_{spec['index']:04d}_{spec['distribution']}"
                    f"_n{spec['n_nodes']}_seed{spec['seed']}"
                ),
                **spec,
                "parameters": parameters,
                "solved": False,
                "solver": None,
            }
        )

    # Keep this structure identical to the original TSPAEL64.pkl.
    return {
        "coordinate": coordinates,
        "optimal_tour": [None] * count,
        "distance_matrix": matrices,
        "cost": [None] * count,
        "_generation_metadata": metadata,
    }


def write_pickle(path: Path, dataset: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "both"), default="both")
    parser.add_argument("--train-count", type=int, default=600)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES),
        help="possible node counts; default: 50 75 100 125 150 200",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "TrainingData",
    )
    parser.add_argument(
        "--output-name", default=None,
        help="output PKL filename; useful for a single split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(size < 3 for size in args.sizes):
        raise ValueError("all node counts must be at least 3")
    if args.train_count < 5 or args.validation_count < 5:
        raise ValueError("each split needs at least 5 instances")

    jobs = []
    if args.split in ("train", "both"):
        jobs.append(("train", args.train_count, args.seed))
    if args.split in ("validation", "both"):
        jobs.append(("validation", args.validation_count, args.seed + 1))

    for split, count, split_seed in jobs:
        dataset = generate_dataset(
            split, count, tuple(args.sizes), split_seed, DEFAULT_DISTRIBUTION_WEIGHTS
        )
        metadata = dataset.pop("_generation_metadata")
        if args.output_name is not None:
            if len(jobs) != 1:
                raise ValueError("--output-name can only be used for one split")
            filename = args.output_name
        else:
            filename = f"custom_{split}_seed{split_seed}.pkl"
        output_path = args.output_dir / filename
        write_pickle(output_path, dataset)
        print(f"{split}: {count} instances -> {output_path}")
        print(f"fields: {list(dataset)}")
        print(f"distribution counts: {dict(Counter(item['distribution'] for item in metadata))}")
        print(f"node counts: {dict(Counter(item['n_nodes'] for item in metadata))}")
        print("optimal_tour/cost: None (not solved yet)")


if __name__ == "__main__":
    main()
