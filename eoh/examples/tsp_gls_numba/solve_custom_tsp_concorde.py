"""Solve generated TSP instances with Concorde and fill the four-field PKL.

The input PKL must contain the same four fields as TSPAEL64.pkl:
``coordinate``, ``optimal_tour``, ``distance_matrix`` and ``cost``.
Coordinates are scaled to integer TSPLIB coordinates before Concorde is
called.  The returned route length is recalculated on the original floating
point distance matrix so it remains compatible with this project's GLS code.

Concorde writes tours with zero-based node ids.  The script checkpoints after
each solved instance, so an interrupted run can be started again with the same
input and output paths.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = ("coordinate", "optimal_tour", "distance_matrix", "cost")


def find_concorde(path: Path) -> Path:
    """Accept either a Concorde executable or the downloaded source folder."""
    if path.is_file():
        return path
    candidates = (
        path / "concorde.exe",
        path / "TSP" / "concorde.exe",
        path / "TSP" / "concorde",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No Concorde executable found. The downloaded directory is source code; "
        "compile it first or pass the path to concorde.exe. Checked: "
        + ", ".join(str(item) for item in candidates)
    )


def write_tsplib(path: Path, coordinates: np.ndarray, scale: int) -> None:
    integer_coordinates = np.rint(coordinates * scale).astype(np.int64)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"NAME: {path.stem}\n")
        handle.write("TYPE: TSP\n")
        handle.write("COMMENT: generated Euclidean instance\n")
        handle.write(f"DIMENSION: {len(coordinates)}\n")
        handle.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        handle.write("NODE_COORD_TYPE: TWOD_COORDS\n")
        handle.write("NODE_COORD_SECTION\n")
        for node, (x_coord, y_coord) in enumerate(integer_coordinates, start=1):
            handle.write(f"{node} {int(x_coord)} {int(y_coord)}\n")
        handle.write("EOF\n")


def read_concorde_tour(path: Path, n_nodes: int) -> np.ndarray:
    """Read Concorde's text tour format: n followed by n zero-based ids."""
    numbers = [int(token) for token in re.findall(r"[-+]?\d+", path.read_text())]
    if numbers and numbers[0] == n_nodes:
        candidates = numbers[1 : n_nodes + 1]
    else:
        candidates = numbers[-n_nodes:]
    tour = np.asarray(candidates, dtype=np.int32)
    if tour.shape != (n_nodes,) or set(tour.tolist()) != set(range(n_nodes)):
        raise ValueError(f"invalid Concorde tour in {path}")
    return tour


def route_cost(distance_matrix: np.ndarray, tour: np.ndarray) -> float:
    return float(
        np.sum(distance_matrix[tour, np.roll(tour, -1)], dtype=np.float64)
    )


def atomic_pickle(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def solve_dataset(
    concorde: Path,
    input_path: Path,
    output_path: Path,
    scale: int,
    start: int,
    limit: int | None,
) -> None:
    with input_path.open("rb") as handle:
        source = pickle.load(handle)
    if not isinstance(source, dict) or set(source) != set(REQUIRED_FIELDS):
        raise ValueError(f"input must contain exactly {list(REQUIRED_FIELDS)}")

    coordinates = source["coordinate"]
    matrices = source["distance_matrix"]
    tours = list(source["optimal_tour"])
    costs = list(source["cost"])
    if not (len(coordinates) == len(matrices) == len(tours) == len(costs)):
        raise ValueError("dataset fields have different instance counts")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "coordinate": [np.asarray(item) for item in coordinates],
        "optimal_tour": tours,
        "distance_matrix": [np.asarray(item) for item in matrices],
        "cost": costs,
    }
    atomic_pickle(output_path, data)

    end = len(coordinates) if limit is None else min(len(coordinates), start + limit)
    for index in range(start, end):
        if tours[index] is not None and costs[index] is not None:
            print(f"[{index + 1}/{len(coordinates)}] skip: already solved")
            continue
        coords = np.asarray(coordinates[index], dtype=np.float64)
        matrix = np.asarray(matrices[index], dtype=np.float64)
        with tempfile.TemporaryDirectory(prefix=f"concorde_{index:04d}_") as temporary:
            work = Path(temporary)
            tsp_path = work / f"instance_{index:04d}.tsp"
            sol_path = work / f"instance_{index:04d}.sol"
            write_tsplib(tsp_path, coords, scale)
            command = [str(concorde), "-o", str(sol_path), str(tsp_path)]
            print(f"[{index + 1}/{len(coordinates)}] solving n={len(coords)}")
            completed = subprocess.run(
                command,
                cwd=work,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0 or not sol_path.is_file():
                print(completed.stdout[-2000:])
                print(completed.stderr[-4000:])
                raise RuntimeError(
                    f"Concorde failed for instance {index}; return code "
                    f"{completed.returncode}"
                )
            tour = read_concorde_tour(sol_path, len(coords))
            cost = route_cost(matrix, tour)
        tours[index] = tour
        costs[index] = cost
        data["optimal_tour"] = tours
        data["cost"] = costs
        atomic_pickle(output_path, data)
        print(f"  cost={cost:.12f}; checkpoint={output_path}")

    print("Temporary TSPLIB and .sol files were removed after each instance.")
    print(f"Solved {sum(item is not None for item in tours)}/{len(tours)} instances.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concorde", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scale", type=int, default=1_000_000,
        help="integer coordinate scale used to create EUC_2D input (default: 1000000)",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scale <= 0:
        raise ValueError("--scale must be positive")
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    concorde = find_concorde(args.concorde.resolve())
    solve_dataset(
        concorde=concorde,
        input_path=args.input.resolve(),
        output_path=args.output.resolve(),
        scale=args.scale,
        start=args.start,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
