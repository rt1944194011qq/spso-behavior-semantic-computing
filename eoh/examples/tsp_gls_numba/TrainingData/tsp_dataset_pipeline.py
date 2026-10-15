"""Generate six TSP distributions and label them with Concorde under WSL.

The PKL schema matches generate_custom_tsp.py / custom_TSP200_seed2026.pkl:
coordinate, optimal_tour, distance_matrix, cost (all lists).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


DISTRIBUTIONS = ("Uniform", "Clustered", "GaussianMixture", "Grid", "Circle", "Urban")
SIZES = (20, 50, 100, 200, 500)
COUNTS = {20: 1000, 50: 1000, 100: 1000, 200: 500, 500: 100}
DEFAULT_ROOT = Path("/mnt/d/spso-behavior-semantic-computing/eoh/examples/tsp_gls_numba/TrainingData/GeneratedTSP")
DEFAULT_CONCORDE = Path("/mnt/d/tsp求解器/concorde")
SCHEMA_KEYS = ("coordinate", "optimal_tour", "distance_matrix", "cost")
DATA_PATTERN = re.compile(r"^TSP(\d{3})_([A-Za-z]+)_N(\d{4})_seed(\d+)\.pkl$")


def reflect_unit(values: np.ndarray) -> np.ndarray:
    """Use the first generator's boundary rule instead of clipping points."""
    folded = np.mod(values, 2.0)
    return np.where(folded <= 1.0, folded, 2.0 - folded)


def generate_points(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "Uniform":
        points = rng.random((n, 2))
    elif kind == "Clustered":
        k = int(rng.integers(2, min(8, n) + 1))
        centers = rng.uniform(0.10, 0.90, size=(k, 2))
        scales = rng.uniform(0.025, 0.085, size=k)
        labels = np.arange(n) % k
        rng.shuffle(labels)
        points = centers[labels] + rng.normal(0, scales[labels, None], size=(n, 2))
    elif kind == "GaussianMixture":
        k = int(rng.integers(2, min(6, n) + 1))
        centers = rng.uniform(0.12, 0.88, size=(k, 2))
        weights = rng.dirichlet(np.ones(k))
        labels = rng.choice(k, n, p=weights)
        angles = rng.uniform(0, 2 * np.pi, size=k)
        major = rng.uniform(0.055, 0.16, size=k)
        minor = rng.uniform(0.012, 0.055, size=k)
        noise = rng.normal(size=(n, 2))
        x = noise[:, 0] * major[labels]
        y = noise[:, 1] * minor[labels]
        c, s = np.cos(angles[labels]), np.sin(angles[labels])
        points = centers[labels] + np.column_stack((c * x - s * y, s * x + c * y))
    elif kind == "Grid":
        side = math.ceil(math.sqrt(n))
        axis = (np.arange(side, dtype=np.float64) + 0.5) / side
        gx, gy = np.meshgrid(axis, axis, indexing="xy")
        lattice = np.column_stack((gx.ravel(), gy.ravel()))[:n].copy()
        rng.shuffle(lattice, axis=0)
        points = lattice + rng.normal(0, 0.22 / side, size=(n, 2))
    elif kind == "Circle":
        rings = int(rng.choice([1, 1, 1, 2, 3]))
        centers = rng.uniform(0.38, 0.62, size=(rings, 2))
        radii = np.sort(rng.uniform(0.14, 0.42, size=rings))
        labels = np.arange(n) % rings
        rng.shuffle(labels)
        angles = rng.uniform(0, 2 * np.pi, size=n)
        radius = np.maximum(0.02, radii[labels] + rng.normal(0, rng.uniform(0.008, 0.035), size=n))
        points = centers[labels] + radius[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    elif kind == "Urban":
        # Dense downtown grid plus lower-density satellite neighborhoods.
        k = int(rng.integers(3, 7))
        centers = np.vstack((rng.uniform(0.40, 0.60, size=(1, 2)), rng.uniform(0.10, 0.90, size=(k - 1, 2))))
        weights = rng.dirichlet(np.array([5.0] + [1.0] * (k - 1)))
        labels = rng.choice(k, n, p=weights)
        spread = np.where(labels == 0, 0.17, 0.07)
        raw = centers[labels] + rng.normal(size=(n, 2)) * spread[:, None]
        spacing = rng.uniform(0.035, 0.065)
        points = np.round(raw / spacing) * spacing + rng.normal(0, spacing * 0.09, size=(n, 2))
    else:
        raise ValueError(f"Unknown distribution: {kind}")
    return reflect_unit(points).astype(np.float64, copy=False)


def distance_matrix(points: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    matrix = np.sqrt(np.sum(delta * delta, axis=-1))
    np.fill_diagonal(matrix, 0.0)
    return matrix.astype(np.float64, copy=False)


def dataset_name(n: int, kind: str, count: int, seed: int) -> str:
    return f"TSP{n:03d}_{kind}_N{count:04d}_seed{seed}.pkl"


def atomic_pickle(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def validate_schema(data: dict, n: int, count: int) -> None:
    if not isinstance(data, dict) or tuple(data) != SCHEMA_KEYS:
        raise ValueError(f"PKL fields must be {SCHEMA_KEYS}")
    if any(not isinstance(data[key], list) or len(data[key]) != count for key in SCHEMA_KEYS):
        raise ValueError("PKL field lengths do not match file name")
    for i, (points, dist) in enumerate(zip(data["coordinate"], data["distance_matrix"])):
        if not isinstance(points, np.ndarray) or points.shape != (n, 2) or points.dtype != np.float64:
            raise ValueError(f"coordinate[{i}] shape or dtype is wrong")
        if not isinstance(dist, np.ndarray) or dist.shape != (n, n) or dist.dtype != np.float64:
            raise ValueError(f"distance_matrix[{i}] shape or dtype is wrong")


def generate(args: argparse.Namespace) -> None:
    args.root.mkdir(parents=True, exist_ok=True)
    for n in args.sizes:
        count = args.count if args.count is not None else (args.tsp500_count if n == 500 else COUNTS[n])
        if count <= 0 or count > 9999:
            raise ValueError("Count must be between 1 and 9999")
        for kind in args.distributions:
            path = args.root / dataset_name(n, kind, count, args.seed)
            if path.exists() and not args.force:
                print(f"SKIP {path.name} (already exists)", flush=True)
                continue
            # Independent streams keep existing files identical if other configurations are added.
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, n, DISTRIBUTIONS.index(kind)]))
            data = {key: [] for key in SCHEMA_KEYS}
            for _ in range(count):
                points = generate_points(kind, n, rng)
                data["coordinate"].append(points)
                data["optimal_tour"].append(None)
                data["distance_matrix"].append(distance_matrix(points))
                data["cost"].append(None)
            atomic_pickle(path, data)
            print(f"GENERATED {path.name}: {count} instances", flush=True)


def selected_files(args: argparse.Namespace) -> list[tuple[Path, int, str, int, int]]:
    result = []
    for path in sorted(args.root.glob("TSP*.pkl")):
        match = DATA_PATTERN.fullmatch(path.name)
        if not match:
            continue
        n, kind, count, seed = int(match[1]), match[2], int(match[3]), int(match[4])
        if n in args.sizes and kind in args.distributions:
            result.append((path, n, kind, count, seed))
    return result


def load_data(path: Path, n: int, count: int) -> dict:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    validate_schema(data, n, count)
    return data


def progress_path(raw_path: Path) -> Path:
    return raw_path.with_name(raw_path.stem + "_progress.sqlite3")


def solved_path(raw_path: Path) -> Path:
    return raw_path.with_name(raw_path.stem + "_solved.pkl")


def connect_progress(raw_path: Path, scale: int) -> sqlite3.Connection:
    path = progress_path(raw_path)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS solution (idx INTEGER PRIMARY KEY, tour TEXT NOT NULL, cost REAL NOT NULL, integer_cost INTEGER NOT NULL, exit_code INTEGER NOT NULL, elapsed_s REAL NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS failure (idx INTEGER PRIMARY KEY, error TEXT NOT NULL, elapsed_s REAL NOT NULL)")
    stat = raw_path.stat()
    expected = {"raw_size": str(stat.st_size), "raw_mtime_ns": str(stat.st_mtime_ns), "scale": str(scale)}
    stored = dict(connection.execute("SELECT key, value FROM meta"))
    if stored and stored != expected:
        connection.close()
        raise ValueError(f"Raw PKL or scale changed; inspect {path.name} before resuming")
    if not stored:
        connection.executemany("INSERT INTO meta VALUES (?, ?)", expected.items())
        connection.commit()
    return connection


def integer_weights(dist: np.ndarray, scale: int) -> np.ndarray:
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("Distance matrix is not square")
    if not np.isfinite(dist).all() or np.any(dist < 0):
        raise ValueError("Distance matrix contains invalid values")
    if not np.allclose(dist, dist.T, rtol=0, atol=1e-10):
        raise ValueError("Distance matrix is not symmetric")
    if scale <= 0 or float(np.max(dist)) * scale > 2_000_000_000:
        raise ValueError("Scale creates invalid Concorde edge weights")
    weights = np.floor(dist * scale + 0.5).astype(np.int64)
    np.fill_diagonal(weights, 0)
    return weights


def write_tsplib(path: Path, weights: np.ndarray) -> None:
    n = len(weights)
    with path.open("w", encoding="ascii") as handle:
        handle.write(f"NAME: instance\nTYPE: TSP\nDIMENSION: {n}\n")
        handle.write("EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
        for row in weights:
            handle.write(" ".join(map(str, row)) + "\n")
        handle.write("EOF\n")


def solve_one(dist: np.ndarray, concorde: Path, scale: int, timeout: int, idx: int) -> tuple[list[int], float, int, int]:
    n = len(dist)
    weights = integer_weights(dist, scale)
    with tempfile.TemporaryDirectory(prefix=f"tsp_{n}_{idx:04d}_") as folder_name:
        folder = Path(folder_name)
        tsp_file, sol_file = folder / "instance.tsp", folder / "instance.sol"
        write_tsplib(tsp_file, weights)
        result = subprocess.run([str(concorde), "-o", str(sol_file), str(tsp_file)], cwd=folder,
                                capture_output=True, text=True, errors="replace", timeout=timeout, check=False)
        log = result.stdout + "\n" + result.stderr
        match = re.search(r"Optimal Solution:\s*([0-9]+(?:\.[0-9]+)?)", log)
        if not match:
            raise RuntimeError(f"No optimality proof, exit={result.returncode}: {log[-900:]}")
        if not sol_file.exists():
            raise RuntimeError(f"No .sol file, exit={result.returncode}: {log[-900:]}")
        numbers = [int(token) for token in sol_file.read_text(encoding="ascii").split()]
        if len(numbers) != n + 1 or numbers[0] != n:
            raise ValueError("Concorde .sol length/header is invalid")
        tour = numbers[1:]
        if sorted(tour) != list(range(n)):
            raise ValueError("Concorde tour is not a 0-based permutation")
        a = np.asarray(tour, dtype=np.int64)
        b = np.roll(a, -1)
        integer_cost = int(weights[a, b].sum())
        if abs(integer_cost - float(match[1])) > 0.1:
            raise ValueError(f"Integer cost mismatch: tour={integer_cost}, log={match[1]}")
        cost = float(dist[a, b].sum())
        return tour, cost, integer_cost, result.returncode


def write_solved_snapshot(raw_path: Path, data: dict, connection: sqlite3.Connection) -> int:
    tours = [None] * len(data["coordinate"])
    costs = [None] * len(tours)
    for idx, tour_json, cost in connection.execute("SELECT idx, tour, cost FROM solution"):
        tours[idx] = json.loads(tour_json)
        costs[idx] = cost
    snapshot = {"coordinate": data["coordinate"], "optimal_tour": tours,
                "distance_matrix": data["distance_matrix"], "cost": costs}
    atomic_pickle(solved_path(raw_path), snapshot)
    return sum(cost is not None for cost in costs)


def solve(args: argparse.Namespace) -> None:
    if not args.concorde.is_file():
        raise FileNotFoundError(f"Concorde not found: {args.concorde}")
    files = selected_files(args)
    if not files:
        raise FileNotFoundError(f"No selected raw TSP datasets in {args.root}")
    attempted = 0
    failures = 0
    for raw_path, n, kind, count, _ in files:
        if args.max_total is not None and attempted >= args.max_total:
            break
        data = load_data(raw_path, n, count)
        connection = connect_progress(raw_path, args.scale)
        since_snapshot = 0
        print(f"SOLVE {raw_path.name}", flush=True)
        try:
            done = {row[0] for row in connection.execute("SELECT idx FROM solution")}
            for idx in range(count):
                if idx in done:
                    continue
                if args.max_total is not None and attempted >= args.max_total:
                    break
                attempted += 1
                started = time.perf_counter()
                try:
                    tour, cost, int_cost, exit_code = solve_one(data["distance_matrix"][idx], args.concorde,
                                                                 args.scale, args.timeout, idx)
                    elapsed = time.perf_counter() - started
                    connection.execute("INSERT INTO solution VALUES (?, ?, ?, ?, ?, ?)",
                                       (idx, json.dumps(tour), cost, int_cost, exit_code, elapsed))
                    connection.execute("DELETE FROM failure WHERE idx = ?", (idx,))
                    connection.commit()
                    since_snapshot += 1
                    print(f"  OK {idx + 1}/{count} integer={int_cost} cost={cost:.10f} exit={exit_code} {elapsed:.1f}s", flush=True)
                    if since_snapshot >= args.checkpoint_every:
                        completed = write_solved_snapshot(raw_path, data, connection)
                        print(f"  snapshot {completed}/{count}", flush=True)
                        since_snapshot = 0
                except Exception as exc:
                    failures += 1
                    connection.execute("INSERT OR REPLACE INTO failure VALUES (?, ?, ?)",
                                       (idx, str(exc)[-2000:], time.perf_counter() - started))
                    connection.commit()
                    print(f"  FAIL {idx + 1}/{count}: {exc}", flush=True)
                    if args.stop_on_error:
                        raise
        finally:
            completed = connection.execute("SELECT COUNT(*) FROM solution").fetchone()[0]
            if completed or solved_path(raw_path).exists():
                completed = write_solved_snapshot(raw_path, data, connection)
            connection.close()
            print(f"  saved {completed}/{count} solutions", flush=True)
    print(f"Finished: attempted={attempted}, failed={failures}", flush=True)


def status(args: argparse.Namespace) -> None:
    for raw_path, n, kind, count, _ in selected_files(args):
        db = progress_path(raw_path)
        if db.exists():
            with sqlite3.connect(db) as connection:
                solved = connection.execute("SELECT COUNT(*) FROM solution").fetchone()[0]
                has_failure_table = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='failure'"
                ).fetchone()
                failed = connection.execute("SELECT COUNT(*) FROM failure").fetchone()[0] if has_failure_table else 0
        else:
            solved = 0
            failed = 0
        print(f"{raw_path.name}: {solved}/{count} solved; failures={failed}; snapshot={solved_path(raw_path).exists()}")


def verify(args: argparse.Namespace) -> None:
    files = selected_files(args)
    if not files:
        raise FileNotFoundError(f"No selected raw TSP datasets in {args.root}")
    for raw_path, n, _, count, _ in files:
        data = load_data(raw_path, n, count)
        for idx, (points, dist) in enumerate(zip(data["coordinate"], data["distance_matrix"])):
            if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
                raise ValueError(f"{raw_path.name}[{idx}]: coordinates outside [0,1]")
            if not np.allclose(dist, dist.T, rtol=0, atol=1e-12) or not np.allclose(np.diag(dist), 0):
                raise ValueError(f"{raw_path.name}[{idx}]: invalid distance matrix")
            if not np.allclose(dist, distance_matrix(points), rtol=0, atol=1e-12):
                raise ValueError(f"{raw_path.name}[{idx}]: distances do not match coordinates")
        output = solved_path(raw_path)
        if not output.exists():
            print(f"VERIFIED raw {raw_path.name}: {count} instances; unsolved")
            continue
        solved = load_data(output, n, count)
        complete = 0
        for idx, (tour, cost) in enumerate(zip(solved["optimal_tour"], solved["cost"])):
            if tour is None and cost is None:
                continue
            if tour is None or cost is None or sorted(tour) != list(range(n)):
                raise ValueError(f"{output.name}[{idx}]: invalid tour/label")
            a = np.asarray(tour, dtype=np.int64)
            expected = float(data["distance_matrix"][idx][a, np.roll(a, -1)].sum())
            if not math.isclose(float(cost), expected, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"{output.name}[{idx}]: cost mismatch")
            complete += 1
        print(f"VERIFIED {output.name}: {complete}/{count} solutions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "solve", "status", "verify"):
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        p.add_argument("--sizes", nargs="+", type=int, choices=SIZES, default=list(SIZES))
        p.add_argument("--distributions", nargs="+", choices=DISTRIBUTIONS, default=list(DISTRIBUTIONS))
        if command == "generate":
            p.add_argument("--seed", type=int, default=2026)
            p.add_argument("--count", type=int, help="Override count for each selected size (small test)")
            p.add_argument("--tsp500-count", type=int, choices=(100, 200), default=100)
            p.add_argument("--force", action="store_true", help="Replace existing raw PKL")
        if command == "solve":
            p.add_argument("--concorde", type=Path, default=DEFAULT_CONCORDE)
            p.add_argument("--scale", type=int, default=1_000_000)
            p.add_argument("--timeout", type=int, default=3600, help="Seconds per instance")
            p.add_argument("--max-total", type=int, help="Maximum new attempts across selected files")
            p.add_argument("--checkpoint-every", type=int, default=25)
            behavior = p.add_mutually_exclusive_group()
            behavior.add_argument("--stop-on-error", dest="stop_on_error", action="store_true", default=True)
            behavior.add_argument("--continue-on-error", dest="stop_on_error", action="store_false")
    args = parser.parse_args()
    if args.command == "solve" and (args.checkpoint_every < 1 or args.timeout < 1 or
                                    (args.max_total is not None and args.max_total < 1)):
        parser.error("checkpoint-every, timeout and max-total must be positive")
    return args


def main() -> None:
    args = parse_args()
    {"generate": generate, "solve": solve, "status": status, "verify": verify}[args.command](args)


if __name__ == "__main__":
    main()
