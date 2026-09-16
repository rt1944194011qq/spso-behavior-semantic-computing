"""Route evidence captured during the same evaluation that supplies fitness."""

import math
import time

import numpy as np

from prob import TSPGLS, guided_local_search, nearest_neighbor_2End, tour_cost_2End


def verify_route(matrix, route, reported_cost):
    """Validate one Hamiltonian cycle and independently sum its original edges."""
    n = len(matrix)
    route = np.asarray(route)
    if n < 3 or route.shape != (n, 2) or not np.issubdtype(route.dtype, np.integer):
        raise ValueError("invalid two-end route shape or type")
    if np.any(route < 0) or np.any(route >= n):
        raise ValueError("route node outside instance")
    nodes, visited, node = [], set(), 0
    for _ in range(n):
        if node in visited:
            raise ValueError("route revisits a node before visiting all cities")
        visited.add(node)
        nodes.append(node)
        nxt = int(route[node, 1])
        if int(route[nxt, 0]) != node:
            raise ValueError("route predecessor/successor mismatch")
        node = nxt
    if node != 0 or len(visited) != n:
        raise ValueError("route is not a Hamiltonian cycle")
    closed = nodes + [0]
    lengths = [float(matrix[u, v]) for u, v in zip(closed, closed[1:])]
    length = math.fsum(lengths)
    if not math.isfinite(length) or not math.isclose(length, reported_cost, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"route length {length} disagrees with solver cost {reported_cost}")
    return {"tour_nodes": nodes, "closed_tour": closed, "edge_lengths": lengths,
            "tour_length": length, "solver_best_cost": float(reported_cost),
            "cost_check_error": length - float(reported_cost), "valid_tour": True}


class TSPGLSWithDetails(TSPGLS):
    """S-PSO adapter; original TSPGLS scalar interface remains available to EoH."""

    def evaluate_program(self, program_str, callable_func):
        details = []
        for index, (matrix, optimum) in enumerate(zip(self.instances, self.opt_costs), 1):
            started = time.time()
            original = matrix.copy()
            initial = nearest_neighbor_2End(matrix, 0).astype(int)
            initial_cost = tour_cost_2End(matrix, initial)
            neighbours = np.argsort(matrix, axis=1)[:, 1:101].astype(int)
            route, cost, iterations = guided_local_search(
                matrix, neighbours, initial, initial_cost,
                started + self.time_limit, self.ite_max, self.perturbation_moves,
                callable_func, capture_best_route=True,
            )
            if not np.array_equal(original, matrix):
                raise ValueError("heuristic mutated the original distance matrix")
            evidence = verify_route(original, route, cost)
            evidence.update({
                "instance": index, "node_count": len(matrix),
                "reference_length": float(optimum),
                "gap_percent": float((cost / optimum - 1) * 100),
                "recomputed_gap_percent": float((evidence['tour_length'] / optimum - 1) * 100),
                "iterations": int(iterations), "elapsed_seconds": time.time() - started,
                "stop_reason": "iteration_limit" if iterations >= self.ite_max else "time_limit",
            })
            details.append(evidence)
        if len(details) != self.n_inst_eva or not details:
            raise ValueError("training instance count mismatch")
        return {"objective": float(np.mean([d['gap_percent'] for d in details])),
                "details": {"instances": details, "node_index_base": 0,
                            "fitness_definition": "mean((solver_best_cost/reference_length-1)*100)"}}


def print_route_report(result, emit=print):
    """Print all training routes produced by one heuristic (one per instance)."""
    emit(f"Fitness (mean gap): {result['objective']:.12g}%")
    for item in result.get('details', {}).get('instances', []):
        emit(f"instance={item['instance']} length={item['tour_length']:.15g} "
             f"reference={item['reference_length']:.15g} gap={item['gap_percent']:.12g}% "
             f"valid={item['valid_tour']} iterations={item['iterations']}")
        emit("closed tour (0-based): " + " -> ".join(map(str, item['closed_tour'])))
