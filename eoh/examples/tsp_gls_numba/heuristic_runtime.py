"""Explicit Numba execution and a measured budget for evolved TSP callbacks."""
from __future__ import annotations

import inspect
import statistics
import time

import numpy as np


RUNTIME_VERSION = "numba-optional-v3"
NUMBA_CONTRACT = (
    "Execution contract: the host attempts to compile your function and local helpers "
    "with numba.njit, never object mode. Prefer Numba-compatible code. An unsupported "
    "Python version is permitted within the total candidate evaluation deadline. "
    "Inputs are float64 distance/history matrices and an int64 tour. Return ordinary "
    "Python function definitions without JIT decorators. Use Numba-supported NumPy "
    "operations or scalar-indexed numeric loops. Use explicit np.float64, np.int64, "
    "np.bool_ dtypes. Avoid np.add.at, two-dimensional boolean indexing, multiple "
    "array indices such as D[u_array, v_array], Python objects, and dynamic imports. "
    "Compute whole-matrix reductions once per call rather than inside tour loops. "
    "Avoid enumerating and sorting all possible edge-pair moves on every GLS call. "
    "Keep every callback efficient: the candidate must finish all evaluation instances "
    "within a hard total deadline. An optional per-callback budget may also be enabled."
)


class HeuristicExecutionError(RuntimeError):
    def __init__(self, stage: str, reason: str, **details):
        super().__init__(reason)
        self.details = {"stage": stage, "reason": reason, **details}


class _BudgetedHeuristic:
    """Check a window of calls, including calls made after GLS history grows."""

    def __init__(self, function, max_seconds: float, window: int = 32):
        self.function = function
        self.max_seconds = max_seconds
        self.window = window
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, distance, tour, history):
        started = time.perf_counter()
        result = self.function(distance, tour, history)
        self.seconds += time.perf_counter() - started
        self.calls += 1
        if self.calls >= self.window:
            average = self.seconds / self.calls
            if average > self.max_seconds:
                raise HeuristicExecutionError(
                    "heuristic_runtime", "callback exceeded its mean time budget",
                    mean_call_seconds=average, limit_seconds=self.max_seconds,
                    window_calls=self.calls,
                )
            self.calls = 0
            self.seconds = 0.0
        return result


def prepare_heuristic(function, distance, *, use_numba=True, max_call_seconds=0.0,
                      require_numba=False):
    """Compile once per candidate; warm-up is outside each instance's GLS clock.

    No algorithm rewriting, fast-math or parallel loop transformation is used.
    Optional Python fallback is controlled by the candidate's outer hard timeout.
    A positive callback budget adds early rejection. Preflight uses copies and
    cannot change GLS history.
    """
    started = time.perf_counter()
    compiled = function
    compilation_error = None
    numba_version = None
    local_functions = {}
    if use_numba:
        try:
            import numba
            from numba.core.registry import CPUDispatcher
            numba_version = numba.__version__

            if not isinstance(function, CPUDispatcher):
                namespace = function.__globals__
                local_functions = {
                    name: value for name, value in list(namespace.items())
                    if inspect.isfunction(value) and value.__module__ == function.__module__
                }
                dispatchers = {
                    name: numba.njit(
                        cache=False, nogil=True, fastmath=False, error_model="numpy"
                    )(value)
                    for name, value in local_functions.items()
                }
                namespace.update(dispatchers)
                compiled = dispatchers.get(function.__name__)
                if compiled is None:
                    compiled = numba.njit(
                        cache=False, nogil=True, fastmath=False, error_model="numpy"
                    )(function)
            # Trigger all typing/lowering errors before a full evaluation starts.
            compiled(distance.copy(), np.arange(len(distance), dtype=np.int64),
                     np.zeros_like(distance))
            if not compiled.nopython_signatures:
                raise RuntimeError("no nopython signature was produced")
        except Exception as exc:
            compilation_error = f"{type(exc).__name__}: {exc}"
            if require_numba:
                raise HeuristicExecutionError(
                    "numba_compile", compilation_error,
                    preparation_seconds=time.perf_counter() - started,
                ) from exc
            # Do not leave partially JIT-wrapped helpers in the Python path.
            if local_functions:
                function.__globals__.update(local_functions)
            compiled = getattr(function, "py_func", function)

    preparation_seconds = time.perf_counter() - started
    timings = []
    for _ in range(5):
        matrix = distance.copy()
        tour = np.arange(len(matrix), dtype=np.int64)
        history = np.zeros_like(matrix)
        t0 = time.perf_counter()
        try:
            result = np.asarray(compiled(matrix, tour, history))
        except Exception as exc:
            raise HeuristicExecutionError(
                "heuristic_preflight", f"{type(exc).__name__}: {exc}",
                numba_error=compilation_error,
            ) from exc
        timings.append(time.perf_counter() - t0)
        if (not np.array_equal(matrix, distance)
                or not np.array_equal(tour, np.arange(len(matrix)))):
            raise HeuristicExecutionError("runtime_contract", "distance or tour was modified")
        if (result.shape != matrix.shape or not np.all(np.isfinite(result))
                or np.any(result < 0) or not np.allclose(result, result.T)):
            raise HeuristicExecutionError("runtime_contract", "invalid guided distance matrix")
    median = statistics.median(timings)
    metadata = {
        "runtime_version": RUNTIME_VERSION,
        "execution": (("python-budgeted-fallback" if max_call_seconds > 0 else "python-fallback")
                      if compilation_error else
                      "numba-nopython" if use_numba else "python"),
        "numba_error": compilation_error,
        "numba_version": numba_version,
        "preparation_seconds": preparation_seconds,
        "preflight_median_call_seconds": median,
        "max_call_seconds": max_call_seconds,
        "preflight_node_count": len(distance),
    }
    if max_call_seconds > 0 and median > max_call_seconds:
        raise HeuristicExecutionError(
            "heuristic_preflight", "callback exceeded its preflight time budget",
            **metadata,
        )
    if max_call_seconds > 0:
        return _BudgetedHeuristic(compiled, max_call_seconds), metadata
    return compiled, metadata
