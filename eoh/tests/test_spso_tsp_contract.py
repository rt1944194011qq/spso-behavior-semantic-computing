"""Contract tests for the typed TSP GLS compiler."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eoh" / "src"))
sys.path.insert(0, str(ROOT / "examples" / "tsp_gls_numba"))

from spso_task import TSPGLSCompiler, build_tsp_gls_registry  # noqa: E402
from spso_task.compiler import classic_gls_position  # noqa: E402


def _inputs(n=6):
    coords = np.arange(n * 2, dtype=float).reshape(n, 2)
    distance = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
    return distance, np.arange(n, dtype=int), np.zeros((n, n), dtype=float)


class TSPGLSContractTests(unittest.TestCase):
    def test_classic_sequence_matches_classic_penalty(self):
        registry = build_tsp_gls_registry()
        compiler = TSPGLSCompiler(registry)
        source, _ = compiler.compile(classic_gls_position(registry))
        namespace = {"np": np}
        exec(source, namespace)
        distance, tour, used = _inputs()
        used[0, 1] = used[1, 0] = 1.0
        used[2, 3] = used[3, 2] = 2.0
        actual = namespace["update_edge_distance"](distance, tour, used)
        expected = distance.copy()
        for i in range(len(tour)):
            u = int(tour[i])
            v = int(tour[(i + 1) % len(tour)])
            delta = distance[u, v] / (1.0 + used[u, v])
            expected[u, v] += delta
            expected[v, u] += delta
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(distance, _inputs()[0])


    def test_every_registered_sequence_obeys_public_contract(self):
        registry = build_tsp_gls_registry()
        compiler = TSPGLSCompiler(registry)
        distance, tour, used = _inputs()
        for e in registry.slots[0].modules:
            for f in registry.slots[1].modules:
                for h in registry.slots[2].modules:
                    for t in registry.slots[3].modules:
                        for w in registry.slots[4].modules:
                            from spso.models import ModuleChoice, ParticlePosition

                            params = w.default_params()
                            position = ParticlePosition(
                                (
                                    ModuleChoice.create("E", e.module_id),
                                    ModuleChoice.create("F", f.module_id),
                                    ModuleChoice.create("H", h.module_id),
                                    ModuleChoice.create("T", t.module_id),
                                    ModuleChoice.create("W", w.module_id, params),
                                )
                            )
                            source, _ = compiler.compile(position)
                            namespace = {"np": np}
                            exec(source, namespace)
                            result = namespace["update_edge_distance"](distance, tour, used)
                            self.assertEqual(result.shape, distance.shape)
                            self.assertTrue(np.all(np.isfinite(result)))
                            np.testing.assert_allclose(result, result.T)
                            self.assertTrue(np.all(result >= 0))


if __name__ == "__main__":
    unittest.main()
