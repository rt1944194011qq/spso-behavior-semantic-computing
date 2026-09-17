"""Regression tests for route evidence and its transport through the engine."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eoh' / 'src'))
sys.path.insert(0, str(ROOT / 'examples' / 'tsp_gls_numba'))

from eoh.eoh.evolution import _eval_with_timeout
from spso import SPSOConfig, SPSOEngine
from spso_task import TSPGLSCompiler, build_tsp_gls_registry
from spso_task.reporting import TSPGLSWithDetails, verify_route


class RouteReportingTests(unittest.TestCase):
    def test_route_validation(self):
        matrix = np.ones((4, 4)) - np.eye(4)
        route = np.array([[3, 1], [0, 2], [1, 3], [2, 0]])
        evidence = verify_route(matrix, route, 4.0)
        self.assertEqual(evidence['closed_tour'], [0, 1, 2, 3, 0])
        with self.assertRaises(ValueError):
            verify_route(matrix, route, 3.0)
        with self.assertRaises(ValueError):
            verify_route(matrix, np.array([[1, 1], [0, 0], [3, 3], [2, 2]]), 4.0)

    def test_real_parallel_evaluation_and_checkpoint_details(self):
        registry = build_tsp_gls_registry()
        compiler = TSPGLSCompiler(registry)
        task = TSPGLSWithDetails(n_inst_eva=1, time_limit=10, ite_max=3, timeout=60)
        with tempfile.TemporaryDirectory() as output, contextlib.redirect_stdout(io.StringIO()):
            engine = SPSOEngine(registry, compiler,
                                lambda code: _eval_with_timeout(task, code, 60),
                                SPSOConfig(population_size=2, initial_samples=2, generations=1,
                                           num_samplers=2, num_evaluators=2,
                                           cache_evaluations=False, output_dir=output))
            result = engine.run()
            self.assertEqual(result['independent_evaluations'], 4)
            best = json.loads((Path(output) / 'samples/best.json').read_text(encoding='utf-8'))
            self.assertTrue(best['details']['instances'][0]['valid_tour'])
            self.assertEqual(len(best['details']['instances'][0]['closed_tour']), 101)
            checkpoint = json.loads((Path(output) / 'checkpoints/generation_0001.json').read_text())
            self.assertTrue(all(p['current_details'] for p in checkpoint['particles']))
            self.assertIn('reference=', (Path(output) / 'run_log.txt').read_text(encoding='utf-8'))
            cut_rows = [
                json.loads(line)
                for line in (Path(output) / 'cut_log/generation_0001.jsonl').read_text(encoding='utf-8').splitlines()
            ]
            self.assertEqual(len(cut_rows), 2)
            self.assertEqual([slot['slot'] for slot in cut_rows[0]['slots']], list(registry.slot_ids))
            self.assertTrue(all(slot['cut_modules'] for slot in cut_rows[0]['slots']))
            # Resume round-trips current evidence even when no generations remain.
            restored, _ = engine._restore(Path(output) / 'checkpoints/generation_0001.json')
            self.assertTrue(all(p.current_details for p in restored))
            # Cached retrieval preserves the evidence belonging to that score.
            engine.config.cache_evaluations = True
            first = engine._evaluate(engine.gbest_position, 'test', 'B')
            second = engine._evaluate(engine.gbest_position, 'test', 'B')
            self.assertTrue(second.cached)
            self.assertEqual(first.details, second.details)


if __name__ == '__main__':
    unittest.main()
