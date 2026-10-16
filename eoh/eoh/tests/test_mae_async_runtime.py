from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from eoh.eoh.evolution import _eval_with_timeout
from eoh.mae import MAEConfig, MAEEngine, ProgramDraft
from eoh.mae.legacy_engine import _CandidateSpec
from eoh.mae.validation import ProgramValidator
from test_mae import ControlledGenerator, FakeGenerator, FakeProblem

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "tsp_gls_numba"
sys.path.insert(0, str(EXAMPLE))
from heuristic_runtime import HeuristicExecutionError, prepare_heuristic


class LargeResultProblem:
    def evaluate(self, code):
        return {"fitness": 1.0, "details": "x" * 1048576}


class BlockingProblem:
    def evaluate(self, code):
        time.sleep(30)


class AsyncPipelineTests(unittest.TestCase):
    def test_fast_agent_reaches_h6_while_other_agent_waits_in_h3(self):
        slow_started = threading.Event()
        fast_reached_h6 = threading.Event()
        slow_was_released = []

        class Generator(ControlledGenerator):
            def generate_variation(self, agent_id, guidance, variation, parents, selected):
                if agent_id == 0 and variation == "H3":
                    slow_started.set()
                    slow_was_released.append(fast_reached_h6.wait(3))
                if agent_id == 1 and variation == "H3":
                    if not slow_started.wait(3):
                        raise AssertionError("slow agent never entered H3")
                if agent_id == 1 and variation == "H6":
                    fast_reached_h6.set()
                return super().generate_variation(agent_id, guidance, variation, parents, selected)

        with tempfile.TemporaryDirectory() as temporary:
            engine = MAEEngine(
                MAEConfig(num_agents=2, generations=1, initial_multiplier=2,
                          num_llm_workers=4, num_evaluators=4,
                          isolate_evaluation=False, output_dir=temporary),
                FakeProblem(), Generator(),
            )
            engine.run()
            self.assertEqual(slow_was_released, [True])
            decisions = [json.loads(line) for line in
                         (Path(temporary) / "metrics/pool_decisions.jsonl").read_text().splitlines()]
            for agent_id in (0, 1):
                self.assertEqual([item["operator"] for item in decisions
                                  if item["agent_id"] == agent_id], ["H3", "H4", "H5", "H6"])

    def test_initial_generation_and_evaluation_overlap(self):
        evaluation_started = threading.Event()
        overlapped = []

        class Problem(FakeProblem):
            def evaluate_program(self, code, function):
                evaluation_started.set()
                return super().evaluate_program(code, function)

        class Generator(FakeGenerator):
            def generate_h0(self, agent_id, sample_index):
                if agent_id == 1:
                    overlapped.append(evaluation_started.wait(3))
                return super().generate_h0(agent_id, sample_index)

        with tempfile.TemporaryDirectory() as temporary:
            MAEEngine(
                MAEConfig(num_agents=2, generations=0, initial_multiplier=1,
                          num_llm_workers=2, num_evaluators=2,
                          isolate_evaluation=False, output_dir=temporary),
                Problem(), Generator(),
            ).run()
            self.assertEqual(overlapped, [True])

    def test_concurrent_identical_candidates_share_one_evaluation(self):
        started = threading.Event()
        release = threading.Event()

        class Problem(FakeProblem):
            def evaluate_program(self, code, function):
                started.set()
                if not release.wait(3):
                    raise AssertionError("evaluation was not released")
                return super().evaluate_program(code, function)

        with tempfile.TemporaryDirectory() as temporary:
            problem = Problem()
            engine = MAEEngine(
                MAEConfig(generations=0, output_dir=temporary, isolate_evaluation=False),
                problem, FakeGenerator(), ProgramValidator(problem.template_program),
            )
            code = problem.template_program
            specs = [_CandidateSpec(agent_id, 0, ProgramDraft("same", code), "H0")
                     for agent_id in (0, 1)]
            try:
                with ThreadPoolExecutor(2) as evaluations, ThreadPoolExecutor(2) as callers:
                    engine._eval_pool = evaluations
                    futures = [callers.submit(engine._evaluate_specs, [spec], {}) for spec in specs]
                    self.assertTrue(started.wait(3))
                    release.set()
                    results = [future.result() for future in futures]
                self.assertEqual(engine._evaluation_count, 1)
                self.assertEqual(engine._cache_hits, 1)
                self.assertEqual(len({result[0].individual_id for result in results}), 2)
                records = (Path(temporary) / "archive/programs.jsonl").read_text().splitlines()
                self.assertEqual(len([json.loads(line) for line in records]), 2)
            finally:
                engine._close_logger()


class NumbaRuntimeTests(unittest.TestCase):
    def test_compiled_helpers_preserve_output_and_history_writeback(self):
        code = '''import numpy as np
def penalty(value):
    return value * 0.2
def update_edge_distance(distance, tour, history):
    result = distance.copy()
    for i in range(len(tour)):
        a, b = tour[i], tour[(i + 1) % len(tour)]
        history[a, b] += 1.0
        history[b, a] += 1.0
        result[a, b] += penalty(distance[a, b])
        result[b, a] = result[a, b]
    return result
'''
        modules = [types.ModuleType(name) for name in ("plain_fixture", "jit_fixture")]
        for module in modules:
            exec(code, module.__dict__)
        distance = np.ones((12, 12)) - np.eye(12)
        compiled, metadata = prepare_heuristic(
            modules[1].update_edge_distance, distance, max_call_seconds=0,
        )
        tour = np.arange(12, dtype=np.int64)
        h1, h2 = np.zeros_like(distance), np.zeros_like(distance)
        expected = modules[0].update_edge_distance(distance, tour, h1)
        actual = compiled(distance, tour, h2)
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(h1, h2)
        self.assertEqual(metadata["execution"], "numba-nopython")
        self.assertTrue(compiled.nopython_signatures)

    def test_unsupported_code_is_rejected_without_python_fallback(self):
        module = types.ModuleType("unsupported_fixture")
        exec("def update_edge_distance(d,t,h):\n    return eval('d.copy()')\n", module.__dict__)
        with self.assertRaises(HeuristicExecutionError) as context:
            prepare_heuristic(module.update_edge_distance, np.ones((4, 4)), require_numba=True)
        self.assertEqual(context.exception.details["stage"], "numba_compile")

    def test_fast_unsupported_numpy_code_can_use_budgeted_fallback(self):
        module = types.ModuleType("advanced_index_fixture")
        module.__dict__["np"] = np
        exec("def update_edge_distance(d,t,h):\n    return eval('d.copy()')\n", module.__dict__)
        function, metadata = prepare_heuristic(
            module.update_edge_distance, np.ones((200, 200)) - np.eye(200),
            require_numba=False, max_call_seconds=0.005,
        )
        self.assertEqual(metadata["execution"], "python-budgeted-fallback")
        self.assertTrue(metadata["numba_error"])
        self.assertIsNotNone(function)

    def test_optional_numba_allows_python_with_only_the_outer_deadline(self):
        module = types.ModuleType("unlimited_callback_fixture")
        exec("def update_edge_distance(d,t,h):\n    return eval('d.copy()')\n", module.__dict__)
        function, metadata = prepare_heuristic(
            module.update_edge_distance, np.ones((4, 4)),
            require_numba=False, max_call_seconds=0,
        )
        self.assertEqual(metadata["execution"], "python-fallback")
        self.assertIsNotNone(function)

    def test_optional_numba_handles_a_missing_compiler(self):
        def function(distance, tour, history):
            return distance.copy()

        with patch.dict(sys.modules, {"numba": None}):
            prepared, metadata = prepare_heuristic(
                function, np.ones((4, 4)), require_numba=False, max_call_seconds=0,
            )
        self.assertIsNotNone(prepared)
        self.assertEqual(metadata["execution"], "python-fallback")
        self.assertIsNone(metadata["numba_version"])

    def test_frozen_eoh_best_is_not_excluded_only_for_jit_incompatibility(self):
        frozen = Path(__file__).resolve().parents[3] / "tsp_baselines/frozen/EOH_USER_BEST.py"
        if not frozen.is_file():
            self.skipTest("project EoH fixture is unavailable")
        module = types.ModuleType("eoh_frozen_jit_fixture")
        exec(frozen.read_text(encoding="utf-8"), module.__dict__)
        function, metadata = prepare_heuristic(
            module.update_edge_distance, np.ones((200, 200)) - np.eye(200),
            require_numba=False, max_call_seconds=0.005,
        )
        self.assertIsNotNone(function)
        self.assertIn(metadata["execution"], ("numba-nopython", "python-budgeted-fallback"))

    def test_slow_callback_is_rejected_before_full_gls_evaluation(self):
        def slow(distance, tour, history):
            time.sleep(0.01)
            return distance.copy()

        with self.assertRaises(HeuristicExecutionError) as context:
            prepare_heuristic(slow, np.ones((4, 4)), use_numba=False, max_call_seconds=0.001)
        self.assertEqual(context.exception.details["stage"], "heuristic_preflight")

    def test_history_dependent_slowdown_is_also_detected(self):
        def slow_after_history_changes(distance, tour, history):
            if history[0, 0] > 0:
                time.sleep(0.002)
            return distance.copy()

        distance = np.ones((4, 4))
        guarded, _ = prepare_heuristic(
            slow_after_history_changes, distance, use_numba=False, max_call_seconds=0.001,
        )
        with self.assertRaises(HeuristicExecutionError) as context:
            for _ in range(32):
                guarded(distance, np.arange(4), np.ones_like(distance))
        self.assertEqual(context.exception.details["stage"], "heuristic_runtime")

    def test_contract_is_checked_at_the_real_training_size(self):
        def wrong_for_large_instances(distance, tour, history):
            return -distance if len(tour) > 4 else distance.copy()

        with self.assertRaises(HeuristicExecutionError) as context:
            prepare_heuristic(wrong_for_large_instances, np.ones((200, 200)),
                              use_numba=False, max_call_seconds=0)
        self.assertEqual(context.exception.details["stage"], "runtime_contract")


class EvaluationTransportTests(unittest.TestCase):
    def test_large_traced_result_is_drained_before_process_join(self):
        result = _eval_with_timeout(LargeResultProblem(), "", 10)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["details"]), 1048576)

    def test_hung_evaluation_is_terminated(self):
        self.assertIsNone(_eval_with_timeout(BlockingProblem(), "", 2))

    def test_mae_timeout_records_the_whole_candidate_limit(self):
        result = _eval_with_timeout(BlockingProblem(), "", 2, report_failures=True)
        self.assertIsNone(result["fitness"])
        self.assertEqual(result["details"]["failure"]["stage"], "evaluation_timeout")
        self.assertEqual(result["details"]["failure"]["timeout_seconds"], 2)


if __name__ == "__main__":
    unittest.main()
