from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from eoh.mae import MAEConfig, MAEEngine, ProgramDraft
from eoh.mae.models import CutItem, ProgramIndividual
from eoh.mae.set_algebra import scalar_multiply, set_add, update_velocity
from eoh.mae.topology import ring_neighbors
from eoh.mae.validation import ProgramValidator
from eoh.problem import BaseProblem


class FakeProblem(BaseProblem):
    template_program = """
def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    return edge_distance.copy()
"""
    task_description = "Return a modified matrix."

    def __init__(self):
        super().__init__(timeout=5, n_processes=1)

    def evaluate_program(self, program_str, callable_func):
        matrix = np.ones((3, 3), dtype=float)
        result = callable_func(matrix, np.array([0, 1, 2]), np.zeros_like(matrix))
        return float(np.asarray(result).mean())


class FakeGenerator:
    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()

    def _draft(self, label):
        with self._lock:
            self._counter += 1
            value = self._counter / 1000.0
            index = self._counter
        code = (
            "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n"
            f"    return edge_distance.copy() + {value:.6f}\n"
        )
        return ProgramDraft(f"{label} deterministic candidate {index}", code, f"fake-{index}")

    def generate_h0(self, agent_id, sample_index):
        return self._draft(f"H0 agent {agent_id}")

    def select_cut(self, agent_id, guidance, cut, individuals, selection_size):
        return [item.individual_id for item in cut[:selection_size]]

    def generate_variation(self, agent_id, guidance, variation, parents, selected):
        return self._draft(f"{guidance}->{variation} agent {agent_id}")


class SetAlgebraTests(unittest.TestCase):
    def test_add_keeps_max_probability(self):
        self.assertEqual(set_add({"a": 0.2, "b": 0.8}, {"a": 0.7, "c": 0.4}),
                         {"a": 0.7, "b": 0.8, "c": 0.4})

    def test_scalar_multiply_clips(self):
        self.assertEqual(scalar_multiply({"a": 0.8}, 2.0), {"a": 1.0})

    def test_h1_and_h2_weights_change_velocity(self):
        h1 = update_velocity(
            {}, ["x"], "p", "g", "l", "r", omega=0.5,
            personal_weight=0.9, global_weight=0.25,
            left_weight=0.4, right_weight=0.4,
            random_values=(1.0, 1.0, 1.0, 1.0),
        )
        h2 = update_velocity(
            {}, ["x"], "p", "g", "l", "r", omega=0.5,
            personal_weight=0.25, global_weight=0.9,
            left_weight=0.4, right_weight=0.4,
            random_values=(1.0, 1.0, 1.0, 1.0),
        )
        self.assertGreater(h1["p"], h2["p"])
        self.assertGreater(h2["g"], h1["g"])

    def test_ring_topology(self):
        self.assertEqual(ring_neighbors(0, 4), (3, 1))
        self.assertEqual(ring_neighbors(3, 4), (2, 0))


class EngineIntegrationTests(unittest.TestCase):
    def test_two_generation_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            problem = FakeProblem()
            config = MAEConfig(
                num_agents=3,
                position_size=2,
                generations=1,
                initial_multiplier=2,
                max_cut_size=4,
                selection_size=2,
                num_llm_workers=4,
                num_evaluators=4,
                isolate_evaluation=False,
                output_dir=temporary,
                evaluation_fingerprint="fake-v1",
            )
            engine = MAEEngine(
                config,
                problem,
                FakeGenerator(),
                ProgramValidator(problem.template_program),
            )
            summary = engine.run()

            self.assertIsNotNone(summary["best_fitness"])
            self.assertEqual(len(engine.agents), 3)
            self.assertTrue(all(len(agent.position_ids) == 2 for agent in engine.agents))
            self.assertTrue(all(agent.pbest_id for agent in engine.agents))
            paths = {item.operator_path for item in engine.archive.values()}
            self.assertIn("H0", paths)
            self.assertIn("H1->H3", paths)
            self.assertIn("H2->H5", paths)

            root = Path(temporary)
            self.assertTrue((root / "best" / "best_heuristic.py").is_file())
            self.assertTrue((root / "checkpoints" / "generation_0001.json").is_file())
            self.assertTrue((root / "metrics" / "generation_metrics.csv").is_file())
            self.assertTrue((root / "metrics" / "agent_metrics.jsonl").is_file())
            self.assertTrue((root / "metrics" / "operator_metrics.jsonl").is_file())
            self.assertTrue((root / "telemetry" / "events.jsonl").is_file())
            self.assertTrue((root / "telemetry" / "velocity_updates.jsonl").is_file())
            self.assertTrue((root / "telemetry" / "cut_decisions.jsonl").is_file())
            self.assertTrue((root / "telemetry" / "agent_transitions.jsonl").is_file())
            metrics = (root / "metrics" / "generation_metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(metrics), 2)
            self.assertEqual(json.loads(metrics[-1])["generation"], 1)
            events = [
                json.loads(line)["event"]
                for line in (root / "telemetry" / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertIn("run_started", events)
            self.assertIn("velocity_updated", events)
            self.assertIn("agent_position_updated", events)
            self.assertIn("run_finished", events)


if __name__ == "__main__":
    unittest.main()
