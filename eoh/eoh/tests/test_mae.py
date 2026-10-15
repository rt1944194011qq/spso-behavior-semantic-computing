from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from eoh.mae import LLMProgramGenerator, MAEConfig, MAEEngine, ProgramDraft
from eoh.mae.experience import birth_probability, calibrated_multiplier, decay_and_prune, sample_cut
from eoh.mae.models import ExperienceRecord, ProgramIndividual
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

    def describe_difference(self, agent_id, source, target, direction_type="improve"):
        return (
            f"Adapt the behavior of {source.individual_id} toward "
            f"{target.individual_id} as {direction_type}."
        )

    def apply_experience(self, agent_id, current, experience, exemplar):
        return self._draft(f"DIFF {experience.source_role} agent {agent_id}")

    def generate_variation(self, agent_id, guidance, variation, parents, selected):
        draft = self._draft(f"{guidance}->{variation} agent {agent_id}")
        if variation == "H4":
            draft.code = draft.code.replace("edge_distance.copy() +", "edge_distance.copy() + 2 *")
        return draft


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

    def test_birth_probability_uses_both_endpoint_fitnesses(self):
        both_best = birth_probability(0.1, 0.1, 0.1, 0.1, 0.2, 0.6, 0.6)
        toward_best = birth_probability(0.2, 0.1, 0.1, 0.1, 0.2, 0.6, 0.6)
        away_from_best = birth_probability(0.1, 0.2, 0.1, 0.1, 0.2, 0.6, 0.6)
        both_worse = birth_probability(0.2, 0.2, 0.1, 0.1, 0.2, 0.6, 0.6)
        tiny_scale = birth_probability(
            2e-6, 1e-6, 1e-6, 1e-6, 0.2, 0.6, 0.6
        )

        self.assertAlmostEqual(both_best, 0.6)
        self.assertGreater(toward_best, away_from_best)
        self.assertGreater(away_from_best, both_worse)
        self.assertAlmostEqual(toward_best, tiny_scale)

    def test_experience_decay(self):
        item = ExperienceRecord("e", "left", "a", "b", "change", 0.1, 0.8, 1, 1)
        decayed = decay_and_prune(
            {"e": item}, half_life=1.0, drop_below=0.05, capacity=4
        )
        self.assertAlmostEqual(decayed["e"].probability, 0.4)
        self.assertAlmostEqual(item.probability, 0.8)

    def test_lambda_and_probability_clipping(self):
        self.assertAlmostEqual(calibrated_multiplier([0.5] * 10, 4), 0.8)
        items = {
            "personal": ExperienceRecord("personal", "personal", "a", "b", "change", 0.1, 0.8, 1, 1),
            "left": ExperienceRecord("left", "left", "a", "b", "change", 0.2, 0.2, 1, 1),
            "global": ExperienceRecord("global", "global", "a", "b", "change", 0.1, 0.4, 1, 1),
        }
        import random
        cut, info = sample_cut(
            items, target_size=3, neighbor_reserved_slots=1,
            generation=1, agent_id=0,
            enabled_operators=("H1", "H2", "H3"),
            h1_boost=1.5, h2_boost=1.5, rng=random.Random(1),
        )
        self.assertEqual(len(cut), 3)
        self.assertIn("left", [item.source_role for item in cut])
        self.assertEqual(info["boosted_probabilities"]["personal"], 1.0)
        self.assertEqual(items["personal"].probability, 0.8)
        self.assertTrue(all(0 <= item.selection_probability <= 1 for item in cut))

    def test_llm_difference_and_complete_program_contract(self):
        class StubLLM:
            def __init__(self):
                self.responses = [
                    "Adjust the edge penalty using the stronger strategy.",
                    "{Apply the historical edge-penalty difference.}\n"
                    "```python\n"
                    "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n"
                    "    return edge_distance.copy() + 0.5\n"
                    "```",
                ]

            def get_response(self, prompt):
                return self.responses.pop(0)

        code = FakeProblem.template_program
        first = ProgramIndividual("a", "first strategy", code, "hash-a", 2.0, True, 0, 0, "H0")
        second = ProgramIndividual("b", "second strategy", code, "hash-b", 1.0, True, 0, 0, "H0")
        generator = LLMProgramGenerator(
            StubLLM(), "Return a modified matrix.", code, retries=1
        )
        difference = generator.describe_difference(0, first, second)
        self.assertIn("edge penalty", difference)
        experience = ExperienceRecord(
            "e", "global", "a", "b", difference, 1.0, 0.6, 1, 1
        )
        draft = generator.apply_experience(0, first, experience, second)
        self.assertIsNotNone(draft.code)
        self.assertIn("def update_edge_distance", draft.code)
        self.assertTrue(draft.description)


class EngineIntegrationTests(unittest.TestCase):
    def test_two_generation_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            problem = FakeProblem()
            config = MAEConfig(
                num_agents=3,
                position_size=1,
                generations=1,
                initial_multiplier=2,
                cut_target_size=4,
                cut_min_size=2,
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
            self.assertTrue(all(agent.current_id in engine.archive for agent in engine.agents))
            self.assertTrue(all(agent.pbest_id for agent in engine.agents))
            self.assertTrue(any(agent.current_id != agent.pbest_id for agent in engine.agents))
            paths = {item.operator_path for item in engine.archive.values()}
            self.assertIn("H0", paths)
            self.assertIn("DIFF", paths)
            self.assertIn("H1+H2->H3", paths)
            self.assertIn("H1+H2->H5", paths)

            root = Path(temporary)
            self.assertTrue((root / "best" / "best_heuristic.py").is_file())
            self.assertTrue((root / "checkpoints" / "generation_0001.json").is_file())
            self.assertTrue((root / "metrics" / "generation_metrics.csv").is_file())
            self.assertTrue((root / "metrics" / "agent_metrics.jsonl").is_file())
            self.assertTrue((root / "metrics" / "operator_metrics.jsonl").is_file())
            self.assertTrue((root / "metrics" / "experience_source_metrics.jsonl").is_file())
            self.assertTrue((root / "lineage" / "experience_edges.jsonl").is_file())
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
            checkpoint = json.loads(
                (root / "checkpoints" / "generation_0001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["schema_version"], 2)
            self.assertTrue(all("velocity" in agent for agent in checkpoint["agents"]))
            self.assertTrue(all("current_id" in agent for agent in checkpoint["agents"]))
            self.assertTrue(any(
                item["direction_type"] == "diversify"
                and item["source_role"] in ("left", "right")
                and "source_id" in item and "target_id" in item
                and item["source_fitness"] is not None
                and item["target_fitness"] is not None
                for agent in checkpoint["agents"]
                for item in agent["velocity"].values()
            ))

    def test_resume_preserves_experience_and_reaches_second_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            problem = FakeProblem()
            config = MAEConfig(
                num_agents=3, generations=1, initial_multiplier=2,
                cut_target_size=4, cut_min_size=2,
                num_llm_workers=4, num_evaluators=4,
                isolate_evaluation=False, output_dir=temporary,
                evaluation_fingerprint="fake-resume-v2",
            )
            MAEEngine(config, problem, FakeGenerator(),
                      ProgramValidator(problem.template_program)).run()
            checkpoint = Path(temporary) / "checkpoints" / "generation_0001.json"
            original_config = (Path(temporary) / "config.json").read_text(encoding="utf-8")
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertTrue(any(agent["velocity"] for agent in saved["agents"]))

            resumed_config = MAEConfig(
                num_agents=3, generations=2, initial_multiplier=2,
                cut_target_size=4, cut_min_size=2,
                num_llm_workers=4, num_evaluators=4,
                isolate_evaluation=False, output_dir=temporary,
                resume_from=str(checkpoint),
                evaluation_fingerprint="fake-resume-v2",
            )
            engine = MAEEngine(
                resumed_config, problem, FakeGenerator(),
                ProgramValidator(problem.template_program),
            )
            engine.run()
            self.assertTrue((Path(temporary) / "checkpoints" / "generation_0002.json").is_file())
            self.assertTrue(all(agent.velocity for agent in engine.agents))
            self.assertEqual(
                (Path(temporary) / "config.json").read_text(encoding="utf-8"),
                original_config,
            )

            incompatible = MAEConfig(
                num_agents=3, generations=3, initial_multiplier=2,
                cut_target_size=4, cut_min_size=2,
                h1_selection_boost=2.0,
                num_llm_workers=4, num_evaluators=4,
                isolate_evaluation=False, output_dir=temporary,
                resume_from=str(checkpoint),
                evaluation_fingerprint="fake-resume-v2",
            )
            with self.assertRaisesRegex(ValueError, "h1_selection_boost"):
                MAEEngine(
                    incompatible, problem, FakeGenerator(),
                    ProgramValidator(problem.template_program),
                )
            self.assertEqual(
                (Path(temporary) / "config.json").read_text(encoding="utf-8"),
                original_config,
            )


if __name__ == "__main__":
    unittest.main()
