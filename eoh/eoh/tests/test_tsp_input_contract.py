from __future__ import annotations

import types
import unittest
from pathlib import Path

import numpy as np

from eoh.mae.generator import LLMProgramGenerator
from eoh.mae.models import ExperienceRecord, ProgramIndividual
from eoh.mae.validation import (
    TSP_INPUT_CONTRACT,
    ProgramValidator,
    make_tsp_smoke_test,
)


TEMPLATE = """def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    return edge_distance.copy()
"""
HISTORY_UPDATE = """def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    edge_n_used[0, 1] += 1.0
    edge_n_used[1, 0] += 1.0
    return edge_distance.copy()
"""


class TSPInputContractTests(unittest.TestCase):
    def validator(self, **options):
        return ProgramValidator(TEMPLATE, make_tsp_smoke_test(**options))

    def test_history_updates_are_allowed_by_default(self):
        self.assertTrue(self.validator().validate(HISTORY_UPDATE).valid)

    def test_old_readonly_history_rule_remains_available(self):
        result = self.validator(allow_edge_n_used_update=False).validate(HISTORY_UPDATE)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "edge_n_used was modified in place")

    def test_original_distances_and_tour_are_still_readonly(self):
        for mutation, reason in (
            ("edge_distance[0, 1] += 1.0", "edge_distance was modified in place"),
            ("local_opt_tour[0] = 1", "local_opt_tour was modified in place"),
        ):
            with self.subTest(mutation=mutation):
                code = TEMPLATE.replace("return edge_distance.copy()", mutation + "\n    return edge_distance.copy()")
                result = self.validator().validate(code)
                self.assertFalse(result.valid)
                self.assertEqual(result.reason, reason)

    def test_invalid_output_is_still_rejected(self):
        for expression in (
            "np.zeros((2, 2))",
            "np.full_like(edge_distance, np.nan)",
            "-np.ones_like(edge_distance)",
            "edge_distance + np.triu(np.ones_like(edge_distance), 1)",
        ):
            with self.subTest(expression=expression):
                code = TEMPLATE.replace("edge_distance.copy()", expression)
                self.assertFalse(self.validator().validate(code).valid)

    def test_history_writeback_persists_between_calls_on_same_state(self):
        module = types.ModuleType("stateful_heuristic")
        exec(HISTORY_UPDATE, module.__dict__)
        distance = np.ones((4, 4)) - np.eye(4)
        tour = np.arange(4)
        used = np.zeros_like(distance)
        for _ in range(2):
            module.update_edge_distance(distance, tour, used)
        self.assertEqual(used[0, 1], 2.0)
        self.assertEqual(used[1, 0], 2.0)
        np.testing.assert_array_equal(distance, np.ones((4, 4)) - np.eye(4))
        np.testing.assert_array_equal(tour, np.arange(4))

    def test_frozen_eoh_best_passes_aligned_but_not_old_rule(self):
        frozen = Path(__file__).resolve().parents[3] / "tsp_baselines/frozen/EOH_USER_BEST.py"
        if not frozen.is_file():
            self.skipTest("Project's frozen EoH fixture is not present")
        code = frozen.read_text(encoding="utf-8")
        self.assertTrue(self.validator().validate(code).valid)
        result = self.validator(allow_edge_n_used_update=False).validate(code)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "edge_n_used was modified in place")

    def test_all_code_generation_paths_receive_same_input_contract(self):
        class StubLLM:
            def __init__(self):
                self.prompts = []

            def get_response(self, prompt):
                self.prompts.append(prompt)
                return "{Update GLS history and return fresh distances.}\n```python\n" + HISTORY_UPDATE + "```"

        llm = StubLLM()
        generator = LLMProgramGenerator(
            llm, "Return a modified matrix.", TEMPLATE,
            retries=1, input_contract=TSP_INPUT_CONTRACT,
        )
        parent = ProgramIndividual("a", "parent", TEMPLATE, "hash-a", 2.0, True, 0, 0, "H0")
        exemplar = ProgramIndividual("b", "target", TEMPLATE, "hash-b", 1.0, True, 0, 0, "H0")
        experience = ExperienceRecord("e", "global", "a", "b", "Adapt penalties.", 1.0, 0.6, 1, 1)
        self.assertIsNotNone(generator.generate_h0(0, 0).code)
        self.assertIsNotNone(generator.apply_experience(0, parent, experience, exemplar).code)
        for operator in ("H3", "H4", "H5"):
            generator.generate_variation(0, "H1+H2", operator, [parent, exemplar], [parent, exemplar])
        self.assertEqual(len(llm.prompts), 5)
        for prompt in llm.prompts:
            self.assertIn(TSP_INPUT_CONTRACT, prompt)
            self.assertNotIn("instead of modifying inputs", prompt)


if __name__ == "__main__":
    unittest.main()
