"""Focused offline acceptance tests for the EoH-style S-PSO flow."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from spso import SPSOConfig, SPSOEngine
from spso.models import ModuleChoice, ParticlePosition
from spso.selector import SemanticSelector
from spso.registry import ModuleSpec
from spso_task import (
    DeterministicFakeLLM,
    TSPGLSCompiler,
    TSPModuleEvolution,
    TSPModuleLibraryBuilder,
    build_tsp_gls_registry,
)


def _build_flow(output_dir=None):
    prototype = build_tsp_gls_registry()
    fake = DeterministicFakeLLM(prototype)
    builder = TSPModuleLibraryBuilder(
        prototype,
        fake,
        modules_per_slot=10,
        max_retries=8,
        output_dir=output_dir,
    )
    registry = builder.build()
    fake.prototype = registry
    compiler = TSPGLSCompiler(registry)
    evolution = TSPModuleEvolution(
        registry,
        compiler,
        fake,
        max_retries=3,
        output_dir=output_dir,
    )
    selector = SemanticSelector(
        registry,
        llm=fake,
        module_code_provider=compiler.module_code,
        position_code_provider=lambda position: compiler.compile(position)[0],
        evolution=evolution,
    )
    return prototype, fake, builder, registry, compiler, evolution, selector


class EoHStyleFakeFlowTests(unittest.TestCase):
    def test_library_order_schema_and_executable_matrix_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, builder, registry, compiler, _, _ = _build_flow(Path(temp))

            self.assertEqual({slot.slot_id: len(slot.modules) for slot in registry.slots}, {
                "E": 10, "F": 10, "H": 10, "T": 10, "W": 10,
            })
            for slot in registry.slots:
                fingerprints = {
                    module.implementation_fingerprint(slot.slot_id)
                    for module in slot.modules
                }
                self.assertEqual(len(fingerprints), 10)

            # The real protocol is two calls per module and description comes
            # before implementation; this catches fake/online prompt drift.
            calls = builder.llm.calls
            self.assertGreaterEqual(len(calls), 100)
            self.assertEqual(
                sum(call["operator"] == "library_description" for call in calls),
                sum(call["operator"] == "library_implementation" for call in calls),
            )
            for index in range(0, len(calls), 2):
                self.assertEqual(calls[index]["operator"], "library_description")
                self.assertEqual(calls[index + 1]["operator"], "library_implementation")

            position = registry.random_position(random.Random(7))
            source, _ = compiler.compile(position)
            namespace = {}
            exec(source, namespace)
            matrix = np.ones((6, 6), dtype=float)
            np.fill_diagonal(matrix, 0.0)
            result = namespace["update_edge_distance"](
                matrix, np.arange(6), np.zeros_like(matrix)
            )
            self.assertEqual(result.shape, matrix.shape)
            self.assertTrue(np.all(np.isfinite(result)))
            self.assertTrue(np.all(result >= 0.0))
            self.assertTrue(np.allclose(result, result.T))
            with self.assertRaises(ValueError):
                namespace["update_edge_distance"](matrix[:2], np.arange(6), np.zeros_like(matrix))

    def test_initial_p0_has_twenty_unique_examples_and_keeps_top_ten(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, registry, compiler, evolution, selector = _build_flow(Path(temp))
            config = SPSOConfig(
                population_size=10,
                generations=0,
                initial_samples=20,
                output_dir=temp,
                heuristic_operators=("p1", "p2", "p3", "p4"),
            )
            engine = SPSOEngine(
                registry,
                compiler,
                lambda code: float(sum(map(ord, code)) % 100000),
                config,
                selector,
            )
            particles = engine._initialise()
            self.assertEqual(len(engine.records), 20)
            self.assertEqual({record["operator"] for record in engine.records}, {"p0"})
            self.assertEqual(len({record["position_text"] for record in engine.records}), 20)
            self.assertEqual(len(particles), 10)
            all_scores = sorted(record["objective"] for record in engine.records)
            kept_scores = sorted(particle.current_objective for particle in particles)
            self.assertEqual(kept_scores, all_scores[:10])

    def test_operator_proposals_change_the_right_things(self):
        for operator in ("p1", "p2", "p3", "p4"):
            with self.subTest(operator=operator), tempfile.TemporaryDirectory() as temp:
                _, fake, _, registry, compiler, evolution, _ = _build_flow(Path(temp))
                cuts = {
                    slot.slot_id: [module.module_id for module in slot.modules[:3]]
                    for slot in registry.slots
                }
                base, _ = evolution.generate_sequence(cuts, operator="p0")
                proposal = evolution.propose(
                    operator, base, base, cuts, generation=1, parent_id="p00001"
                )
                self.assertIsNotNone(proposal.position)
                if operator == "p4":
                    self.assertEqual(
                        [choice.module for choice in proposal.position.choices],
                        [choice.module for choice in base.choices],
                    )
                    self.assertNotEqual(proposal.position.canonical(registry.slot_ids), base.canonical(registry.slot_ids))
                else:
                    self.assertEqual(len(proposal.new_modules), 1)
                    changed_slots = [
                        slot for slot in registry.slot_ids
                        if proposal.position.as_mapping()[slot].module != base.as_mapping()[slot].module
                    ]
                    self.assertGreaterEqual(len(changed_slots), 1)
                    item = proposal.new_modules[0]
                    module = ModuleSpec(
                        module_id=item.module_id,
                        description=item.description,
                        parameters=item.parameters,
                        recipe={key: value for key, value in item.recipe.items() if key != "parameters"},
                        source_operator=operator,
                        semantic_family=item.recipe.get("family", ""),
                        polarity=item.recipe.get("polarity", "neutral"),
                    )
                    registry.register_module(item.slot, module)
                    compiled, _ = compiler.compile(proposal.position)
                    base_compiled, _ = compiler.compile(base)
                    self.assertNotEqual(compiled, base_compiled)
                    self.assertEqual(item.slot, changed_slots[0])

    def test_all_four_operators_produce_forty_audited_child_records(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, registry, compiler, _, selector = _build_flow(Path(temp))
            # Rebuild the evolution object after the helper's builder calls so
            # the engine owns one output directory and one audit stream.
            fake = selector.llm
            evolution = TSPModuleEvolution(registry, compiler, fake, max_retries=3, output_dir=Path(temp))
            selector.evolution = evolution
            config = SPSOConfig(
                population_size=10,
                generations=1,
                initial_samples=20,
                max_evaluations=60,
                output_dir=temp,
                heuristic_operators=("p1", "p2", "p3", "p4"),
                num_samplers=4,
                num_evaluators=4,
                seed=4,
            )
            engine = SPSOEngine(
                registry,
                compiler,
                lambda code: float(sum(map(ord, code)) % 100000),
                config,
                selector,
            )
            summary = engine.run()
            children = [record for record in engine.records if record["generation"] == 1]
            self.assertEqual(len(children), 40)
            self.assertEqual(
                {operator: sum(record["operator"] == operator for record in children)
                 for operator in ("p1", "p2", "p3", "p4")},
                {"p1": 10, "p2": 10, "p3": 10, "p4": 10},
            )
            self.assertEqual(summary["samples"], 60)
            for record in children:
                self.assertIsNotNone(record["operator"])
                if record["objective"] is None:
                    self.assertTrue(record["failure_reason"])

    def test_resume_loads_checkpoint_library_without_builder(self):
        import runSPSO

        with tempfile.TemporaryDirectory() as temp:
            _, fake, _, registry, _, _, _ = _build_flow()
            checkpoint = Path(temp) / "generation_0000.json"
            checkpoint.write_text(json.dumps({"module_library": registry.library_payload()}), encoding="utf-8")
            old_resume = runSPSO.RESUME_FROM
            old_frozen = runSPSO.FROZEN_MODULE_LIBRARY
            try:
                runSPSO.RESUME_FROM = checkpoint
                runSPSO.FROZEN_MODULE_LIBRARY = None
                with mock.patch.object(runSPSO, "TSPModuleLibraryBuilder", side_effect=AssertionError("builder must not run on resume")):
                    restored = runSPSO.load_module_registry(build_tsp_gls_registry(), fake)
            finally:
                runSPSO.RESUME_FROM = old_resume
                runSPSO.FROZEN_MODULE_LIBRARY = old_frozen
            self.assertEqual(restored.library_hash(), registry.library_hash())
            self.assertIs(fake.prototype, restored)


if __name__ == "__main__":
    unittest.main()
