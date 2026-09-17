"""Unit tests for the set-based possibility PSO layer."""

import random
import unittest

from spso.models import SetPosition
from spso.velocity import alpha_cut_sets, construct_set_position, update_velocity
from spso_task import build_tsp_gls_registry


class SetPositionTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_tsp_gls_registry()

    def test_set_position_validates_membership_capacity_and_duplicates(self):
        valid = SetPosition.create({
            slot.slot_id: [module.module_id for module in slot.modules[:3]]
            for slot in self.registry.slots
        })
        self.assertEqual(self.registry.validate_set_position(valid), valid)

        duplicate = SetPosition.create({
            slot.slot_id: [slot.modules[0].module_id, slot.modules[0].module_id]
            for slot in self.registry.slots
        })
        with self.assertRaises(ValueError):
            self.registry.validate_set_position(duplicate)

        too_large = SetPosition.create({
            slot.slot_id: [module.module_id for module in slot.modules[:4]]
            for slot in self.registry.slots
        })
        with self.assertRaises(ValueError):
            self.registry.validate_set_position(too_large)

    def test_velocity_uses_set_difference_and_stays_in_unit_interval(self):
        current = SetPosition.create({
            slot.slot_id: [module.module_id for module in slot.modules[:2]]
            for slot in self.registry.slots
        })
        pbest = SetPosition.create({
            slot.slot_id: [module.module_id for module in slot.modules[1:4]]
            for slot in self.registry.slots
        })
        gbest = SetPosition.create({
            slot.slot_id: [module.module_id for module in slot.modules[2:5]]
            for slot in self.registry.slots
        })
        old = {
            slot.slot_id: {module.module_id: 0.1 for module in slot.modules}
            for slot in self.registry.slots
        }
        velocity = update_velocity(
            old, current, pbest, gbest, self.registry, random.Random(7),
            omega=0.7, c_personal=1.4, c_global=1.8,
        )
        for values in velocity.values():
            self.assertTrue(all(0.0 <= value <= 1.0 for value in values.values()))
        # This module is in both pbest and gbest but not current, so it can
        # receive attraction; a current-only module receives only inertia.
        slot_id = self.registry.slot_ids[0]
        attracted = pbest.modules(slot_id)[-1]
        current_only = current.modules(slot_id)[0]
        self.assertGreaterEqual(velocity[slot_id][attracted], velocity[slot_id][current_only])

    def test_alpha_cut_is_absolute_and_position_is_capacity_bounded(self):
        velocity = {
            slot.slot_id: {
                module.module_id: (0.7 if index == 0 else 0.2)
                for index, module in enumerate(slot.modules)
            }
            for slot in self.registry.slots
        }
        raw = alpha_cut_sets(velocity, self.registry, 0.5)
        self.assertTrue(all(values == [slot.modules[0].module_id] for slot, values in zip(self.registry.slots, raw.values())))
        current = SetPosition.create({
            slot.slot_id: [module.module_id for module in slot.modules[:2]]
            for slot in self.registry.slots
        })
        constructed = construct_set_position(raw, current, self.registry, random.Random(9))
        for slot in self.registry.slots:
            members = constructed.modules(slot.slot_id)
            self.assertEqual(len(members), slot.capacity)
            self.assertEqual(members[0], slot.modules[0].module_id)


if __name__ == "__main__":
    unittest.main()
