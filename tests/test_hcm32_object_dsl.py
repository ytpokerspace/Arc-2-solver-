import tempfile
import unittest
from pathlib import Path

from arc_r2.hcm32_object_bridge import HCM32ObjectDSLBridge
from arc_r2.object_dsl import ObjectInstruction, ObjectProgram


class HCM32ObjectDSLBridgeTests(unittest.TestCase):
    def test_object_dsl_round_trip_reload_and_cross_grid_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            source = (
                (0, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 0, 0),
            )
            expected = (
                (0, 0, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 0),
            )
            program = ObjectProgram((
                ObjectInstruction("MOVE", 0, delta_row=0, delta_col=1),
            ))

            bridge = HCM32ObjectDSLBridge(path)
            procedure_id = bridge.store_program(program, ((source, expected),))
            self.assertIsNotNone(procedure_id)

            # Reload in a fresh bridge/runtime: only declarative DSL data persisted.
            restored = HCM32ObjectDSLBridge(path)
            match = restored.find_program(((source, expected),))
            self.assertIsNotNone(match)
            self.assertEqual(match[0], procedure_id)
            self.assertEqual(match[1].canonical(), program.canonical())

            # Transfer the learned relative operation to a different grid,
            # with different dimensions, position, and object color.
            unseen = (
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 3, 0, 0),
                (0, 0, 0, 0, 0),
            )
            unseen_expected = (
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 3, 0),
                (0, 0, 0, 0, 0),
            )
            predicted = restored.runtime.apply_procedure(procedure_id, unseen)
            self.assertEqual(predicted, unseen_expected)
            self.assertEqual(restored.memory.data["procedures"][procedure_id]["status"], "verified")

    def test_composed_program_persists_transfers_and_inverts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            source = (
                (0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
            )
            expected = (
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
            )
            program = ObjectProgram((
                ObjectInstruction("MOVE", 0, delta_col=1),
                ObjectInstruction("MOVE", 0, delta_row=1),
            ))
            self.assertEqual(program.inverse().apply(expected), source)

            bridge = HCM32ObjectDSLBridge(path)
            procedure_id = bridge.store_program(program, ((source, expected),))
            self.assertIsNotNone(procedure_id)

            restored = HCM32ObjectDSLBridge(path)
            match = restored.find_program(((source, expected),))
            self.assertIsNotNone(match)
            self.assertEqual(match[1].canonical(), program.canonical())

            unseen = (
                (0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
                (0, 0, 2, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
            )
            unseen_expected = (
                (0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0),
                (0, 0, 0, 2, 0, 0),
                (0, 0, 0, 0, 0, 0),
            )
            predicted = restored.runtime.apply_procedure(procedure_id, unseen)
            self.assertEqual(predicted, unseen_expected)
            self.assertEqual(match[1].inverse().apply(predicted), unseen)

    def test_inverse_rejects_operations_without_a_general_inverse(self):
        for instruction in (
            ObjectInstruction("SCALE", 0, factor=2),
            ObjectInstruction("PLACE", 0, top=1, left=1),
        ):
            with self.subTest(operation=instruction.operation):
                with self.assertRaises(ValueError):
                    instruction.inverse()

    def test_does_not_store_program_that_fails_training_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = HCM32ObjectDSLBridge(Path(directory) / "memory.json")
            source = ((0, 0, 0), (0, 1, 0), (0, 0, 0))
            wrong_target = ((0, 0, 0), (1, 0, 0), (0, 0, 0))
            program = ObjectProgram((
                ObjectInstruction("MOVE", 0, delta_row=0, delta_col=1),
            ))
            self.assertIsNone(bridge.store_program(program, ((source, wrong_target),)))
            self.assertEqual(bridge.memory.data["procedures"], {})


if __name__ == "__main__":
    unittest.main()
