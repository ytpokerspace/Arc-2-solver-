"""Integration tests for HCM-32 runtime selection and held-out transfer."""
import tempfile
import unittest
from pathlib import Path

from arc_r2.hcm32 import HCM32Memory, HCM32Runtime


class HCM32ProcedureSelectionExperiment(unittest.TestCase):
    def _build_memory(self, path):
        memory = HCM32Memory(path)
        definitions = (
            ("PR-REFLECT-LR", "left_right_reflection",
             [{"operator": "REFLECT", "parameters": {"axis": "left-right"}}]),
            ("PR-REFLECT-TB", "top_bottom_reflection",
             [{"operator": "REFLECT", "parameters": {"axis": "top-bottom"}}]),
            ("PR-ROTATE-90", "rotate_clockwise",
             [{"operator": "ROTATE", "parameters": {"quarter_turns": 1}}]),
            ("PR-TRANSPOSE", "transpose",
             [{"operator": "TRANSPOSE", "parameters": {}}]),
        )
        baseline = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
        expected = {
            "PR-REFLECT-LR": ((3, 2, 1), (6, 5, 4), (9, 8, 7)),
            "PR-REFLECT-TB": ((7, 8, 9), (4, 5, 6), (1, 2, 3)),
            "PR-ROTATE-90": ((7, 4, 1), (8, 5, 2), (9, 6, 3)),
            "PR-TRANSPOSE": ((1, 4, 7), (2, 5, 8), (3, 6, 9)),
        }
        for procedure_id, name, operations in definitions:
            memory.add_procedure(
                procedure_id, name, operations,
                input_constraints=["non_empty_rectangular_grid"],
            )
            memory.add_evidence(
                "EV-BASE-" + procedure_id, "procedure", procedure_id,
                "independent_baseline_fit", "passed",
            )
            memory.set_procedure_status(procedure_id, "verified")
        memory.add_node(
            "KN-GRID-TRANSFORMS", "transformation_family",
            "verified grid transformations", ["Q22.S02", "Q30.S03"],
            procedure_refs=[item[0] for item in definitions],
        )
        memory.save()
        return baseline, expected

    def test_runtime_selects_four_rules_and_transfers_to_new_grids_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hcm32.json"
            baseline, demonstrations = self._build_memory(path)

            # A fresh object is the restart boundary.
            restored = HCM32Memory(path)
            runtime = HCM32Runtime(restored)
            held_out = ((1, 2, 3), (4, 5, 6))
            held_out_expected = {
                "PR-REFLECT-LR": ((3, 2, 1), (6, 5, 4)),
                "PR-REFLECT-TB": ((4, 5, 6), (1, 2, 3)),
                "PR-ROTATE-90": ((4, 1), (5, 2), (6, 3)),
                "PR-TRANSPOSE": ((1, 4), (2, 5), (3, 6)),
            }
            for procedure_id, demo_output in demonstrations.items():
                selected = runtime.select_procedure(
                    "KN-GRID-TRANSFORMS", baseline, demo_output
                )
                self.assertIsNotNone(selected)
                self.assertEqual(selected["procedure_id"], procedure_id)
                actual = runtime.apply_procedure(selected["procedure_id"], held_out)
                self.assertEqual(actual, held_out_expected[procedure_id])
                restored.add_evidence(
                    "EV-TRANSFER-" + procedure_id, "procedure", procedure_id,
                    "held_out_transfer", "passed",
                    details={"selected_by_runtime": True, "after_reload": True},
                )
                restored.record_application(procedure_id, "passed", task_ref="held-out-grid")
            restored.save()

            final_memory = HCM32Memory(path)
            for procedure_id in demonstrations:
                self.assertEqual(
                    final_memory.data["procedures"][procedure_id]["reuse_successes"], 1
                )
                self.assertIn(
                    "EV-TRANSFER-" + procedure_id,
                    final_memory.data["procedures"][procedure_id]["evidence_refs"],
                )

    def test_ambiguous_demo_abstains_and_unverified_procedure_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hcm32.json"
            self._build_memory(path)
            memory = HCM32Memory(path)
            runtime = HCM32Runtime(memory)

            # Three distinct verified operations produce the same output here.
            symmetric = ((1, 2), (2, 1))
            ambiguous_output = ((2, 1), (1, 2))
            self.assertIsNone(
                runtime.select_procedure(
                    "KN-GRID-TRANSFORMS", symmetric, ambiguous_output
                )
            )

            # A draft procedure must not participate in selection or execution.
            memory.add_procedure(
                "PR-DRAFT", "unverified draft",
                [{"operator": "REFLECT", "parameters": {"axis": "left-right"}}],
            )
            memory.data["nodes"]["KN-GRID-TRANSFORMS"]["procedure_refs"].append("PR-DRAFT")
            self.assertIsNone(
                runtime.select_procedure(
                    "KN-GRID-TRANSFORMS", ((1, 2, 3),), ((3, 2, 1),)
                )
            )
            with self.assertRaises(ValueError):
                runtime.apply_procedure("PR-DRAFT", ((1, 2),))

    def test_runtime_rejects_bad_grids_unknown_operators_and_bad_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hcm32.json"
            self._build_memory(path)
            memory = HCM32Memory(path)
            runtime = HCM32Runtime(memory)
            with self.assertRaises(ValueError):
                runtime.apply_procedure("PR-REFLECT-LR", ((1, 2), (3,)))
            with self.assertRaises(ValueError):
                runtime.register_operator("REFLECT", lambda grid, params: grid)

            memory.add_procedure(
                "PR-UNKNOWN", "unknown operator",
                [{"operator": "PYTHON_EVAL", "parameters": {"code": "unsafe"}}],
            )
            memory.add_evidence(
                "EV-UNKNOWN", "procedure", "PR-UNKNOWN", "baseline_fit", "passed"
            )
            memory.set_procedure_status("PR-UNKNOWN", "verified")
            with self.assertRaises(ValueError):
                runtime.apply_procedure("PR-UNKNOWN", ((1, 2),))

            memory.add_procedure(
                "PR-BAD-ROTATE", "bad rotation parameter",
                [{"operator": "ROTATE", "parameters": {"quarter_turns": 9}}],
            )
            memory.add_evidence(
                "EV-BAD-ROTATE", "procedure", "PR-BAD-ROTATE", "baseline_fit", "passed"
            )
            memory.set_procedure_status("PR-BAD-ROTATE", "verified")
            with self.assertRaises(ValueError):
                runtime.apply_procedure("PR-BAD-ROTATE", ((1, 2),))


if __name__ == "__main__":
    unittest.main()
