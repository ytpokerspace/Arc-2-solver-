"""HCM-32 Experiment 002: evidence-based selection between stored procedures.

The test runtime explicitly registers two reflection operators. This checks
selection from a new demonstrated pair, not autonomous rule discovery.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from arc_r2.hcm32 import HCM32Memory


def fingerprint(grid):
    payload = json.dumps(grid, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def reflect(grid, axis):
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("expected a non-empty rectangular grid")
    if axis == "left-right":
        return tuple(tuple(reversed(row)) for row in grid)
    if axis == "top-bottom":
        return tuple(reversed(tuple(tuple(row) for row in grid)))
    raise ValueError("unregistered reflection axis")


def execute(procedure, grid):
    result = tuple(tuple(row) for row in grid)
    for operation in procedure["operations"]:
        if operation["operator"] != "REFLECT":
            raise ValueError("unregistered DSL operator")
        result = reflect(result, operation["parameters"].get("axis"))
    return result


def select_by_demonstration(procedures, demo_input, demo_output):
    """Return the sole exact-fit verified procedure; abstain on 0 or >1 fits."""
    matches = []
    for procedure in procedures:
        if procedure["status"] != "verified":
            continue
        try:
            if execute(procedure, demo_input) == demo_output:
                matches.append(procedure)
        except (KeyError, TypeError, ValueError):
            continue
    return matches[0] if len(matches) == 1 else None


class HCM32ProcedureSelectionExperiment(unittest.TestCase):
    def test_selects_correct_persisted_rule_and_transfers_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hcm32.json"
            memory = HCM32Memory(path)

            left_right = memory.add_procedure(
                "PR-REFLECT-LR", "left_right_reflection",
                [{"operator": "REFLECT", "parameters": {"axis": "left-right"}}],
                input_constraints=["non_empty_rectangular_grid"],
                inverse_procedure_id="PR-REFLECT-LR",
            )
            top_bottom = memory.add_procedure(
                "PR-REFLECT-TB", "top_bottom_reflection",
                [{"operator": "REFLECT", "parameters": {"axis": "top-bottom"}}],
                input_constraints=["non_empty_rectangular_grid"],
                inverse_procedure_id="PR-REFLECT-TB",
            )

            # Independently establish both procedures before selection.
            fixtures = (
                ("PR-REFLECT-LR", ((1, 2, 3), (4, 5, 6)),
                 ((3, 2, 1), (6, 5, 4)), left_right),
                ("PR-REFLECT-TB", ((1, 2, 3), (4, 5, 6)),
                 ((4, 5, 6), (1, 2, 3)), top_bottom),
            )
            for evidence_id, (procedure_id, source, expected, procedure) in zip(
                ("EV-LR-BASELINE", "EV-TB-BASELINE"), fixtures
            ):
                actual = execute(procedure, source)
                self.assertEqual(actual, expected)
                memory.add_evidence(
                    evidence_id, "procedure", procedure_id, "training_fit", "passed",
                    input_fingerprint=fingerprint(source),
                    expected_fingerprint=fingerprint(expected),
                    actual_fingerprint=fingerprint(actual),
                    details={"fixture": "independent_baseline"},
                )
                memory.set_procedure_status(procedure_id, "verified")

            memory.add_node(
                "KN-REFLECTION-FAMILY", "transformation_family",
                "row and column reflection", ["Q22.S02", "Q30.S03"],
                procedure_refs=["PR-REFLECT-LR", "PR-REFLECT-TB"],
            )
            memory.save()

            # Fresh instance proves both candidate rules survive persistence.
            restored = HCM32Memory(path)
            candidates = restored.procedures_for_node("KN-REFLECTION-FAMILY")
            demo_input = (
                (9, 2, 5, 1),
                (3, 8, 4, 7),
                (6, 0, 2, 5),
            )
            demo_output = (
                (6, 0, 2, 5),
                (3, 8, 4, 7),
                (9, 2, 5, 1),
            )
            selected = select_by_demonstration(candidates, demo_input, demo_output)
            self.assertIsNotNone(selected)
            self.assertEqual(selected["procedure_id"], "PR-REFLECT-TB")

            # Apply selected procedure to a separate held-out grid.
            held_out = ((1, 4), (7, 2), (8, 5), (3, 9))
            expected = ((3, 9), (8, 5), (7, 2), (1, 4))
            actual = execute(selected, held_out)
            self.assertEqual(actual, expected)
            self.assertEqual(execute(selected, actual), held_out)

            restored.add_evidence(
                "EV-TB-TRANSFER", "procedure", "PR-REFLECT-TB",
                "transfer_validation", "passed",
                input_fingerprint=fingerprint(held_out),
                expected_fingerprint=fingerprint(expected),
                actual_fingerprint=fingerprint(actual),
                details={
                    "selected_from_demonstration": True,
                    "selected_procedure": selected["procedure_id"],
                    "other_verified_candidate": "PR-REFLECT-LR",
                    "inverse_round_trip_passed": True,
                },
            )
            restored.record_application(
                "PR-REFLECT-TB", "passed", task_ref="synthetic-held-out-task"
            )
            restored.save()

            final_memory = HCM32Memory(path)
            final_proc = final_memory.data["procedures"]["PR-REFLECT-TB"]
            self.assertEqual(final_proc["reuse_successes"], 1)
            self.assertEqual(
                final_memory.data["evidence"]["EV-TB-TRANSFER"]["result"], "passed"
            )

    def test_ambiguous_demonstration_causes_abstention(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = HCM32Memory(Path(directory) / "hcm32.json")
            procedures = []
            for procedure_id, axis, evidence_id in (
                ("PR-LR", "left-right", "EV-LR"),
                ("PR-TB", "top-bottom", "EV-TB"),
            ):
                procedure = memory.add_procedure(
                    procedure_id, procedure_id,
                    [{"operator": "REFLECT", "parameters": {"axis": axis}}],
                )
                memory.add_evidence(
                    evidence_id, "procedure", procedure_id, "baseline_fit", "passed"
                )
                memory.set_procedure_status(procedure_id, "verified")
                procedures.append(procedure)

            symmetric = ((1, 2), (2, 1))
            same_output = ((2, 1), (1, 2))
            self.assertIsNone(
                select_by_demonstration(procedures, symmetric, same_output)
            )


if __name__ == "__main__":
    unittest.main()
