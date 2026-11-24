"""HCM-32 Experiment 001: persisted reflection-procedure transfer.

This is a controlled DSL-memory test, not an ARC benchmark result. The runtime
operator is explicitly registered by the test; HCM-32 stores its declarative
procedure and evidence, then retrieves it after a fresh memory load.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from arc_r2.hcm32 import HCM32Memory


def reflect_left_right(grid):
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("expected a non-empty rectangular grid")
    return tuple(tuple(reversed(row)) for row in grid)


def fingerprint(grid):
    payload = json.dumps(grid, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def execute_registered_procedure(procedure, grid):
    """Execute only the explicitly registered REFLECT operator."""
    result = tuple(tuple(row) for row in grid)
    for operation in procedure["operations"]:
        if operation["operator"] != "REFLECT":
            raise ValueError("unregistered DSL operator")
        params = operation["parameters"]
        if params != {"axis": "left-right"}:
            raise ValueError("unsupported reflection parameters")
        result = reflect_left_right(result)
    return result


class HCM32ReflectionTransferExperiment(unittest.TestCase):
    def test_learn_persist_reload_and_transfer_without_rediscovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hcm32.json"

            # Task A: learn a candidate from one demonstrated pair.
            task_a_input = (
                (1, 0, 0),
                (2, 3, 0),
            )
            task_a_output = ((0, 0, 1), (0, 3, 2))
            memory = HCM32Memory(path)
            memory.add_node(
                "KN-REFLECT-LR",
                "transformation_rule",
                "left-right reflection",
                ["Q22.S02", "Q30.S03"],
                description="Reverse the cell order in every row.",
            )
            memory.add_procedure(
                "PR-REFLECT-LR",
                "left_right_reflection",
                [{"operator": "REFLECT", "parameters": {"axis": "left-right"}}],
                input_constraints=["non_empty_rectangular_grid"],
                inverse_procedure_id="PR-REFLECT-LR",
            )
            memory.data["nodes"]["KN-REFLECT-LR"]["procedure_refs"].append(
                "PR-REFLECT-LR"
            )

            learned = memory.data["procedures"]["PR-REFLECT-LR"]
            self.assertEqual(
                execute_registered_procedure(learned, task_a_input),
                task_a_output,
            )
            memory.add_evidence(
                "EV-REFLECT-LEARN",
                "procedure",
                "PR-REFLECT-LR",
                "training_fit",
                "passed",
                input_fingerprint=fingerprint(task_a_input),
                expected_fingerprint=fingerprint(task_a_output),
                actual_fingerprint=fingerprint(
                    execute_registered_procedure(learned, task_a_input)
                ),
                details={"task": "A", "note": "initial demonstration"},
            )
            memory.set_procedure_status("PR-REFLECT-LR", "verified")
            memory.save()

            # Simulate process restart: construct a new object from disk.
            reloaded = HCM32Memory(path)
            node = reloaded.data["nodes"]["KN-REFLECT-LR"]
            retrieved_ids = node["procedure_refs"]
            self.assertEqual(retrieved_ids, ["PR-REFLECT-LR"])
            recalled = reloaded.data["procedures"][retrieved_ids[0]]
            self.assertEqual(recalled["status"], "verified")

            # Task B is structurally different and was not used for learning.
            task_b_input = (
                (4, 5, 6, 7),
                (8, 0, 9, 1),
                (2, 3, 4, 5),
            )
            expected_b = ((7, 6, 5, 4), (1, 9, 0, 8), (5, 4, 3, 2))

            # Preconditions are checked before execution.
            self.assertTrue(task_b_input and task_b_input[0])
            self.assertTrue(all(len(row) == len(task_b_input[0]) for row in task_b_input))

            actual_b = execute_registered_procedure(recalled, task_b_input)
            self.assertEqual(actual_b, expected_b)

            # Reflection is self-inverse: round-trip restores Task B input.
            restored_b = execute_registered_procedure(recalled, actual_b)
            self.assertEqual(restored_b, task_b_input)

            # Record transfer evidence only after the actual comparison passes.
            reloaded.add_evidence(
                "EV-REFLECT-TRANSFER-B",
                "procedure",
                "PR-REFLECT-LR",
                "transfer_validation",
                "passed" if actual_b == expected_b else "failed",
                input_fingerprint=fingerprint(task_b_input),
                expected_fingerprint=fingerprint(expected_b),
                actual_fingerprint=fingerprint(actual_b),
                details={
                    "task": "B",
                    "retrieved_after_reload": True,
                    "rediscovery_performed": False,
                    "inverse_round_trip_passed": restored_b == task_b_input,
                },
            )
            reloaded.record_application(
                "PR-REFLECT-LR", "passed", task_ref="synthetic-transfer-task-B"
            )
            reloaded.save()

            final_memory = HCM32Memory(path)
            final_procedure = final_memory.data["procedures"]["PR-REFLECT-LR"]
            self.assertEqual(final_procedure["reuse_successes"], 1)
            transfer_evidence = final_memory.data["evidence"]["EV-REFLECT-TRANSFER-B"]
            self.assertEqual(transfer_evidence["result"], "passed")
            self.assertTrue(transfer_evidence["details"]["retrieved_after_reload"])
            self.assertFalse(transfer_evidence["details"]["rediscovery_performed"])

    def test_invalid_grid_precondition_abstains(self):
        with self.assertRaises(ValueError):
            reflect_left_right(((1, 2), (3,)))


if __name__ == "__main__":
    unittest.main()
