import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from arc_r2.reasoning_layer import solve_and_remember
from arc_r2.reasoning_memory import ReasoningMemory


class FakeSolver:
    def __init__(self, exact, total):
        self.exact = exact
        self.total = total

    def solve(self, examples):
        return SimpleNamespace(
            exact_examples=self.exact,
            total_examples=self.total,
            score=self.exact / self.total,
            candidates_evaluated=7,
            program="identity",
        )


class ReasoningLayerTests(unittest.TestCase):
    def test_solver_run_is_recorded_without_overclaiming_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = ReasoningMemory(Path(directory) / "reasoning.json")
            result, record = solve_and_remember(
                FakeSolver(2, 2),
                examples=("example-a", "example-b"),
                memory=memory,
                record_id="task-001-run-001",
                intent="Infer a transformation from ARC training pairs",
                requirements=["exact output equality"],
                representations=["grid", "object graph"],
                hypotheses=["candidate transformation"],
                tags=["arc2", "geometry"],
            )
            self.assertEqual(result.exact_examples, 2)
            self.assertEqual(record["status"], "tested")
            self.assertEqual(record["outcome"]["evidence_scope"],
                             "visible training examples only; not held-out validation")
            self.assertEqual(memory.verified_patterns(["arc2"]), [])

    def test_partial_fit_is_not_marked_tested(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = ReasoningMemory(Path(directory) / "reasoning.json")
            _, record = solve_and_remember(
                FakeSolver(1, 2), ("a", "b"), memory,
                record_id="partial", intent="Test a candidate rule",
            )
            self.assertEqual(record["status"], "implemented")
            self.assertFalse(record["outcome"]["training_fit_complete"])


if __name__ == "__main__":
    unittest.main()
