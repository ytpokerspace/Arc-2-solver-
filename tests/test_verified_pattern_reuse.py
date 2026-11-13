import tempfile
import unittest
from pathlib import Path

from arc_r2.grid import as_grid
from arc_r2.reasoning_layer import find_applicable_verified_pattern
from arc_r2.reasoning_memory import ReasoningMemory
from arc_r2.solver import Example


class VerifiedPatternReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.memory = ReasoningMemory(Path(self.temp.name) / "memory.json")
        self.memory.put({
            "id": "validated-horizontal-flip",
            "title": "Validated horizontal flip",
            "intent": "Reuse a verified horizontal flip transformation",
            "status": "validated",
            "tags": ["arc2", "baseline"],
            "outcome": {"program": "REVERSE.HORIZONTAL"},
        })

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_fit_validated_pattern_is_reused(self):
        examples = (
            Example(as_grid([[1, 0, 2]]), as_grid([[2, 0, 1]])),
            Example(as_grid([[3, 4]]), as_grid([[4, 3]])),
        )
        result, record = find_applicable_verified_pattern(
            self.memory, examples, tags=["arc2", "baseline"]
        )
        self.assertIsNotNone(result)
        self.assertEqual(record["status"], "validated")
        self.assertEqual(result.exact_examples, 2)
        self.assertEqual(result.candidates_evaluated, 0)
        self.assertEqual(result.program.apply(examples[0].input), examples[0].output)

    def test_pattern_is_rejected_when_any_example_does_not_fit(self):
        examples = (
            Example(as_grid([[1, 0, 2]]), as_grid([[2, 0, 1]])),
            Example(as_grid([[3, 4]]), as_grid([[3, 4]])),
        )
        result, record = find_applicable_verified_pattern(
            self.memory, examples, tags=["arc2", "baseline"]
        )
        self.assertIsNone(result)
        self.assertIsNone(record)

    def test_unvalidated_pattern_is_never_reused(self):
        self.memory.put({
            "id": "merely-tested",
            "title": "Only training fit",
            "intent": "Must not be reused as verified",
            "status": "tested",
            "tags": ["arc2", "baseline"],
            "outcome": {"program": "REVERSE.HORIZONTAL"},
        })
        examples = (
            Example(as_grid([[1, 0, 2]]), as_grid([[2, 0, 1]])),
        )
        result, record = find_applicable_verified_pattern(
            self.memory, examples, tags=["arc2", "baseline"]
        )
        self.assertIsNotNone(result)
        self.assertEqual(record["id"], "validated-horizontal-flip")


if __name__ == "__main__":
    unittest.main()
