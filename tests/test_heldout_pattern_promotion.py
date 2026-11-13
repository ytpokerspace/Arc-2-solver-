import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from arc_r2.dsl import parse_program
from arc_r2.grid import as_grid
from arc_r2.reasoning_layer import validate_and_promote_pattern
from arc_r2.reasoning_memory import ReasoningMemory
from arc_r2.solver import Example


class FixedSolver:
    def __init__(self, program):
        self.program = program

    def solve(self, examples):
        return SimpleNamespace(program=self.program)


class HeldoutPromotionTests(unittest.TestCase):
    def test_promotes_only_when_same_program_passes_every_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = ReasoningMemory(Path(directory) / "memory.json")
            record = memory.put({
                "id": "candidate",
                "title": "Candidate flip",
                "intent": "Test a horizontal flip",
                "status": "tested",
                "tags": ["arc2", "baseline"],
                "outcome": {"program": "REVERSE.HORIZONTAL"},
            })
            program = parse_program("REVERSE.HORIZONTAL")
            examples = (
                Example(as_grid([[1, 2]]), as_grid([[2, 1]])),
                Example(as_grid([[3, 4, 5]]), as_grid([[5, 4, 3]])),
            )
            promoted = validate_and_promote_pattern(
                memory, record, SimpleNamespace(program=program), examples,
                solver_factory=lambda: FixedSolver(program),
            )
            self.assertEqual(promoted["status"], "validated")
            self.assertEqual(promoted["outcome"]["validation_exact"], 2)
            self.assertEqual(len(memory.verified_patterns(["arc2", "baseline"])), 1)

    def test_does_not_promote_if_any_heldout_pair_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = ReasoningMemory(Path(directory) / "memory.json")
            record = memory.put({
                "id": "candidate",
                "title": "Candidate flip",
                "intent": "Test a horizontal flip",
                "status": "tested",
                "tags": ["arc2", "baseline"],
                "outcome": {"program": "REVERSE.HORIZONTAL"},
            })
            program = parse_program("REVERSE.HORIZONTAL")
            examples = (
                Example(as_grid([[1, 2]]), as_grid([[2, 1]])),
                Example(as_grid([[3, 4, 5]]), as_grid([[3, 4, 5]])),
            )
            result = validate_and_promote_pattern(
                memory, record, SimpleNamespace(program=program), examples,
                solver_factory=lambda: FixedSolver(program),
            )
            self.assertEqual(result["status"], "tested")
            self.assertEqual(memory.verified_patterns(["arc2", "baseline"]), [])


if __name__ == "__main__":
    unittest.main()
