import unittest

from arc_r2.search import SearchLimits
from arc_r2.solver import Example, HybridSolver


class QuantumBrainstormingSolverTests(unittest.TestCase):
    def test_qbs_discovers_a_two_operation_hybrid(self):
        # Horizontal reflection followed by foreground/background polarity swap.
        examples = (
            Example(
                ((1, 0, 0), (0, 2, 0)),
                ((1, 1, 0), (1, 2, 1)),
            ),
            Example(
                ((0, 3, 0), (4, 0, 0)),
                ((1, 3, 1), (1, 1, 4)),
            ),
        )
        solver = HybridSolver(
            SearchLimits(
                max_depth=2,
                candidate_cap=500,
                mcts_iterations=0,
            ),
            strategy="qbs",
        )

        result = solver.solve(examples)

        self.assertEqual(result.exact_examples, 2)
        self.assertEqual(result.total_examples, 2)
        self.assertGreater(result.qbs_rounds, 0)
        for example in examples:
            self.assertEqual(result.program.apply(example.input), example.output)

    def test_qbs_rejects_empty_examples(self):
        solver = HybridSolver(strategy="qbs")
        with self.assertRaises(ValueError):
            solver.solve(())


if __name__ == "__main__":
    unittest.main()
