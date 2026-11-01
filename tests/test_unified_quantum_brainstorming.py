import unittest

from arc_r2.object_solver import ObjectExample
from arc_r2.unified_quantum_brainstorming import unified_qbs_search


class UnifiedQBSTests(unittest.TestCase):
    def test_unified_qbs_discovers_object_dsl_program_from_examples(self):
        examples = (
            ObjectExample(((1, 0, 0), (0, 0, 0)),
                          ((0, 1, 0), (0, 0, 0))),
            ObjectExample(((0, 0, 0), (0, 2, 0)),
                          ((0, 0, 0), (0, 0, 2))),
        )
        result = unified_qbs_search(
            examples, max_depth=2, candidate_cap=500,
            parameter_cap=200, max_offset=1, rounds=10,
        )
        self.assertEqual(result.exact_examples, 2)
        self.assertEqual(result.total_examples, 2)
        self.assertEqual(result.fit_rate, 1.0)
        self.assertEqual(result.feasibility, 1.0)
        self.assertEqual(len(result.retained_hypotheses), 4)
        for example in examples:
            self.assertEqual(result.program.apply(example.input), example.output)

    def test_unified_qbs_executes_grid_dsl_operation(self):
        examples = (
            ObjectExample(((1, 0, 0), (0, 0, 0)),
                          ((0, 0, 1), (0, 0, 0))),
            ObjectExample(((0, 2, 0), (0, 0, 0)),
                          ((0, 2, 0), (0, 0, 0))),
        )
        result = unified_qbs_search(
            examples, max_depth=1, candidate_cap=200,
            parameter_cap=100, max_offset=1,
        )
        self.assertEqual(result.exact_examples, 2)
        for example in examples:
            self.assertEqual(result.program.apply(example.input), example.output)


if __name__ == "__main__":
    unittest.main()
