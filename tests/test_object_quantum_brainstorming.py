import unittest

from arc_r2.object_dsl import ObjectProgram
from arc_r2.object_quantum_brainstorming import object_quantum_brainstorm_search
from arc_r2.object_solver import ObjectExample


class ObjectQuantumBrainstormingTests(unittest.TestCase):
    def test_qbs_object_dsl_discovers_parameterized_move(self):
        examples = (
            ObjectExample(
                ((1, 0, 0), (0, 0, 0)),
                ((0, 1, 0), (0, 0, 0)),
            ),
            ObjectExample(
                ((0, 0, 0), (0, 2, 0)),
                ((0, 0, 0), (0, 0, 2)),
            ),
        )
        result = object_quantum_brainstorm_search(
            examples,
            max_depth=2,
            candidate_cap=500,
            parameter_cap=200,
            max_offset=1,
        )
        self.assertEqual(result.exact_examples, 2)
        self.assertEqual(result.total_examples, 2)
        self.assertEqual(result.fit_rate, 1.0)
        self.assertEqual(result.feasibility, 1.0)
        self.assertGreater(result.candidates_evaluated, 1)
        for example in examples:
            self.assertEqual(result.program.apply(example.input), example.output)

    def test_identity_is_a_valid_object_qbs_hypothesis(self):
        examples = (
            ObjectExample(((1, 0), (0, 0)), ((1, 0), (0, 0))),
        )
        result = object_quantum_brainstorm_search(
            examples, candidate_cap=50, parameter_cap=50, max_offset=1
        )
        self.assertEqual(result.program, ObjectProgram())
        self.assertEqual(result.exact_examples, 1)


if __name__ == "__main__":
    unittest.main()
