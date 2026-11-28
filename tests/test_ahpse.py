import unittest

from arc_r2.ahpse import TrainingExample, synthesize


class AHPSETests(unittest.TestCase):
    def test_discovers_rotation_and_color_mapping(self):
        examples = (
            TrainingExample.from_values(
                [[0, 1], [2, 0]],
                [[3, 0], [0, 4]],
            ),
            TrainingExample.from_values(
                [[0, 5], [6, 0]],
                [[7, 0], [0, 8]],
            ),
        )
        result = synthesize(examples, max_depth=1)
        self.assertFalse(result.abstained)
        self.assertTrue(any(
            candidate.operations == ("ROT90",)
            and candidate.apply(examples[0].input) == examples[0].output
            for candidate in result.hypotheses
        ))

    def test_rejects_candidate_that_fits_only_first_pair(self):
        examples = (
            TrainingExample.from_values([[1, 0], [0, 2]], [[2, 0], [0, 1]]),
            TrainingExample.from_values([[1, 0], [0, 3]], [[3, 0], [0, 1]]),
        )
        result = synthesize(examples, max_depth=0)
        self.assertTrue(result.abstained)
        self.assertEqual(result.exact_fit_count, 0)

    def test_candidate_budget_is_enforced(self):
        examples = (TrainingExample.from_values([[1]], [[1]]),)
        result = synthesize(examples, max_depth=3, candidate_cap=4)
        self.assertEqual(result.candidates_generated, 4)

    def test_empty_and_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            synthesize([])
        with self.assertRaises(ValueError):
            synthesize((TrainingExample.from_values([[1]], [[1]]),),
                       candidate_cap=0)

    def test_ambiguous_exact_fits_are_preserved(self):
        examples = (TrainingExample.from_values([[1, 1], [1, 1]],
                                                [[2, 2], [2, 2]]),)
        result = synthesize(examples, max_depth=1)
        self.assertGreaterEqual(result.exact_fit_count, 1)
        # The engine reports all exact fits; it does not claim ambiguity is certainty.
        self.assertEqual(result.best.apply(examples[0].input), examples[0].output)


if __name__ == "__main__":
    unittest.main()
