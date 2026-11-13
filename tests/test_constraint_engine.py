import unittest

from arc_r2.constraint_engine import check_prediction, check_program_on_examples
from arc_r2.primitives import Grid, Program, Primitive, as_grid


class PredictionConstraintTests(unittest.TestCase):
    def test_shape_constraint_rejects_before_cell_comparison(self):
        result = check_prediction(as_grid([[1, 2]]), as_grid([[1], [2]]))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "shape")
        self.assertFalse(result.exact_comparison_performed)

    def test_palette_constraint_rejects_before_cell_comparison(self):
        result = check_prediction(as_grid([[1, 1]]), as_grid([[1, 2]]))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "palette")
        self.assertFalse(result.exact_comparison_performed)

    def test_color_count_constraint_rejects_before_cell_comparison(self):
        result = check_prediction(as_grid([[1, 2], [1, 2]]), as_grid([[1, 1], [2, 2]]))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "color_counts")
        self.assertFalse(result.exact_comparison_performed)

    def test_cell_position_constraint_requires_exact_comparison(self):
        result = check_prediction(as_grid([[1, 2], [2, 1]]), as_grid([[2, 1], [1, 2]]))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "cell_positions")
        self.assertTrue(result.exact_comparison_performed)

    def test_exact_match_accepted(self):
        grid = as_grid([[1, 2], [2, 1]])
        result = check_prediction(grid, grid)
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "exact_match")

    def test_program_checks_all_examples(self):
        examples = (
            (as_grid([[1, 2]]), as_grid([[2, 1]])),
            (as_grid([[3, 4]]), as_grid([[4, 3]])),
        )
        accepted, exact_checks, counters = check_program_on_examples(
            Program((Primitive("FLIP_H"),)), examples
        )
        self.assertTrue(accepted)
        self.assertEqual(counters["exact_match"], 2)
        self.assertEqual(exact_checks, 2)


if __name__ == "__main__":
    unittest.main()
