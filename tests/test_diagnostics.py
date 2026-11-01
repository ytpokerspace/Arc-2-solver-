import unittest

from arc_r2.diagnostics import diagnose_candidates
from arc_r2.dsl import Program
from arc_r2.transforms import (
    negate_polarity,
    reverse_horizontal,
    reverse_vertical,
    rotate_180,
)


class R2DiagnosticTests(unittest.TestCase):
    def test_sanity_identity_and_core_grid_transforms(self):
        grid = ((1, 0, 0), (0, 2, 0), (0, 0, 0))
        self.assertEqual(Program().apply(grid), grid)
        self.assertEqual(
            reverse_horizontal(grid),
            ((0, 0, 1), (0, 2, 0), (0, 0, 0)),
        )
        self.assertEqual(
            reverse_vertical(grid),
            ((0, 0, 0), (0, 2, 0), (1, 0, 0)),
        )
        self.assertEqual(
            rotate_180(grid),
            ((0, 0, 0), (0, 2, 0), (0, 0, 1)),
        )
        self.assertEqual(
            negate_polarity(((1, 0), (0, 0))),
            ((0, 1), (1, 1)),
        )

    def test_candidate_coverage_is_separate_from_ranking(self):
        source = [[1, 0], [0, 0]]
        target = [[0, 1], [0, 0]]
        report = diagnose_candidates(
            "baseline",
            ({"input": source, "output": source},),
            {"input": source, "output": target},
            max_depth=1,
            candidate_cap=30,
        )
        self.assertGreater(report["held_out_coverage_candidates"], 0)
        self.assertGreater(report["train_fit_candidates"], 0)

    def test_invalid_object_candidates_are_counted_not_fatal(self):
        source = [[1, 0], [0, 0]]
        report = diagnose_candidates(
            "object",
            ({"input": source, "output": source},),
            {"input": source, "output": source},
            max_depth=1,
            candidate_cap=120,
        )
        self.assertGreater(report["rejected_invalid_candidates"], 0)
        self.assertGreaterEqual(report["unique_behaviors"], 1)
        self.assertGreaterEqual(report["duplicate_behaviors_removed"], 0)


if __name__ == "__main__":
    unittest.main()
