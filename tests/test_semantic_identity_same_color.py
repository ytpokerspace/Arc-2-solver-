"""Tests for same-color, shape-based semantic identity matching."""
import unittest

from arc_r2.objects import GridObject
from arc_r2.semantic_identity import _appearance_score
from arc_r2.semantic_identity_same_color import run_same_color_stress


class SemanticIdentitySameColorTests(unittest.TestCase):
    def test_aspect_similarity_is_rotation_invariant(self):
        horizontal = GridObject(
            1,
            frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
            (0, 0, 1, 2),
        )
        rotated = GridObject(
            1,
            frozenset({(0, 1), (1, 0), (1, 1), (2, 1)}),
            (0, 0, 2, 1),
        )
        self.assertAlmostEqual(_appearance_score(horizontal, rotated), 0.9)

    def test_same_color_shape_matching_passes_seeded_trials(self):
        report = run_same_color_stress(seed=20261125, trials=80)
        self.assertTrue(report["passed"], report["failure_examples"])
        self.assertEqual(report["incorrect_assignments"], 0)
        self.assertEqual(report["abstentions"], 0)
        self.assertEqual(report["correct_matches"], report["expected_matches"])
        self.assertFalse(report["arc_task_accuracy_claim"])

    def test_trial_count_is_bounded(self):
        with self.assertRaises(ValueError):
            run_same_color_stress(seed=1, trials=0)
        with self.assertRaises(ValueError):
            run_same_color_stress(seed=1, trials=10001)


if __name__ == "__main__":
    unittest.main()
