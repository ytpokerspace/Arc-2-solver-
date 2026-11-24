"""Tests for seeded randomized semantic identity stress."""
import unittest

from arc_r2.semantic_identity_randomized import run_randomized_stress


class SemanticIdentityRandomizedTests(unittest.TestCase):
    def test_seeded_run_is_reproducible(self):
        first = run_randomized_stress(seed=12345, trials=20)
        second = run_randomized_stress(seed=12345, trials=20)
        self.assertEqual(first["result_sha256"], second["result_sha256"])
        self.assertEqual(first["trial_records"], second["trial_records"])

    def test_randomized_known_correspondences_pass(self):
        report = run_randomized_stress(seed=20261124, trials=40)
        self.assertTrue(report["passed"], report["failure_examples"])
        self.assertEqual(report["incorrect_assignments"], 0)
        self.assertEqual(report["abstentions"], 0)
        self.assertEqual(report["correct_matches"], report["expected_matches"])
        self.assertFalse(report["arc_task_accuracy_claim"])

    def test_trial_count_is_bounded(self):
        with self.assertRaises(ValueError):
            run_randomized_stress(seed=1, trials=0)
        with self.assertRaises(ValueError):
            run_randomized_stress(seed=1, trials=10001)


if __name__ == "__main__":
    unittest.main()
