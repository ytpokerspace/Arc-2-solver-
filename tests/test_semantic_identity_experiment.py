"""Regression test for the controlled semantic identity preflight."""
import unittest

from arc_r2.semantic_identity_experiment import run_experiment


class SemanticIdentityExperimentTests(unittest.TestCase):
    def test_semantic_matcher_recovers_relocated_reordered_objects(self):
        report = run_experiment()
        self.assertEqual(report["scenario_count"], 1)
        self.assertEqual(report["baseline_index_order"]["correct"], 0)
        self.assertEqual(report["semantic_identity"]["correct"], 3)
        self.assertEqual(report["semantic_identity"]["incorrect_assignments"], 0)
        self.assertEqual(report["semantic_identity"]["abstained"], 0)
        self.assertFalse(report["arc_task_accuracy_claim"])


if __name__ == "__main__":
    unittest.main()
