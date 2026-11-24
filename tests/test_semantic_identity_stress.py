"""Regression tests for the semantic identity stress matrix."""
import unittest

from arc_r2.semantic_identity_stress import run_stress_matrix


class SemanticIdentityStressTests(unittest.TestCase):
    def test_stress_matrix_has_four_scenarios(self):
        report = run_stress_matrix()
        self.assertEqual(report["scenario_count"], 4)
        self.assertFalse(report["arc_task_accuracy_claim"])

    def test_far_relocation_rotation_and_order_change(self):
        case = run_stress_matrix()["scenarios"]["far_relocation_rotation_order_change"]
        self.assertEqual(case["correct"], 3)
        self.assertEqual(case["incorrect_assignments"], 0)
        self.assertEqual(case["abstained"], 0)

    def test_repeated_identical_objects_abstain(self):
        case = run_stress_matrix()["scenarios"]["repeated_identical_objects"]
        self.assertEqual(case["assigned"], 0)
        self.assertEqual(case["expected_ambiguous"], 2)
        self.assertEqual(len(case["ambiguous_reference"]), 2)
        self.assertEqual(case["incorrect_assignments"], 0)

    def test_missing_object_is_left_unmatched(self):
        case = run_stress_matrix()["scenarios"]["one_missing_object"]
        self.assertEqual(case["correct"], 2)
        self.assertEqual(case["expected_unmatched_reference"], 1)
        self.assertEqual(case["incorrect_assignments"], 0)

    def test_new_object_is_left_unmatched(self):
        case = run_stress_matrix()["scenarios"]["one_new_object"]
        self.assertEqual(case["correct"], 2)
        self.assertEqual(len(case["unmatched_target"]), 1)
        self.assertEqual(case["incorrect_assignments"], 0)


if __name__ == "__main__":
    unittest.main()
