"""Tests for paired index-order and semantic identity evaluation."""
import unittest

from arc_r2.semantic_identity_arc_fixture import run_training_pair_experiment


class SemanticIdentityArcFixtureTests(unittest.TestCase):
    def test_d89b689b_compares_both_methods_on_identical_training_pairs(self):
        report = run_training_pair_experiment()
        self.assertTrue(report["passed"], report["pair_reports"])
        self.assertEqual(report["training_pairs"], 3)
        self.assertEqual(report["expected_matches"], 12)

        semantic = report["semantic_identity"]
        self.assertEqual(semantic["correct_matches"], 12)
        self.assertEqual(semantic["incorrect_assignments"], 0)
        self.assertEqual(semantic["abstentions"], 0)

        baseline = report["baseline_index_order"]
        self.assertEqual(baseline["expected_matches"], 12)
        self.assertEqual(
            baseline["correct_matches"] + baseline["incorrect_assignments"]
            + baseline["abstentions"],
            12,
        )
        self.assertLess(baseline["correct_matches"], semantic["correct_matches"])
        self.assertEqual(report["unmatched_anchor_objects"], 3)
        self.assertFalse(report["arc_task_accuracy_claim"])
        self.assertFalse(report["held_out_test_data_used"])

    def test_only_training_split_is_accepted(self):
        from pathlib import Path
        import json
        import tempfile

        source = Path(__file__).parent / "fixtures" / "arc_agi2_d89b689b_train.json"
        task = json.loads(source.read_text(encoding="utf-8"))
        task["split"] = "test"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_training_pair_experiment(path)


if __name__ == "__main__":
    unittest.main()
