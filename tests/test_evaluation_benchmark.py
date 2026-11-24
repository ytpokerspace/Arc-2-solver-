import json
import tempfile
import unittest
from pathlib import Path

from arc_r2.evaluation_benchmark import run_evaluation


class BlindEvaluationHarnessTests(unittest.TestCase):
    def test_predicts_from_train_without_reading_test_output(self):
        with tempfile.TemporaryDirectory() as directory:
            task = {
                "train": [{"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]}],
                "test": [{"input": [[0, 2], [2, 0]], "output": "must never be inspected"}],
            }
            Path(directory, "task-a.json").write_text(json.dumps(task), encoding="utf-8")
            report, predictions = run_evaluation(Path(directory), candidate_cap=20, max_depth=0)
            self.assertEqual(predictions["task-a"], [[[0, 2], [2, 0]]])
            self.assertEqual(report["task_count"], 1)
            self.assertEqual(report["test_input_count"], 1)
            self.assertEqual(report["predictions_produced"], 1)
            self.assertEqual(report["abstentions"], 0)
            self.assertIsNone(report["test_accuracy"])
            self.assertFalse(report["test_outputs_read"])
            self.assertFalse(report["semantic_identity_integrated"])

    def test_abstains_when_no_exact_fit_program_is_found(self):
        with tempfile.TemporaryDirectory() as directory:
            task = {
                "train": [{"input": [[1, 0], [0, 1]], "output": [[2, 0], [0, 2]]}],
                "test": [{"input": [[0, 3], [3, 0]]}],
            }
            Path(directory, "task-b.json").write_text(json.dumps(task), encoding="utf-8")
            report, predictions = run_evaluation(Path(directory), candidate_cap=1, max_depth=0)
            self.assertEqual(predictions["task-b"], [None])
            self.assertEqual(report["predictions_produced"], 0)
            self.assertEqual(report["abstentions"], 1)


if __name__ == "__main__":
    unittest.main()
