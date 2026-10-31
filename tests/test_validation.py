import json
import tempfile
import unittest
from pathlib import Path

from arc_r2.validation import (
    leave_one_out_folds,
    run_leave_one_out,
)


class LeaveOneOutTests(unittest.TestCase):
    def test_each_fold_hides_exactly_one_pair(self):
        pairs = [
            {"input": [[0]], "output": [[0]]},
            {"input": [[1]], "output": [[1]]},
            {"input": [[2]], "output": [[2]]},
        ]
        folds = list(leave_one_out_folds(pairs))
        self.assertEqual(len(folds), 3)
        for heldout_index, visible, heldout in folds:
            self.assertEqual(len(visible), 2)
            self.assertEqual(heldout, pairs[heldout_index])
            self.assertNotIn(heldout, visible)

    def test_single_pair_has_no_valid_holdout(self):
        self.assertEqual(list(leave_one_out_folds([{"input": [[0]], "output": [[0]]}])), [])

    def test_identity_baseline_on_known_identity_task(self):
        task = {
            "task_id": "synthetic-identity",
            "train": [
                {"input": [[0, 1]], "output": [[0, 1]]},
                {"input": [[2, 3]], "output": [[2, 3]]},
            ],
            "test": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "tasks.json"
            dataset.write_text(json.dumps([task]), encoding="utf-8")
            report = run_leave_one_out(
                dataset,
                limit=1,
                max_depth=1,
                candidate_cap=20,
                mcts_iterations=5,
                selected_variants={"identity"},
            )

        self.assertEqual(report["protocol"], "leave_one_training_pair_out")
        self.assertEqual(len(report["variants"]), 1)
        result = report["variants"][0]
        self.assertEqual(result["heldout_exact"], 2)
        self.assertEqual(result["heldout_total"], 2)
        self.assertEqual(result["heldout_accuracy"], 1.0)
        self.assertEqual(result["tasks_all_holdouts_exact"], 1)


if __name__ == "__main__":
    unittest.main()
