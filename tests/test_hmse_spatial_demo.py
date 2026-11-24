import json
import unittest
from pathlib import Path

from arc_r2.hmse_spatial_demo import run_comparison, SpatialProgram


class HmseSpatialDemoTests(unittest.TestCase):
    def test_d89b689b_leave_one_pair_out(self):
        fixture = Path(__file__).parent / "fixtures" / "arc_agi2_d89b689b_train.json"
        task = json.loads(fixture.read_text(encoding="utf-8"))
        self.assertNotIn("test", task)
        report = run_comparison(task, candidate_cap=1500, max_depth=2)
        self.assertEqual(report["task_id"], "d89b689b")
        self.assertEqual(report["protocol"], "leave-one-training-pair-out")
        self.assertFalse(report["official_test_outputs_read"])
        self.assertEqual(report["training_pairs"], 3)
        self.assertEqual(report["hmse"]["held_out_exact"], 3)
        self.assertEqual(report["hmse"]["folds"], 3)
        self.assertEqual(
            {fold["hmse_strategy"] for fold in report["fold_results"]},
            {"anchor-quadrant-relocation"},
        )
        for fold in report["fold_results"]:
            self.assertTrue(fold["hmse_exact"])
            self.assertEqual(
                [item["stage_id"] for item in fold["hmse_stage_trace"]],
                ["perception", "rule-synthesis", "prediction"],
            )
            self.assertEqual(fold["hmse_total_stage_evaluations"], 3)

    def test_spatial_program_rejects_missing_quadrant(self):
        grid = [[0] * 8 for _ in range(8)]
        grid[3][3] = grid[3][4] = grid[4][3] = grid[4][4] = 8
        grid[0][0] = 1
        grid[0][7] = 2
        grid[7][0] = 3
        with self.assertRaises(ValueError):
            SpatialProgram().apply(grid)


if __name__ == "__main__":
    unittest.main()
