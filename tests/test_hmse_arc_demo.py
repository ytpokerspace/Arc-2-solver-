import json
import unittest
from pathlib import Path

from arc_r2.hmse_demo import run_comparison


class HmseArcDemoTests(unittest.TestCase):
    def test_00d62c1b_end_to_end_against_primitive_baseline(self):
        fixture = Path(__file__).parent / "fixtures" / "arc_agi2_00d62c1b_train.json"
        task = json.loads(fixture.read_text(encoding="utf-8"))
        report = run_comparison(task, candidate_cap=1500, max_depth=2)

        self.assertEqual(report["task_id"], "00d62c1b")
        self.assertEqual(report["protocol"], "leave-one-training-pair-out")
        self.assertFalse(report["official_test_outputs_read"])
        self.assertEqual(report["training_pairs"], 5)
        self.assertEqual(report["baseline"]["held_out_exact"], 5)
        self.assertEqual(report["hmse"]["held_out_exact"], 5)
        self.assertEqual(report["hmse"]["folds"], 5)
        self.assertEqual(
            {fold["hmse_strategy"] for fold in report["fold_results"]},
            {"shape-radar-hole-rule"},
        )
        for fold in report["fold_results"]:
            self.assertTrue(fold["baseline_exact"])
            self.assertTrue(fold["hmse_exact"])
            self.assertGreater(fold["hmse_radar_hole_objects"], 0)
            self.assertEqual(
                [item["stage_id"] for item in fold["hmse_stage_trace"]],
                ["perception", "rule-synthesis", "prediction"],
            )
            self.assertEqual(fold["hmse_total_stage_evaluations"], 3)


if __name__ == "__main__":
    unittest.main()
