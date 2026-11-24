import json
import unittest
from pathlib import Path

from arc_r2.hmse_rule_selector import run_suite, run_task


class HmseRuleSelectorTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parent / "fixtures"
        self.hole = json.loads((root / "arc_agi2_00d62c1b_train.json").read_text(encoding="utf-8"))
        self.spatial = json.loads((root / "arc_agi2_d89b689b_train.json").read_text(encoding="utf-8"))

    def test_selects_rule_family_without_task_label(self):
        report = run_suite((self.hole, self.spatial), candidate_cap=1500, max_depth=2)
        self.assertEqual(report["task_count"], 2)
        self.assertEqual(report["total_folds"], 8)
        self.assertEqual(report["selector_exact"], 8)
        by_task = {item["task_id"]: item for item in report["tasks"]}
        self.assertEqual(by_task["00d62c1b"]["selector"]["exact"], 5)
        self.assertEqual(by_task["d89b689b"]["selector"]["exact"], 3)
        self.assertEqual(
            {fold["selected_rule"] for fold in by_task["00d62c1b"]["fold_results"]},
            {"hole-fill"},
        )
        self.assertEqual(
            {fold["selected_rule"] for fold in by_task["d89b689b"]["fold_results"]},
            {"anchor-quadrant-relocation"},
        )
        for item in report["tasks"]:
            self.assertEqual(item["selector"]["selected"], item["selector"]["folds"])
            self.assertEqual(item["selector"]["ambiguous"], 0)
            self.assertEqual(item["selector"]["unresolved"], 0)
            for fold in item["fold_results"]:
                self.assertFalse("expected" in fold and fold["expected"] is None)
                self.assertEqual(
                    [stage["stage_id"] for stage in fold["stage_trace"]],
                    ["perception", "rule-selection", "prediction"],
                )

    def test_selector_refuses_unknown_rule_family(self):
        task = {
            "task_id": "unknown",
            "train": [
                {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
                {"input": [[0, 2], [0, 0]], "output": [[0, 0], [2, 0]]},
            ],
        }
        report = run_task(task, candidate_cap=100, max_depth=1)
        self.assertEqual(report["selector"]["selected"], 0)
        self.assertEqual(report["selector"]["unresolved"], 2)
        self.assertEqual(report["selector"]["exact"], 0)
        self.assertTrue(all(
            fold["selected_rule"] is None and fold["prediction"] is None
            for fold in report["fold_results"]
        ))


if __name__ == "__main__":
    unittest.main()
