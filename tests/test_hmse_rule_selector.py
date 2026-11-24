import json
import unittest
from pathlib import Path

from arc_r2.hmse_rule_selector import run_suite, run_task
from arc_r2.hmse_latin_square_rule import LatinSquareCompletionProgram


class HmseRuleSelectorTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parent / "fixtures"
        self.hole = json.loads((root / "arc_agi2_00d62c1b_train.json").read_text(encoding="utf-8"))
        self.spatial = json.loads((root / "arc_agi2_d89b689b_train.json").read_text(encoding="utf-8"))
        self.reflection = json.loads((root / "synthetic_reflection_train.json").read_text(encoding="utf-8"))
        self.latin = json.loads((root / "arc_agi2_4cd1b7b2_train.json").read_text(encoding="utf-8"))

    def test_selects_four_rule_families_without_task_label(self):
        report = run_suite((self.hole, self.spatial, self.reflection, self.latin), candidate_cap=1500, max_depth=2)
        self.assertEqual(report["task_count"], 4)
        self.assertEqual(report["total_folds"], 14)
        self.assertEqual(report["selector_exact"], 14)
        by_task = {item["task_id"]: item for item in report["tasks"]}
        expected = {
            "00d62c1b": ("hole-fill", 5),
            "d89b689b": ("anchor-quadrant-relocation", 3),
            "synthetic-left-right-reflection": ("reflection", 3),
            "4cd1b7b2": ("latin-square-completion", 3),
        }
        for task_id, (rule_id, folds) in expected.items():
            item = by_task[task_id]
            self.assertEqual(item["selector"]["exact"], folds)
            self.assertEqual(item["selector"]["selected"], folds)
            self.assertEqual(item["selector"]["ambiguous"], 0)
            self.assertEqual(item["selector"]["unresolved"], 0)
            self.assertEqual(
                {fold["selected_rule"] for fold in item["fold_results"]},
                {rule_id},
            )
            for fold in item["fold_results"]:
                self.assertEqual(
                    [stage["stage_id"] for stage in fold["stage_trace"]],
                    ["perception", "rule-selection", "prediction"],
                )

    def test_selector_refuses_unknown_or_inconsistent_rule_family(self):
        task = {
            "task_id": "unknown",
            "train": [
                {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
                {"input": [[0, 2], [0, 0]], "output": [[0, 0], [2, 0]]},
                {"input": [[1, 0, 0], [0, 0, 2]], "output": [[2, 0, 0], [0, 0, 1]]},
            ],
        }
        report = run_task(task, candidate_cap=100, max_depth=1)
        self.assertEqual(report["selector"]["selected"], 0)
        self.assertEqual(report["selector"]["unresolved"], 3)
        self.assertEqual(report["selector"]["exact"], 0)
        self.assertTrue(all(
            fold["selected_rule"] is None and fold["prediction"] is None
            for fold in report["fold_results"]
        ))

    def test_latin_square_program_rejects_ambiguous_and_invalid_inputs(self):
        program = LatinSquareCompletionProgram((1, 2))
        with self.assertRaises(ValueError):
            program.apply([[0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            program.apply([[1, 1], [0, 0]])

    def test_latin_square_program_preserves_given_cells(self):
        program = LatinSquareCompletionProgram((1, 2, 3, 4))
        result = program.apply([[0, 4, 2, 3], [4, 1, 0, 2], [0, 3, 4, 0], [3, 0, 1, 4]])
        self.assertEqual(result, ((1, 4, 2, 3), (4, 1, 3, 2), (2, 3, 4, 1), (3, 2, 1, 4)))


if __name__ == "__main__":
    unittest.main()
