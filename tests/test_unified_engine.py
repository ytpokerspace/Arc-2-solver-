import json
import tempfile
import unittest
from pathlib import Path

from arc_r2.emoji_engine import render_trace
from arc_r2.unified_engine import (
    UnifiedConfig, predict_task, run_leave_one_out, solve_training_pairs,
)


class UnifiedEngineTests(unittest.TestCase):
    def setUp(self):
        self.config = UnifiedConfig(
            primitive_candidate_cap=500,
            primitive_max_depth=1,
            mcts_iterations=8,
            mcts_branch_width=5,
            mcts_max_depth=2,
        )

    def test_combines_search_paths_and_finds_exact_flip(self):
        pairs = (
            ([[0, 1], [0, 0]], [[1, 0], [0, 0]]),
            ([[0, 2], [0, 0]], [[2, 0], [0, 0]]),
        )
        result = solve_training_pairs(pairs, self.config)
        self.assertTrue(result.exact_fit)
        self.assertIsNotNone(result.program)
        self.assertTrue(all(result.program.apply(source) == target for source, target in pairs))
        self.assertGreaterEqual(result.candidates_evaluated, 0)
        self.assertIn(result.source, {"primitive-enumeration", "predictive-mcts"})

    def test_abstains_when_no_exact_program_is_found(self):
        pairs = (
            ([[0, 1]], [[1, 0]]),
            ([[0, 1]], [[0, 1]]),
        )
        result = solve_training_pairs(pairs, self.config)
        self.assertFalse(result.exact_fit)
        self.assertIsNone(result.program)
        self.assertEqual(result.source, None)

    def test_predicts_test_input_without_reading_test_output(self):
        task = {
            "task_id": "flip-only",
            "train": [
                {"input": [[0, 1], [0, 0]], "output": [[1, 0], [0, 0]]},
                {"input": [[0, 2], [0, 0]], "output": [[2, 0], [0, 0]]},
            ],
            "test": [{
                "input": [[0, 3], [0, 0]],
                "output": "THIS FIELD MUST NOT BE READ",
            }],
        }
        result = predict_task(task, self.config)
        self.assertEqual(result["status"], "predicted")
        self.assertEqual(result["test_predictions"][0]["prediction"], [[3, 0], [0, 0]])
        self.assertNotIn("expected", result["test_predictions"][0])

    def test_leave_one_out_hides_each_training_pair(self):
        task = {
            "task_id": "flip-loo",
            "train": [
                {"input": [[0, 1]], "output": [[1, 0]]},
                {"input": [[0, 2]], "output": [[2, 0]]},
            ],
            "test": [{"input": [[0, 3]], "output": "UNUSED"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps([task]), encoding="utf-8")
            report = run_leave_one_out(path, limit=1, config=self.config)
        self.assertEqual(report["protocol"], "leave_one_training_pair_out")
        self.assertEqual(report["folds"], 2)
        self.assertEqual(report["held_out_exact"], 2)

    def test_emoji_trace_uses_stable_canonical_ids(self):
        self.assertEqual(
            render_trace(["ARC2.INPUT", "ARC2.PERCEPTION", "ARC2.EXACT_FIT", "ARC2.TEST_PREDICTION"]),
            "📥 → 👁️ → ✅ → 📤",
        )
        with self.assertRaisesRegex(ValueError, "unknown ARC-2 engine"):
            render_trace(["ARC3.ACTION"])


if __name__ == "__main__":
    unittest.main()
