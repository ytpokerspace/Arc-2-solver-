import json
import tempfile
import unittest
from pathlib import Path

from arc_r2.full_orchestration_benchmark import run_benchmark


class FullOrchestrationBenchmarkTests(unittest.TestCase):
    def test_leave_one_out_compares_both_engines_without_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            task = {
                "train": [
                    {"input": [[1, 0], [0, 2]], "output": [[1, 0], [0, 2]]},
                    {"input": [[0, 3], [4, 0]], "output": [[0, 3], [4, 0]]},
                ],
                "test": [{"input": [[8, 0], [0, 9]], "output": "must not be read"}],
            }
            path = Path(directory) / "task.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            report = run_benchmark(path, candidate_cap=40, max_depth=0)
            self.assertEqual(report["folds"], 2)
            self.assertEqual(report["full_orchestration"]["held_out_exact"], 2)
            self.assertEqual(report["primitive_baseline"]["held_out_exact"], 2)
            self.assertFalse(report["hidden_test_outputs_read"])
            self.assertFalse(report["memory_enabled"])
            self.assertEqual(
                set(report["family_profile"]),
                {"ahpse", "primitive", "registered_rules", "object"},
            )
            self.assertTrue(all(
                item["runtime_seconds_total"] >= 0
                and item["candidate_count_total"] >= 0
                for item in report["family_profile"].values()
            ))
            self.assertEqual(len(report["fold_results"]), 2)

    def test_skips_tasks_without_two_training_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text(json.dumps({"train": [{"input": [[1]], "output": [[1]]}]}),
                            encoding="utf-8")
            report = run_benchmark(path, candidate_cap=10, max_depth=0)
            self.assertEqual(report["folds"], 0)
            self.assertEqual(report["skipped_tasks"], ["task"])


if __name__ == "__main__":
    unittest.main()
