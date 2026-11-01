import json
import tempfile
import unittest
from pathlib import Path

from arc_r2.compare_engines import compare_engines, load_tasks


class EngineComparisonTests(unittest.TestCase):
    def test_leave_one_out_uses_only_train_pairs(self):
        task = {
            "task_id": "identity-demo",
            "train": [
                {"input": [[0, 0], [0, 0]], "output": [[0, 0], [0, 0]]},
                {"input": [[0, 0], [0, 0]], "output": [[0, 0], [0, 0]]},
            ],
            # Deliberately invalid: comparison must not inspect the test array.
            "test": [{"input": "not-a-grid"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps([task]), encoding="utf-8")
            report = compare_engines(path, limit=1, max_depth=1,
                                     candidate_cap=80, mcts_iterations=10)

        self.assertEqual(report["experiment"], "leave_one_training_pair_out")
        self.assertEqual(report["eligible_tasks"], 1)
        self.assertEqual(report["summary"]["baseline"]["folds"], 2)
        self.assertEqual(report["summary"]["object"]["folds"], 2)
        self.assertEqual(report["summary"]["baseline"]["exact"], 2)
        self.assertEqual(report["summary"]["object"]["exact"], 2)
        self.assertEqual(report["summary"]["hian"]["folds"], 2)
        self.assertEqual(report["summary"]["hian"]["exact"], 2)
        self.assertIsNotNone(report["folds"][0]["engines"]["hian"]["hian_report"])
        self.assertIn("never read", report["data_boundary"])

    def test_single_pair_tasks_are_reported_as_skipped(self):
        task = {"task_id": "one-pair", "train": [
            {"input": [[0]], "output": [[0]]}
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps([task]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least two train pairs"):
                compare_engines(path, limit=1, candidate_cap=20)

    def test_loader_accepts_directory_of_task_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "b-task.json").write_text(json.dumps({
                "train": [{"input": [[0]], "output": [[0]]}], "test": []
            }), encoding="utf-8")
            (folder / "a-task.json").write_text(json.dumps({
                "train": [{"input": [[1]], "output": [[1]]}], "test": []
            }), encoding="utf-8")
            loaded = load_tasks(folder, limit=1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["task_id"], "a-task")

    def test_loader_accepts_standard_task_mapping(self):
        tasks = {"abc": {"train": [
            {"input": [[0]], "output": [[0]]}
        ], "test": []}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps(tasks), encoding="utf-8")
            loaded = load_tasks(path)
        self.assertEqual(loaded[0]["task_id"], "abc")


if __name__ == "__main__":
    unittest.main()
