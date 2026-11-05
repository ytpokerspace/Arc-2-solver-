import json
import tempfile
import unittest
from pathlib import Path

from arc_r2.primitive_benchmark import load_tasks, run_benchmark


class PrimitiveHarnessTests(unittest.TestCase):
    def test_loader_accepts_standard_task_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps({
                "demo": {
                    "train": [
                        {"input": [[0, 1]], "output": [[1, 0]]},
                        {"input": [[0, 2]], "output": [[2, 0]]},
                    ],
                    "test": [{"input": "must not be parsed"}],
                }
            }), encoding="utf-8")
            tasks = load_tasks(path, 10)
            report = run_benchmark(path, 1, 3000, 1)
        self.assertEqual(tasks[0]["task_id"], "demo")
        self.assertEqual(report["engine"], "primitive-first-exhaustive")
        self.assertEqual(report["held_out_exact"], 2)
        self.assertFalse(report["mcts_used"])


if __name__ == "__main__":
    unittest.main()
