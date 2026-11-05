"""Tests for the three-way emoji representation codec benchmark."""
import unittest

from arc_r2.emoji_benchmark import benchmark, load_grids


class EmojiBenchmarkTests(unittest.TestCase):
    def test_all_three_representations_reconstruct_exactly(self):
        grids = (
            ((0, 0, 0), (0, 2, 0)),
            ((3, 3), (3, 3)),
            ((1, 0, 4), (1, 0, 4)),
        )
        report = benchmark(grids, repeats=2)
        representations = report["representations"]
        self.assertEqual(
            set(representations),
            {"integer", "emoji_grid", "emoji_object_graph"},
        )
        for result in representations.values():
            self.assertTrue(result["exact_reconstruction"])
            self.assertGreater(result["total_serialized_json_chars"], 0)

    def test_rejects_zero_repeats(self):
        with self.assertRaises(ValueError):
            benchmark((((0,),),), repeats=0)

    def test_loads_training_inputs_and_outputs(self):
        import json
        import tempfile
        from pathlib import Path

        payload = {
            "task-a": {
                "train": [
                    {"input": [[0, 1]], "output": [[1, 0]]},
                ],
                "test": [{"input": [[2, 2]]}],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            grids = load_grids(str(path))
        self.assertEqual(grids, (((0, 1),), ((1, 0),)))


if __name__ == "__main__":
    unittest.main()
