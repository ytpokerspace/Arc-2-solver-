import json
import tempfile
import unittest
from pathlib import Path

from arc_r2.primitives import (
    Program, Primitive, apply_primitive, as_grid, background_color,
    find_objects, fit_program, flip_horizontal, flip_vertical,
    alternate_tile, append_prefix, fill_holes, gravity, keep_largest_recolor,
    mirror_complete, rotate, stamp_by_mask, transpose,
)
from arc_r2.primitive_benchmark import run_benchmark


class PrimitiveLibraryTests(unittest.TestCase):
    def test_geometric_operations(self):
        grid = as_grid([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(rotate(grid, 1), ((4, 1), (5, 2), (6, 3)))
        self.assertEqual(rotate(grid, 2), ((6, 5, 4), (3, 2, 1)))
        self.assertEqual(flip_horizontal(grid), ((3, 2, 1), (6, 5, 4)))
        self.assertEqual(flip_vertical(grid), ((4, 5, 6), (1, 2, 3)))
        self.assertEqual(transpose(grid), ((1, 4), (2, 5), (3, 6)))

    def test_object_detection_four_and_eight_connected(self):
        grid = as_grid([[0, 2], [2, 0]])
        self.assertEqual(len(find_objects(grid, 4)), 2)
        self.assertEqual(len(find_objects(grid, 8)), 1)
        obj = find_objects(as_grid([[0, 0, 0], [0, 3, 3], [0, 3, 0]]))[0]
        self.assertEqual((obj.color, obj.size, obj.bbox), (3, 3, (1, 1, 2, 2)))

    def test_alternating_tile_repeats_with_row_reflection(self):
        source = as_grid([[7, 9], [4, 3]])
        expected = as_grid([
            [7, 9, 7, 9, 7, 9],
            [4, 3, 4, 3, 4, 3],
            [9, 7, 9, 7, 9, 7],
            [3, 4, 3, 4, 3, 4],
            [7, 9, 7, 9, 7, 9],
            [4, 3, 4, 3, 4, 3],
        ])
        self.assertEqual(alternate_tile(source, 3, 3, "row_flip_h"), expected)
        fit = fit_program(((source, expected),), candidate_cap=5000, max_depth=1)
        self.assertTrue(fit.exact_fit)
        self.assertEqual(fit.program.apply(source), expected)

    def test_mask_stamp_expands_each_non_background_cell(self):
        source = as_grid([[1, 0], [0, 1]])
        expected = as_grid([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.assertEqual(stamp_by_mask(source), expected)

    def test_fill_holes_and_keep_largest_object(self):
        ring = as_grid([
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 0, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ])
        self.assertEqual(fill_holes(ring, 3)[2][2], 3)
        objects = as_grid([
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 2],
            [0, 0, 0, 0, 0],
        ])
        kept = keep_largest_recolor(objects, 4)
        self.assertEqual(kept, ((0, 4, 4, 0, 0), (0, 4, 4, 0, 0), (0, 0, 0, 0, 0)))

    def test_append_prefix_rows_and_columns(self):
        grid = as_grid([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(append_prefix(grid, "rows", 2),
                         ((1, 2), (3, 4), (5, 6), (1, 2), (3, 4)))
        self.assertEqual(append_prefix(grid, "columns", 1),
                         ((1, 2, 1), (3, 4, 3), (5, 6, 5)))

    def test_gravity_preserves_lane_order(self):
        grid = as_grid([[0, 2], [3, 0], [0, 4]])
        self.assertEqual(gravity(grid, "down"), ((0, 0), (0, 2), (3, 4)))
        self.assertEqual(gravity(grid, "left"), ((2, 0), (3, 0), (4, 0)))

    def test_mirror_completion_and_program(self):
        grid = as_grid([[0, 2, 0], [0, 0, 0]])
        self.assertEqual(mirror_complete(grid, "horizontal"), ((0, 2, 0), (0, 0, 0)))
        program = Program((Primitive("FLIP_H"),))
        self.assertEqual(program.apply(as_grid([[1, 2]])), ((2, 1),))

    def test_exact_fit_required_and_shortest_program(self):
        source = as_grid([[0, 1], [0, 0]])
        target = rotate(source, 1)
        result = fit_program(((source, target),), candidate_cap=5000, max_depth=1)
        self.assertTrue(result.exact_fit)
        self.assertEqual(result.program.apply(source), target)
        # A deliberately inconsistent pair set must not yield a partial-fit guess.
        impossible = fit_program(
            ((as_grid([[0, 1]]), as_grid([[1, 0]])),
             (as_grid([[0, 1]]), as_grid([[0, 1]]))),
            candidate_cap=5000, max_depth=1,
        )
        self.assertFalse(impossible.exact_fit)
        self.assertIsNone(impossible.program)


class PrimitiveBenchmarkTests(unittest.TestCase):
    def test_leave_one_training_pair_out_and_never_reads_test(self):
        task = {
            "task_id": "flip-demo",
            "train": [
                {"input": [[0, 1], [0, 0]], "output": [[1, 0], [0, 0]]},
                {"input": [[0, 2], [0, 0]], "output": [[2, 0], [0, 0]]},
            ],
            "test": [{"input": "deliberately invalid and must not be parsed"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps([task]), encoding="utf-8")
            report = run_benchmark(path, limit=1, candidate_cap=5000, max_depth=1)
        self.assertFalse(report["mcts_used"])
        self.assertFalse(report["beam_search_used"])
        self.assertEqual(report["folds"], 2)
        self.assertEqual(report["held_out_exact"], 2)


if __name__ == "__main__":
    unittest.main()
