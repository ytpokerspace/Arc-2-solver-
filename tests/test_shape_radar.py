import unittest

from arc_r2.shape_radar import canonical_shape_signature, scan_grid


class ShapeRadarTests(unittest.TestCase):
    def test_reports_grid_and_object_geometry(self):
        result = scan_grid([
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 0, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ])
        self.assertEqual(result["grid"]["height"], 5)
        self.assertEqual(result["scan"]["cells_examined"], 25)
        obj = result["objects"][0]
        self.assertEqual(obj["size"], 8)
        self.assertEqual(obj["bbox"], [1, 1, 3, 3])
        self.assertEqual(obj["holes"], 1)
        self.assertTrue(obj["symmetry"]["horizontal"])
        self.assertTrue(obj["symmetry"]["vertical"])

    def test_canonical_signature_ignores_rotation_and_reflection(self):
        shape = {(0, 0), (1, 0), (2, 0), (2, 1)}
        rotated = {(0, 2), (0, 1), (0, 0), (1, 0)}
        reflected = {(0, 1), (1, 1), (2, 1), (2, 0)}
        self.assertEqual(canonical_shape_signature(shape),
                         canonical_shape_signature(rotated))
        self.assertEqual(canonical_shape_signature(shape),
                         canonical_shape_signature(reflected))

    def test_repeated_shapes_are_grouped_across_colors_and_positions(self):
        result = scan_grid([
            [0, 1, 0, 0, 2],
            [0, 1, 0, 0, 2],
            [0, 0, 0, 0, 0],
        ])
        self.assertEqual(len(result["objects"]), 2)
        self.assertEqual(len(result["repeated_shapes"]), 1)
        self.assertEqual(result["repeated_shapes"][0]["count"], 2)
        self.assertTrue(result["relationships"][0]["same_shape"])

    def test_connectivity_changes_diagonal_component_partition(self):
        grid = [[0, 3], [3, 0]]
        four = scan_grid(grid, connectivity=4)
        eight = scan_grid(grid, connectivity=8)
        self.assertEqual(four["scan"]["primary_object_count"], 2)
        self.assertEqual(eight["scan"]["primary_object_count"], 1)
        self.assertEqual(four["scan"]["secondary_object_count"], 1)

    def test_rejects_invalid_connectivity_and_empty_shape(self):
        with self.assertRaises(ValueError):
            scan_grid([[0]], connectivity=6)
        with self.assertRaises(ValueError):
            canonical_shape_signature(set())


if __name__ == "__main__":
    unittest.main()
