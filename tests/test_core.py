import unittest

from arc_r2.primitives import (
    as_grid, find_objects, flip_horizontal, flip_vertical, rotate, transpose,
)


class GridPrimitiveTests(unittest.TestCase):
    def test_geometric_transforms_are_distinct(self):
        grid = as_grid([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(flip_horizontal(grid), ((3, 2, 1), (6, 5, 4)))
        self.assertEqual(flip_vertical(grid), ((4, 5, 6), (1, 2, 3)))
        self.assertEqual(transpose(grid), ((1, 4), (2, 5), (3, 6)))
        self.assertEqual(rotate(grid, 1), ((4, 1), (5, 2), (6, 3)))

    def test_connected_components(self):
        grid = as_grid([[0, 2], [2, 0]])
        self.assertEqual(len(find_objects(grid, connectivity=4)), 2)
        self.assertEqual(len(find_objects(grid, connectivity=8)), 1)


if __name__ == "__main__":
    unittest.main()
