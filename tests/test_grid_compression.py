import unittest

from arc_r2.grid_compression import decode_grid, encode_grid


class GridCompressionTests(unittest.TestCase):
    def test_round_trip_repetitive_grid_uses_compact_encoding(self):
        grid = tuple(tuple([0] * 30) for _ in range(20))
        encoded = encode_grid(grid)
        self.assertEqual(decode_grid(encoded), grid)
        self.assertLess(len(encoded), 20 * 30)

    def test_round_trip_mixed_and_checkerboard_grids(self):
        grids = (
            ((0, 1, 2), (2, 1, 0)),
            tuple(tuple((r + c) % 2 for c in range(16)) for r in range(16)),
            ((9,),),
        )
        for grid in grids:
            with self.subTest(grid=grid):
                self.assertEqual(decode_grid(encode_grid(grid)), grid)

    def test_rejects_malformed_data(self):
        with self.assertRaises(ValueError):
            decode_grid(b"not-a-grid")


if __name__ == "__main__":
    unittest.main()
