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
        for encoded in (b"not-a-grid", b"AGC1\x00\x01\x00\x01"):
            with self.subTest(encoded=encoded):
                with self.assertRaises(ValueError):
                    decode_grid(encoded)

    def test_rejects_dimensions_outside_arc_grid_limit(self):
        with self.assertRaisesRegex(ValueError, "30x30"):
            encode_grid(tuple(tuple(0 for _ in range(31)) for _ in range(1)))
        oversized_header = b"AGC1" + (31).to_bytes(2, "big") + (1).to_bytes(2, "big") + b"\x00" + bytes(31)
        with self.assertRaises(ValueError):
            decode_grid(oversized_header)

    def test_rejects_noncanonical_repeated_adjacent_runs(self):
        # A 1x2 grid of color 1 encoded as two adjacent runs is not canonical.
        encoded = b"AGC1" + b"\x00\x01\x00\x02" + b"\x01" + b"\x01\x01\x00\x01\x00\x01"
        with self.assertRaises(ValueError):
            decode_grid(encoded)


if __name__ == "__main__":
    unittest.main()
