import unittest

from arc_r2.dsl import Program
from arc_r2.emoji_alchemy import (
    apply_emoji_program,
    generate_alchemy_candidates,
    semantic_signature,
    semantic_similarity,
)
from arc_r2.emoji_codec import decode_grid, decode_object_graph, encode_grid, encode_object_graph
from arc_r2.grid import as_grid


class EmojiAlchemyTests(unittest.TestCase):
    def test_emoji_grid_round_trip_is_lossless(self):
        grid = as_grid([[0, 2, 2], [0, 3, 0]])
        self.assertEqual(decode_grid(encode_grid(grid)), grid)

    def test_object_graph_round_trip_is_lossless(self):
        grid = as_grid([[0, 2, 2], [0, 3, 0], [0, 0, 0]])
        self.assertEqual(decode_object_graph(encode_object_graph(grid)), grid)

    def test_symbolic_program_matches_canonical_dsl(self):
        grid = as_grid([[1, 0, 2], [3, 4, 5]])
        program = Program().then("REVERSE.HORIZONTAL")
        symbolic = apply_emoji_program(grid, program)
        self.assertEqual(decode_grid(symbolic), program.apply(grid))

    def test_semantic_similarity_is_one_for_identical_grid(self):
        grid = as_grid([[0, 2, 2], [0, 3, 0]])
        self.assertEqual(semantic_similarity(grid, grid), 1.0)
        signature = semantic_signature(grid)
        self.assertIn("color_counts", signature)
        self.assertIn("object_shapes", signature)
        self.assertIn("relations", signature)

    def test_semantic_similarity_distinguishes_different_color_structure(self):
        left = as_grid([[0, 2, 2], [0, 3, 0]])
        right = as_grid([[0, 4, 4], [0, 5, 0]])
        self.assertLess(semantic_similarity(left, right), 1.0)

    def test_alchemy_generates_bounded_compositions(self):
        candidates = generate_alchemy_candidates([Program()], max_depth=2)
        self.assertGreater(len(candidates), 8)
        self.assertIn((), [candidate.canonical() for candidate in candidates])
        self.assertIn(
            ("REVERSE.HORIZONTAL", "ROTATE.180"),
            [candidate.canonical() for candidate in candidates],
        )
        with self.assertRaisesRegex(ValueError, "between 0 and 2"):
            generate_alchemy_candidates(max_depth=3)


if __name__ == "__main__":
    unittest.main()
