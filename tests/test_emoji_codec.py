"""Tests for lossless emoji grid and object-graph representations."""
import unittest

from arc_r2.emoji_codec import (
    COLOR_TO_EMOJI,
    EmojiGrid,
    decode_grid,
    decode_object_graph,
    encode_grid,
    encode_object_graph,
    render_emoji_grid,
)


class EmojiGridCodecTests(unittest.TestCase):
    def test_all_arc_colors_round_trip(self):
        grid = (tuple(range(10)),)
        self.assertEqual(decode_grid(encode_grid(grid)), grid)

    def test_grid_round_trip_preserves_shape_and_cells(self):
        grid = ((0, 0, 2, 2), (0, 3, 3, 0), (4, 4, 0, 0))
        encoded = encode_grid(grid)
        self.assertEqual((encoded.height, encoded.width), (3, 4))
        self.assertEqual(decode_grid(encoded), grid)

    def test_render_has_one_token_per_cell(self):
        encoded = encode_grid(((0, 1), (2, 3)))
        self.assertEqual(render_emoji_grid(encoded), "⬛🟦\n🟥🟩")

    def test_rejects_unknown_emoji(self):
        with self.assertRaises(ValueError):
            decode_grid((("🧩",),))

    def test_rejects_ragged_emoji_grid(self):
        with self.assertRaises(ValueError):
            decode_grid((("⬛", "🟦"), ("🟥",)))

    def test_mapping_covers_exactly_arc_palette(self):
        self.assertEqual(set(COLOR_TO_EMOJI), set(range(10)))
        self.assertEqual(len(set(COLOR_TO_EMOJI.values())), 10)


class EmojiObjectGraphTests(unittest.TestCase):
    def test_object_graph_round_trip(self):
        grid = (
            (0, 0, 0, 0, 0, 0),
            (0, 2, 2, 0, 3, 0),
            (0, 2, 0, 0, 3, 0),
            (0, 0, 0, 4, 4, 0),
            (0, 0, 0, 0, 0, 0),
        )
        graph = encode_object_graph(grid)
        self.assertEqual(decode_object_graph(graph), grid)
        self.assertGreaterEqual(len(graph.objects), 3)

    def test_single_color_grid_round_trip(self):
        grid = ((7, 7), (7, 7))
        self.assertEqual(decode_object_graph(encode_object_graph(grid)), grid)

    def test_custom_background_round_trip(self):
        grid = ((1, 1, 2), (1, 3, 2))
        self.assertEqual(
            decode_object_graph(encode_object_graph(grid, background=1)),
            grid,
        )

    def test_object_graph_contains_spatial_relations(self):
        grid = (
            (0, 2, 0, 3),
            (0, 2, 0, 3),
        )
        graph = encode_object_graph(grid)
        self.assertTrue(any("LEFT_OF" in labels for _, _, labels in graph.relations))

    def test_rejects_object_cells_outside_grid(self):
        from arc_r2.emoji_codec import EmojiObject, EmojiObjectGraph
        graph = EmojiObjectGraph(
            height=1, width=1, background="⬛",
            objects=(EmojiObject("🟥", (0, 0), ((0, 1),)),),
            relations=(),
        )
        with self.assertRaises(ValueError):
            decode_object_graph(graph)


if __name__ == "__main__":
    unittest.main()
