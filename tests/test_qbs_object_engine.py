from __future__ import annotations

import unittest

from arc_r2.grid import Grid
from arc_r2.object_engine import (
    ObjectState, Scene, extract_scene, move, rotate90,
    flip_horizontal, scale_up, render_scene, move_object, object_exists,
)
from arc_r2.primitive_engine import PrimitiveRegistry, PrimitiveSpec
from arc_r2.qbs_search import (
    SeedPolicy, SearchLimits, TaskState, hybridize, run_dual_tracks, beam_rerank,
)


class ObjectEngineTests(unittest.TestCase):
    def test_extract_move_and_render_round_trip(self):
        grid: Grid = ((0, 0, 0, 0), (0, 2, 2, 0), (0, 0, 0, 0))
        scene = extract_scene(grid, background=0)
        self.assertEqual(len(scene.objects), 1)
        obj = scene.objects[0]
        moved = move(obj, 1, 0)
        self.assertEqual((obj.x, obj.y), (1, 1))
        self.assertEqual(render_scene(Scene((moved,), 3, 4)), (
            (0, 0, 0, 0), (0, 0, 2, 2), (0, 0, 0, 0)
        ))

    def test_grid_adapter_transforms_selected_object(self):
        grid: Grid = ((0, 0, 0, 0), (0, 2, 2, 0), (0, 0, 0, 0))
        self.assertTrue(object_exists(0)(grid))
        self.assertFalse(object_exists(1)(grid))
        moved = move_object(grid, 0, 1, 0)
        self.assertEqual(moved, ((0, 0, 0, 0), (0, 0, 2, 2), (0, 0, 0, 0)))

    def test_rotation_top_left_anchor_and_dimensions(self):
        obj = ObjectState("x", ((0, 0, 1), (0, 1, 1), (1, 0, 1)), 4, 3)
        rotated = rotate90(obj)
        self.assertEqual((rotated.x, rotated.y), (4, 3))
        self.assertEqual((rotated.height, rotated.width), (2, 2))
        self.assertEqual(set(rotated.pixels), {(0, 0, 1), (0, 1, 1), (1, 1, 1)})

    def test_center_anchor_rejects_half_cell_shift(self):
        obj = ObjectState("x", ((0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 0, 1)), 0, 0)
        with self.assertRaisesRegex(ValueError, "CENTER_ANCHOR_PARITY"):
            rotate90(obj, anchor="center")

    def test_reference_pixel_anchor_stays_fixed(self):
        obj = ObjectState("x", ((0, 0, 1), (0, 1, 1), (1, 0, 1)), 5, 6)
        rotated = rotate90(obj, anchor="reference_pixel", reference_pixel=(1, 0))
        self.assertEqual((rotated.y, rotated.x), (7, 5))

    def test_flip_and_scale_up(self):
        obj = ObjectState("x", ((0, 0, 3), (1, 0, 3)), 0, 0)
        flipped = flip_horizontal(obj)
        self.assertEqual(set(flipped.pixels), {(0, 0, 3), (1, 0, 3)})
        scaled = scale_up(obj, 2)
        self.assertEqual((scaled.height, scaled.width), (4, 2))
        self.assertEqual(len(scaled.pixels), 8)

    def test_renderer_rejects_collisions_and_out_of_bounds(self):
        a = ObjectState("a", ((0, 0, 1),), 0, 0)
        b = ObjectState("b", ((0, 0, 2),), 0, 0)
        with self.assertRaisesRegex(ValueError, "OBJECT_COLLISION"):
            render_scene(Scene((a, b), 2, 2), 2, 2)
        outside = ObjectState("c", ((0, 0, 1),), 2, 0)
        with self.assertRaisesRegex(ValueError, "OBJECT_OUT_OF_BOUNDS"):
            render_scene(Scene((outside,), 2, 2), 2, 2)

    def test_registry_rejects_duplicate_and_checks_catalog_size(self):
        registry = PrimitiveRegistry()
        noop = lambda grid: grid
        spec = PrimitiveSpec("noop", "identity")
        registry.register(spec, noop)
        with self.assertRaisesRegex(ValueError, "DUPLICATE_PRIMITIVE"):
            registry.register(spec, noop)
        with self.assertRaisesRegex(ValueError, "CATALOG_COUNT_MISMATCH"):
            registry.validate_catalog(200)

    def test_qbs_hybridization_requires_compatibility_tags(self):
        seeds = tuple(
            SeedPolicy(f"S{i}", f"idea-{i}", ("geometry",), ("object-motion",))
            for i in range(10)
        )
        hybrids = hybridize(seeds, limit=10)
        self.assertEqual(len(hybrids), 10)
        self.assertTrue(all(h.seed_id.startswith("HYBRID:") for h in hybrids))
        incompatible = hybridize((
            SeedPolicy("A", "a", ("geometry",), ("shape",)),
            SeedPolicy("B", "b", ("color",), ("count",)),
        ))
        self.assertEqual(incompatible, ())

    def test_dual_mcts_tracks_and_beam_reranking_are_separate(self):
        registry = PrimitiveRegistry()
        source: Grid = ((1, 0), (0, 0))
        source_two: Grid = ((0, 1), (0, 0))
        target: Grid = ((0, 1), (0, 0))
        target_two: Grid = ((0, 0), (0, 1))

        def clockwise(grid: Grid) -> Grid:
            return tuple(tuple(row[c] for row in reversed(grid)) for c in range(len(grid[0])))

        registry.register(PrimitiveSpec("rotate90", "geometry"), clockwise)
        seeds = tuple(
            SeedPolicy(f"S{i}", f"idea-{i}", ("geometry",), ("shape",))
            for i in range(10)
        )
        hybrids = hybridize(seeds, limit=10)
        initial = TaskState((source, source_two))
        expected = TaskState((target, target_two))
        score = lambda state: 1.0 if state == expected else 0.0
        a, b = run_dual_tracks(
            initial, registry, seeds, hybrids, score,
            SearchLimits(iterations=12, max_depth=2, candidate_cap=20, initial_width=1),
        )
        self.assertEqual(a.track, "seed")
        self.assertEqual(b.track, "hybrid")
        self.assertIsNot(a, b)
        merged = beam_rerank((a, b), registry, beam_width=4)
        self.assertTrue(merged)
        self.assertEqual(merged[0].state, expected)


if __name__ == "__main__":
    unittest.main()
