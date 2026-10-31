from __future__ import annotations

import pytest

from arc_r2.grid import Grid
from arc_r2.object_engine import (
    ObjectState, Scene, extract_scene, move, rotate90,
    flip_horizontal, scale_up, render_scene,
)
from arc_r2.primitive_engine import PrimitiveRegistry, PrimitiveSpec
from arc_r2.qbs_search import (
    SeedPolicy, SearchLimits, hybridize, run_dual_tracks, beam_rerank,
)


def test_extract_move_and_render_round_trip():
    grid: Grid = ((0, 0, 0, 0), (0, 2, 2, 0), (0, 0, 0, 0))
    scene = extract_scene(grid, background=0)
    assert len(scene.objects) == 1
    obj = scene.objects[0]
    moved = move(obj, 1, 0)
    assert (obj.x, obj.y) == (1, 1)  # original remains immutable
    assert render_scene(Scene((moved,), 3, 4)) == (
        (0, 0, 0, 0), (0, 0, 2, 2), (0, 0, 0, 0)
    )


def test_rotation_top_left_anchor_and_dimensions():
    obj = ObjectState("x", ((0, 0, 1), (0, 1, 1), (1, 0, 1)), 4, 3)
    rotated = rotate90(obj)
    assert (rotated.x, rotated.y) == (4, 3)
    assert (rotated.height, rotated.width) == (2, 2)
    assert set(rotated.pixels) == {(0, 0, 1), (0, 1, 1), (1, 1, 1)}


def test_center_anchor_rejects_half_cell_shift():
    obj = ObjectState("x", ((0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 0, 1)), 0, 0)
    with pytest.raises(ValueError, match="CENTER_ANCHOR_PARITY"):
        rotate90(obj, anchor="center")


def test_reference_pixel_anchor_stays_fixed():
    obj = ObjectState("x", ((0, 0, 1), (0, 1, 1), (1, 0, 1)), 5, 6)
    rotated = rotate90(obj, anchor="reference_pixel", reference_pixel=(1, 0))
    # Original reference pixel (7, 5) maps to transformed local (0, 0).
    assert (rotated.y, rotated.x) == (7, 5)


def test_flip_and_scale_up():
    obj = ObjectState("x", ((0, 0, 3), (1, 0, 3)), 0, 0)
    flipped = flip_horizontal(obj)
    assert set(flipped.pixels) == {(0, 0, 3), (1, 0, 3)}
    scaled = scale_up(obj, 2)
    assert scaled.height == 4
    assert scaled.width == 2
    assert len(scaled.pixels) == 8


def test_renderer_rejects_collisions_and_out_of_bounds():
    a = ObjectState("a", ((0, 0, 1),), 0, 0)
    b = ObjectState("b", ((0, 0, 2),), 0, 0)
    with pytest.raises(ValueError, match="OBJECT_COLLISION"):
        render_scene(Scene((a, b), 2, 2), 2, 2)
    with pytest.raises(ValueError, match="OBJECT_OUT_OF_BOUNDS"):
        render_scene(Scene((ObjectState("c", ((0, 0, 1),), 2, 0),), 2, 2), 2, 2)


def test_registry_rejects_duplicate_and_checks_catalog_size():
    registry = PrimitiveRegistry()
    noop = lambda grid: grid
    spec = PrimitiveSpec("noop", "identity")
    registry.register(spec, noop)
    with pytest.raises(ValueError, match="DUPLICATE_PRIMITIVE"):
        registry.register(spec, noop)
    with pytest.raises(ValueError, match="CATALOG_COUNT_MISMATCH"):
        registry.validate_catalog(200)


def test_qbs_hybridization_requires_compatibility_tags():
    seeds = tuple(
        SeedPolicy(f"S{i}", f"idea-{i}", ("geometry",), ("object-motion",))
        for i in range(10)
    )
    hybrids = hybridize(seeds, limit=10)
    assert len(hybrids) == 10
    assert all(h.seed_id.startswith("HYBRID:") for h in hybrids)
    incompatible = hybridize((
        SeedPolicy("A", "a", ("geometry",), ("shape",)),
        SeedPolicy("B", "b", ("color",), ("count",)),
    ))
    assert incompatible == ()


def test_dual_mcts_tracks_and_beam_reranking_are_separate():
    registry = PrimitiveRegistry()
    source: Grid = ((1, 0), (0, 0))
    target: Grid = ((0, 1), (0, 0))

    def clockwise(grid: Grid) -> Grid:
        return tuple(tuple(row[c] for row in reversed(grid)) for c in range(len(grid[0])))

    registry.register(PrimitiveSpec("rotate90", "geometry"), clockwise)
    seeds = tuple(
        SeedPolicy(f"S{i}", f"idea-{i}", ("geometry",), ("shape",))
        for i in range(10)
    )
    hybrids = hybridize(seeds, limit=10)
    score = lambda grid: 1.0 if grid == target else 0.0
    a, b = run_dual_tracks(
        source, registry, seeds, hybrids, score,
        SearchLimits(iterations=12, max_depth=2, candidate_cap=20, initial_width=1),
    )
    assert a.track == "seed"
    assert b.track == "hybrid"
    assert a is not b
    merged = beam_rerank((a, b), registry, beam_width=4)
    assert merged
    assert merged[0].grid == target
