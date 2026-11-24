"""Multi-scale structural perception for ARC grids.

Shape Radar reports observable geometry only. It does not infer the task rule or
modify the grid. Coordinates are zero-based; object shape signatures ignore
translation, and canonical signatures additionally ignore rotation/reflection.
"""
from __future__ import annotations

from collections import deque
from itertools import combinations
import json

from .primitives import Grid, as_grid, background_color, find_objects


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _dihedral_variants(cells: set[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    transforms = (
        lambda r, c: (r, c),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (-r, -c),
        lambda r, c: (c, r),
        lambda r, c: (c, -r),
        lambda r, c: (-c, r),
        lambda r, c: (-c, -r),
    )
    return tuple(sorted({_normalize({fn(r, c) for r, c in cells}) for fn in transforms}))


def canonical_shape_signature(cells) -> str:
    """Translation-, rotation-, and reflection-invariant signature for a cell set."""
    points = set(cells)
    if not points:
        raise ValueError("shape must contain at least one cell")
    return json.dumps(min(_dihedral_variants(points)), separators=(",", ":"))


def _has_hole(cells: set[tuple[int, int]]) -> int:
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    r1, c1 = max(r for r, _ in cells), max(c for _, c in cells)
    empty = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
             if (r, c) not in cells}
    exterior = set()
    queue = deque()
    for r, c in empty:
        if r in (r0, r1) or c in (c0, c1):
            exterior.add((r, c))
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nxt in empty and nxt not in exterior:
                exterior.add(nxt)
                queue.append(nxt)
    unseen = empty - exterior
    holes = 0
    while unseen:
        holes += 1
        seed = unseen.pop()
        queue.append(seed)
        while queue:
            r, c = queue.popleft()
            for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nxt in unseen:
                    unseen.remove(nxt)
                    queue.append(nxt)
    return holes


def _object_features(obj, index: int) -> dict:
    cells = set(obj.cells)
    r0, c0, r1, c1 = obj.bbox
    local = set(_normalize(cells))
    height, width = obj.height, obj.width
    horizontal = {(r, width - 1 - c) for r, c in local} == local
    vertical = {(height - 1 - r, c) for r, c in local} == local
    diagonal = height == width and {(c, r) for r, c in local} == local
    anti_diagonal = height == width and {
        (height - 1 - c, width - 1 - r) for r, c in local
    } == local
    perimeter = sum(
        (r + dr, c + dc) not in cells
        for r, c in cells
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )
    return {
        "id": f"obj_{index:02d}",
        "color": obj.color,
        "cells": [list(p) for p in sorted(cells)],
        "size": obj.size,
        "bbox": [r0, c0, r1, c1],
        "height": height,
        "width": width,
        "aspect_ratio": round(width / height, 6),
        "perimeter": perimeter,
        "holes": _has_hole(cells),
        "symmetry": {
            "horizontal": horizontal,
            "vertical": vertical,
            "diagonal": diagonal,
            "anti_diagonal": anti_diagonal,
        },
        "orientation_signature": json.dumps(sorted(local), separators=(",", ":")),
        "canonical_shape": canonical_shape_signature(cells),
    }


def _bbox_relation(a: dict, b: dict) -> dict:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    a_center2 = (ar0 + ar1, ac0 + ac1)
    b_center2 = (br0 + br1, bc0 + bc1)
    overlap = not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)
    contains = (
        (ar0 <= br0 and ac0 <= bc0 and ar1 >= br1 and ac1 >= bc1)
        or (br0 <= ar0 and bc0 <= ac0 and br1 >= ar1 and bc1 >= ac1)
    )
    return {
        "objects": [a["id"], b["id"]],
        "bbox_row_gap": row_gap,
        "bbox_col_gap": col_gap,
        "bbox_overlap": overlap,
        "bbox_contains": contains,
        "same_row_center": a_center2[0] == b_center2[0],
        "same_column_center": a_center2[1] == b_center2[1],
        "same_shape": a["canonical_shape"] == b["canonical_shape"],
    }


def scan_grid(grid, connectivity: int = 4) -> dict:
    """Return a JSON-serializable, multi-scale structural scan of an ARC grid.

    The scan includes cell/palette statistics, 4- and 8-connected object
    decompositions, shape geometry, repeated-shape groups, and pair relations.
    The requested connectivity controls the primary object/relation layer.
    """
    source: Grid = as_grid(grid)
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    height, width = len(source), len(source[0])
    bg = background_color(source)
    counts = {}
    for row in source:
        for color in row:
            counts[color] = counts.get(color, 0) + 1

    primary = tuple(_object_features(obj, i) for i, obj in enumerate(
        find_objects(source, connectivity=connectivity)
    ))
    secondary_connectivity = 8 if connectivity == 4 else 4
    secondary = tuple(_object_features(obj, i) for i, obj in enumerate(
        find_objects(source, connectivity=secondary_connectivity)
    ))

    groups = {}
    for obj in primary:
        groups.setdefault(obj["canonical_shape"], []).append(obj["id"])
    repeated = [
        {"canonical_shape": signature, "object_ids": ids, "count": len(ids)}
        for signature, ids in sorted(groups.items()) if len(ids) > 1
    ]
    relations = [_bbox_relation(a, b) for a, b in combinations(primary, 2)]

    return {
        "schema": "arc-shape-radar/v1",
        "grid": {
            "height": height,
            "width": width,
            "background": bg,
            "palette": sorted(counts),
            "color_counts": {str(k): counts[k] for k in sorted(counts)},
        },
        "scan": {
            "primary_connectivity": connectivity,
            "secondary_connectivity": secondary_connectivity,
            "cells_examined": height * width,
            "primary_object_count": len(primary),
            "secondary_object_count": len(secondary),
        },
        "objects": list(primary),
        "alternate_connectivity_objects": list(secondary),
        "repeated_shapes": repeated,
        "relationships": relations,
        "interpretation": None,
    }
