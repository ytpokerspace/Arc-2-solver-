"""Simple object-to-object spatial relation signatures."""
from __future__ import annotations
from .objects import GridObject

def relation(a: GridObject, b: GridObject) -> frozenset[str]:
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    result = set()
    if ac1 < bc0: result.add("LEFT_OF")
    if ac0 > bc1: result.add("RIGHT_OF")
    if ar1 < br0: result.add("ABOVE")
    if ar0 > br1: result.add("BELOW")
    if ar0 == br0: result.add("TOP_ALIGNED")
    if ar1 == br1: result.add("BOTTOM_ALIGNED")
    if ac0 == bc0: result.add("LEFT_ALIGNED")
    if ac1 == bc1: result.add("RIGHT_ALIGNED")
    if a.cells & b.cells: result.add("OVERLAPS")
    return frozenset(result)

def relation_graph(objects: tuple[GridObject, ...]) -> tuple[tuple[int, int, frozenset[str]], ...]:
    edges = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i != j:
                rel = relation(a, b)
                if rel: edges.append((i, j, rel))
    return tuple(edges)
