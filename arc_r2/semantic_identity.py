"""Conservative cross-grid object identity matching."""
from __future__ import annotations
from dataclasses import dataclass
from math import hypot
from .grid import Grid, as_grid
from .objects import GridObject, detect_objects
from .object_identity_dsl import IdentityState

@dataclass(frozen=True)
class IdentityMatch:
    object_id: str
    target_index: int
    score: float

@dataclass(frozen=True)
class IdentityMatchResult:
    matches: tuple[IdentityMatch, ...]
    unmatched_reference: tuple[str, ...]
    unmatched_target: tuple[int, ...]
    ambiguous_reference: tuple[str, ...]

def _canonical_shape(obj: GridObject) -> tuple[tuple[float, float], ...]:
    points = [(r - obj.bbox[0], c - obj.bbox[1]) for r, c in obj.cells]
    variants = []
    for reflected in (False, True):
        for turns in range(4):
            transformed = []
            for r, c in points:
                if reflected:
                    c = -c
                for _ in range(turns):
                    r, c = c, -r
                transformed.append((r, c))
            min_r = min(p[0] for p in transformed)
            min_c = min(p[1] for p in transformed)
            shifted = [(r - min_r, c - min_c) for r, c in transformed]
            span = max(max(r for r, _ in shifted), max(c for _, c in shifted), 1)
            variants.append(tuple(sorted((round(r / span, 3), round(c / span, 3)) for r, c in shifted)))
    return min(variants)

def _shape_similarity(a: GridObject, b: GridObject) -> float:
    sa, sb = set(_canonical_shape(a)), set(_canonical_shape(b))
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0

def _ratio_similarity(a: float, b: float) -> float:
    return min(a, b) / max(a, b) if a > 0 and b > 0 else 0.0

def _position_similarity(a: GridObject, b: GridObject, a_size: tuple[int, int], b_size: tuple[int, int]) -> float:
    ar = ((a.bbox[0] + a.bbox[2]) / 2) / max(a_size[0] - 1, 1)
    ac = ((a.bbox[1] + a.bbox[3]) / 2) / max(a_size[1] - 1, 1)
    br = ((b.bbox[0] + b.bbox[2]) / 2) / max(b_size[0] - 1, 1)
    bc = ((b.bbox[1] + b.bbox[3]) / 2) / max(b_size[1] - 1, 1)
    return max(0.0, 1.0 - hypot(ar - br, ac - bc) / 2**0.5)

def _appearance_score(a: GridObject, b: GridObject) -> float:
    """Score identity evidence that does not depend on location."""
    color = 1.0 if a.color == b.color else 0.0
    shape = _shape_similarity(a, b)
    size = _ratio_similarity(a.size, b.size)
    # Compare rotation-invariant aspect ratios: a 90-degree rotation swaps
    # width and height but must not make the same object look less similar.
    a_aspect = min(a.width, a.height) / max(a.width, a.height)
    b_aspect = min(b.width, b.height) / max(b.width, b.height)
    aspect = _ratio_similarity(a_aspect, b_aspect)
    return 0.30 * color + 0.35 * shape + 0.15 * size + 0.10 * aspect

def _score(a: GridObject, b: GridObject, a_size: tuple[int, int], b_size: tuple[int, int]) -> float:
    return _appearance_score(a, b) + 0.10 * _position_similarity(a, b, a_size, b_size)

def match_identities(reference: IdentityState, target_grid: Grid, *, threshold: float = 0.82, margin: float = 0.12) -> IdentityMatchResult:
    """Match IDs to a new grid; weak or non-unique candidates are left unmatched.

    Location may break a tie only when appearance evidence is itself sufficiently
    distinctive. If multiple references or targets are indistinguishable by
    color/shape/size/aspect, the matcher abstains instead of inventing identity.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not 0.0 <= margin <= 1.0:
        raise ValueError("margin must be between 0 and 1")
    grid = as_grid(target_grid)
    targets = detect_objects(grid)
    refs = list(reference.objects)
    target_size = (len(grid), len(grid[0]))
    scores = [[_score(obj, target, (reference.height, reference.width), target_size) for target in targets] for _, obj in refs]
    appearance = [[_appearance_score(obj, target) for target in targets] for _, obj in refs]
    accepted, ambiguous, used_targets, accepted_refs = [], [], set(), set()
    for ri, (object_id, _) in enumerate(refs):
        ranked = sorted(range(len(targets)), key=lambda ti: (-scores[ri][ti], ti))
        if not ranked:
            continue
        best = ranked[0]
        best_score = scores[ri][best]
        second = scores[ri][ranked[1]] if len(ranked) > 1 else 0.0
        competing = max((scores[other][best] for other in range(len(refs)) if other != ri), default=0.0)
        best_appearance = appearance[ri][best]
        second_appearance = max((appearance[ri][ti] for ti in range(len(targets)) if ti != best), default=0.0)
        competing_appearance = max((appearance[other][best] for other in range(len(refs)) if other != ri), default=0.0)
        if best_score < threshold:
            continue
        # A location-only advantage is not enough to distinguish repeated objects.
        if (best_score - second < margin or best_score - competing < margin
                or best_appearance - second_appearance < margin
                or best_appearance - competing_appearance < margin
                or best in used_targets):
            ambiguous.append(object_id)
            continue
        accepted.append(IdentityMatch(object_id, best, round(best_score, 6)))
        accepted_refs.add(object_id)
        used_targets.add(best)
    return IdentityMatchResult(
        tuple(accepted),
        tuple(object_id for object_id, _ in refs if object_id not in accepted_refs),
        tuple(i for i in range(len(targets)) if i not in used_targets),
        tuple(ambiguous),
    )
