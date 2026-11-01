"""Exact-fit ARC inference with D4/color TTA and two structurally distinct attempts.

NumPy batches candidate programs in padded int8 tensors. Integer grids remain
canonical; -1 is reserved as padding and never a legal ARC color.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
import re
from typing import Iterable

import numpy as np

from .dsl import Instruction, Program
from .grid import Grid, as_grid
from .transforms import OPERATIONS

PAD = np.int8(-1)
MAX_SIDE = 30
_COLOR_SWAP = re.compile(r"^COLOR\\.SWAP\\.(\\d+)\\.(\\d+)$")
_GEOMETRY = (
    "REVERSE.HORIZONTAL",
    "REVERSE.VERTICAL",
    "ROTATE.180",
    "REFLECT.MAIN_DIAGONAL",
    "REFLECT.ANTI_DIAGONAL",
)
_OPERATION_CACHE: dict[tuple[tuple[int, int], bytes, str], tuple[Grid, tuple[int, int]]] = {}
_OPERATION_CACHE_LIMIT = 50000


@dataclass(frozen=True)
class ExactCandidate:
    program: tuple[str, ...]
    outputs: tuple[Grid, ...]
    complexity: int
    family: tuple[str, ...]


@dataclass(frozen=True)
class Attempt:
    output: Grid
    votes: int
    program: tuple[str, ...]
    family: tuple[str, ...]
    exact_training_fit: bool = True


def _family(program: Program | tuple[str, ...]) -> tuple[str, ...]:
    operations = program.canonical() if isinstance(program, Program) else program
    labels = set()
    for op in operations:
        if op in _GEOMETRY:
            labels.add("geometry")
        elif op.startswith("COLOR.SWAP."):
            labels.add("color-swap")
        elif op == "POLARITY.NEGATE":
            labels.add("polarity")
        elif op.startswith("PRTE."):
            labels.add("adaptive-polarity")
        else:
            labels.add(op.split(".")[0].lower())
    return tuple(sorted(labels)) or ("identity",)


def _pack(grids: Iterable[Grid]) -> tuple[np.ndarray, np.ndarray]:
    rows = tuple(as_grid(grid) for grid in grids)
    if not rows:
        raise ValueError("at least one grid is required")
    n = len(rows)
    tensor = np.full((n, MAX_SIDE, MAX_SIDE), PAD, dtype=np.int8)
    shapes = np.empty((n, 2), dtype=np.int16)
    for index, grid in enumerate(rows):
        h, w = len(grid), len(grid[0])
        if h > MAX_SIDE or w > MAX_SIDE:
            raise ValueError("ARC grid dimensions cannot exceed 30x30")
        tensor[index, :h, :w] = np.asarray(grid, dtype=np.int8)
        shapes[index] = (h, w)
    return tensor, shapes


def _unpack(tensor: np.ndarray, shapes: np.ndarray) -> tuple[Grid, ...]:
    return tuple(
        tuple(tuple(int(x) for x in row[:int(w)]) for row in tensor[i, :int(h)])
        for i, (h, w) in enumerate(shapes)
    )


def _cache_key(grid: Grid, operation: str):
    return ((len(grid), len(grid[0])), np.asarray(grid, dtype=np.int8).tobytes(), operation)


def _apply_one_cached(grid: Grid, operation: str) -> Grid:
    key = _cache_key(grid, operation)
    cached = _OPERATION_CACHE.get(key)
    if cached is not None:
        return cached[0]
    batch, shapes = _pack((grid,))
    result, result_shapes = batch_apply_operation(batch, shapes, operation, use_cache=False)
    output = _unpack(result, result_shapes)[0]
    if len(_OPERATION_CACHE) >= _OPERATION_CACHE_LIMIT:
        _OPERATION_CACHE.clear()
    _OPERATION_CACHE[key] = (output, (len(output), len(output[0])))
    return output


def batch_apply_operation(
    tensor: np.ndarray,
    shapes: np.ndarray,
    operation: str,
    *,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one operation across an (N,30,30) int8 batch."""
    if tensor.dtype != np.int8 or tensor.ndim != 3 or tensor.shape[1:] != (30, 30):
        raise ValueError("tensor must be int8 with shape (N,30,30)")
    if shapes.shape != (tensor.shape[0], 2):
        raise ValueError("shapes must have shape (N,2)")
    n = tensor.shape[0]
    out = np.full_like(tensor, PAD)
    new_shapes = shapes.copy()
    if use_cache and n:
        source_grids = _unpack(tensor, shapes)
        misses = []
        for index, source in enumerate(source_grids):
            cached = _OPERATION_CACHE.get(_cache_key(source, operation))
            if cached is None:
                misses.append(index)
            else:
                packed, packed_shapes = _pack((cached[0],))
                out[index] = packed[0]
                new_shapes[index] = packed_shapes[0]
        if misses:
            computed, computed_shapes = batch_apply_operation(
                tensor[misses], shapes[misses], operation, use_cache=False
            )
            out[misses] = computed
            new_shapes[misses] = computed_shapes
            for source_index, result_grid in zip(misses, _unpack(computed, computed_shapes)):
                if len(_OPERATION_CACHE) >= _OPERATION_CACHE_LIMIT:
                    _OPERATION_CACHE.clear()
                _OPERATION_CACHE[_cache_key(source_grids[source_index], operation)] = (
                    result_grid, (len(result_grid), len(result_grid[0]))
                )
        return out, new_shapes
    h = shapes[:, 0].astype(np.int64)
    w = shapes[:, 1].astype(np.int64)

    if operation in {"REVERSE.HORIZONTAL", "INVERSE.REVERSE.HORIZONTAL"}:
        cols = np.arange(30)[None, None, :]
        rows = np.arange(30)[None, :, None]
        valid = (rows < h[:, None, None]) & (cols < w[:, None, None])
        source_cols = w[:, None, None] - 1 - cols
        out[valid] = tensor[np.broadcast_to(np.arange(n)[:, None, None], valid.shape),
                            np.broadcast_to(rows, valid.shape),
                            np.broadcast_to(source_cols, valid.shape)][valid]
    elif operation in {"REVERSE.VERTICAL", "INVERSE.REVERSE.VERTICAL"}:
        cols = np.arange(30)[None, None, :]
        rows = np.arange(30)[None, :, None]
        valid = (rows < h[:, None, None]) & (cols < w[:, None, None])
        source_rows = h[:, None, None] - 1 - rows
        out[valid] = tensor[np.broadcast_to(np.arange(n)[:, None, None], valid.shape),
                            np.broadcast_to(source_rows, valid.shape),
                            np.broadcast_to(cols, valid.shape)][valid]
    elif operation in {"ROTATE.180", "INVERSE.ROTATE.180"}:
        cols = np.arange(30)[None, None, :]
        rows = np.arange(30)[None, :, None]
        valid = (rows < h[:, None, None]) & (cols < w[:, None, None])
        source_rows = h[:, None, None] - 1 - rows
        source_cols = w[:, None, None] - 1 - cols
        out[valid] = tensor[np.broadcast_to(np.arange(n)[:, None, None], valid.shape),
                            np.broadcast_to(source_rows, valid.shape),
                            np.broadcast_to(source_cols, valid.shape)][valid]
    elif operation in {"REFLECT.MAIN_DIAGONAL", "INVERSE.REFLECT.MAIN_DIAGONAL"}:
        out = np.transpose(tensor, (0, 2, 1)).copy()
        new_shapes[:, 0], new_shapes[:, 1] = shapes[:, 1], shapes[:, 0]
    elif operation in {"REFLECT.ANTI_DIAGONAL", "INVERSE.REFLECT.ANTI_DIAGONAL"}:
        rows = np.arange(30)[None, :, None]
        cols = np.arange(30)[None, None, :]
        valid = (rows < w[:, None, None]) & (cols < h[:, None, None])
        source_rows = h[:, None, None] - 1 - cols
        source_cols = w[:, None, None] - 1 - rows
        out[valid] = tensor[np.broadcast_to(np.arange(n)[:, None, None], valid.shape),
                            np.broadcast_to(source_rows, valid.shape),
                            np.broadcast_to(source_cols, valid.shape)][valid]
        new_shapes[:, 0], new_shapes[:, 1] = shapes[:, 1], shapes[:, 0]
    elif operation in {"POLARITY.NEGATE", "COLOR.SWAP.0.1"}:
        lut = np.arange(10, dtype=np.int8)
        lut[0], lut[1] = 1, 0
        valid = tensor >= 0
        out[valid] = lut[tensor[valid]]
    elif (match := _COLOR_SWAP.fullmatch(operation)) is not None:
        a, b = map(int, match.groups())
        if not (0 <= a <= 9 and 0 <= b <= 9):
            raise ValueError("color swap labels must be in 0..9")
        lut = np.arange(10, dtype=np.int8)
        lut[a], lut[b] = b, a
        valid = tensor >= 0
        out[valid] = lut[tensor[valid]]
    elif operation == "PRTE.POLARITY.REVERSE":
        # scipy-backed component extraction is used elsewhere; PRTE itself
        # depends on each grid's dominant colors, so only this small reduction
        # loops over the batch. Cell transforms remain vectorized.
        for i, (height, width) in enumerate(shapes):
            values = tensor[i, :height, :width].astype(np.int64)
            counts = np.bincount(values.ravel(), minlength=10)
            background = int(np.flatnonzero(counts == counts.max())[0])
            foreground_counts = counts.copy()
            foreground_counts[background] = 0
            if foreground_counts.max() == 0:
                out[i, :height, :width] = values
            else:
                foreground = int(np.flatnonzero(foreground_counts == foreground_counts.max())[0])
                lut = np.arange(10, dtype=np.int8)
                lut[background], lut[foreground] = foreground, background
                out[i, :height, :width] = lut[values]
    else:
        raise ValueError(f"unsupported vectorized operation: {operation}")

    return out, new_shapes


def batch_apply_programs(grid: Grid, programs: Iterable[Program]) -> tuple[Grid, ...]:
    """Run many programs in parallel, grouping each depth by next operation."""
    candidates = tuple(programs)
    if not candidates:
        return ()
    base, base_shape = _pack((grid,))
    tensor = np.repeat(base, len(candidates), axis=0)
    shapes = np.repeat(base_shape, len(candidates), axis=0)
    max_depth = max(program.complexity for program in candidates)
    for depth in range(max_depth):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, program in enumerate(candidates):
            if depth < program.complexity:
                groups[program.instructions[depth].operation].append(index)
        for operation, indices in groups.items():
            subset, subset_shapes = batch_apply_operation(
                tensor[indices], shapes[indices], operation
            )
            tensor[indices] = subset
            shapes[indices] = subset_shapes
    return _unpack(tensor, shapes)


def _candidate_programs(examples: tuple[tuple[Grid, Grid], ...], max_depth: int) -> tuple[Program, ...]:
    palette = sorted({color for pair in examples for grid in pair
                      for row in grid for color in row})
    operations = list(OPERATIONS)
    operations.extend(_GEOMETRY)
    operations.extend(f"COLOR.SWAP.{a}.{b}" for a, b in combinations(palette, 2))
    operations = tuple(dict.fromkeys(operations))
    programs = {(): Program()}
    for depth in range(1, max_depth + 1):
        for chain in product(operations, repeat=depth):
            programs.setdefault(chain, Program(tuple(Instruction(op) for op in chain)))
    return tuple(programs[key] for key in sorted(programs, key=lambda p: (len(p), p)))


def exact_fit_candidates(
    examples: Iterable[tuple[Grid, Grid]],
    *,
    max_depth: int = 2,
) -> tuple[ExactCandidate, ...]:
    """Return only programs reproducing every supplied training output exactly."""
    rows = tuple((as_grid(inp), as_grid(out)) for inp, out in examples)
    if not rows:
        raise ValueError("at least one training pair is required")
    if not 0 <= max_depth <= 2:
        raise ValueError("max_depth must be between 0 and 2")
    candidates = _candidate_programs(rows, max_depth)
    survivors = list(candidates)
    outputs_by_program: dict[tuple[str, ...], list[Grid]] = {
        program.canonical(): [] for program in survivors
    }
    for source, expected in rows:
        predicted = batch_apply_programs(source, survivors)
        keep = []
        for program, output in zip(survivors, predicted):
            if output == expected and 1 <= len(output) <= MAX_SIDE and 1 <= len(output[0]) <= MAX_SIDE:
                outputs_by_program[program.canonical()].append(output)
                keep.append(program)
        survivors = keep
        if not survivors:
            return ()
    # Deduplicate equivalent training behavior; retain the shortest canonical program.
    unique: dict[tuple[Grid, ...], Program] = {}
    for program in survivors:
        behavior = tuple(outputs_by_program[program.canonical()])
        previous = unique.get(behavior)
        if previous is None or (program.complexity, program.canonical()) < (
            previous.complexity, previous.canonical()
        ):
            unique[behavior] = program
    result = []
    for behavior, program in unique.items():
        result.append(ExactCandidate(
            program=program.canonical(),
            outputs=behavior,
            complexity=program.complexity,
            family=_family(program),
        ))
    return tuple(sorted(result, key=lambda item: (item.complexity, item.program)))


def _geometry(grid: Grid, rotation: int, flip: bool) -> Grid:
    array = np.asarray(grid, dtype=np.int8)
    if flip:
        array = np.flip(array, axis=1)
    array = np.rot90(array, k=rotation)
    return tuple(tuple(int(x) for x in row) for row in array.tolist())


def _inverse_geometry(grid: Grid, rotation: int, flip: bool) -> Grid:
    array = np.asarray(grid, dtype=np.int8)
    array = np.rot90(array, k=(-rotation) % 4)
    if flip:
        array = np.flip(array, axis=1)
    return tuple(tuple(int(x) for x in row) for row in array.tolist())


def _color_maps(palette: tuple[int, ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    if len(palette) < 2:
        return (tuple((color, color) for color in palette),)
    variants = [
        palette,
        palette[1:] + palette[:1],
        tuple(reversed(palette)),
    ]
    maps = []
    for variant in variants:
        maps.append(tuple(zip(palette, variant)))
    return tuple(dict.fromkeys(maps))


def _map_colors(grid: Grid, mapping: tuple[tuple[int, int], ...]) -> Grid:
    lut = np.arange(10, dtype=np.int8)
    for source, target in mapping:
        lut[source] = target
    array = np.asarray(grid, dtype=np.int8)
    mapped = lut[array]
    return tuple(tuple(int(x) for x in row) for row in mapped.tolist())


def _invert_color_map(grid: Grid, mapping: tuple[tuple[int, int], ...]) -> Grid:
    inverse = tuple((target, source) for source, target in mapping)
    return _map_colors(grid, inverse)


def _sane_output(grid: Grid, allowed_palette: frozenset[int]) -> bool:
    if not grid or not grid[0] or len(grid) > MAX_SIDE or len(grid[0]) > MAX_SIDE:
        return False
    return all(color in allowed_palette for row in grid for color in row)


def solve_two_attempts(
    examples: Iterable[tuple[Grid, Grid]],
    test_input: Grid,
    *,
    max_depth: int = 2,
) -> dict:
    """TTA vote across D4 geometry and color relabelings; emit at most two attempts."""
    rows = tuple((as_grid(inp), as_grid(out)) for inp, out in examples)
    test_grid = as_grid(test_input)
    if not rows:
        raise ValueError("at least one training pair is required")
    palette = tuple(sorted({color for pair in rows for grid in pair
                            for row in grid for color in row}))
    allowed_palette = frozenset(palette)
    votes: dict[Grid, int] = defaultdict(int)
    supporting: dict[Grid, dict[tuple[str, ...], ExactCandidate]] = defaultdict(dict)
    views = 0
    views_with_fit = 0
    candidates_checked = 0

    for rotation in range(4):
        for flip in (False, True):
            for color_map in _color_maps(palette):
                views += 1
                augmented = tuple(
                    (_map_colors(_geometry(inp, rotation, flip), color_map),
                     _map_colors(_geometry(out, rotation, flip), color_map))
                    for inp, out in rows
                )
                candidates = exact_fit_candidates(augmented, max_depth=max_depth)
                candidates_checked += len(candidates)
                if not candidates:
                    continue
                views_with_fit += 1
                augmented_test = _map_colors(_geometry(test_grid, rotation, flip), color_map)
                # Each distinct exact-fit behavior gets one vote per augmented view.
                per_view: dict[Grid, ExactCandidate] = {}
                for candidate in candidates:
                    program = Program(tuple(Instruction(op) for op in candidate.program))
                    predicted_augmented = batch_apply_programs(augmented_test, (program,))[0]
                    prediction = _inverse_geometry(predicted_augmented, rotation, flip)
                    prediction = _invert_color_map(prediction, color_map)
                    if not _sane_output(prediction, allowed_palette):
                        continue
                    previous = per_view.get(prediction)
                    if previous is None or (candidate.complexity, candidate.program) < (
                        previous.complexity, previous.program
                    ):
                        per_view[prediction] = candidate
                for prediction, candidate in per_view.items():
                    votes[prediction] += 1
                    supporting[prediction].setdefault(candidate.program, candidate)

    ranked_outputs = sorted(
        votes,
        key=lambda output: (
            -votes[output],
            min(c.complexity for c in supporting[output].values()),
            min(c.program for c in supporting[output].values()),
            output,
        ),
    )
    attempts: list[Attempt] = []
    if ranked_outputs:
        first = ranked_outputs[0]
        first_program = min(
            supporting[first].values(),
            key=lambda c: (c.complexity, c.program),
        )
        attempts.append(Attempt(first, votes[first], first_program.program, first_program.family))
        for output in ranked_outputs[1:]:
            distinct = [
                candidate for candidate in supporting[output].values()
                if candidate.family != first_program.family
                and candidate.program != first_program.program
            ]
            if not distinct:
                continue
            second_program = min(distinct, key=lambda c: (c.complexity, c.program))
            attempts.append(Attempt(output, votes[output], second_program.program, second_program.family))
            break

    return {
        "method": "exact-training-filter + D4/color-permutation TTA + inverse voting",
        "views": views,
        "views_with_exact_fit": views_with_fit,
        "exact_fit_candidates_seen": candidates_checked,
        "unique_outputs": len(votes),
        "attempts": [
            {
                "output": [list(row) for row in attempt.output],
                "votes": attempt.votes,
                "program": list(attempt.program),
                "program_family": list(attempt.family),
                "exact_training_fit": attempt.exact_training_fit,
            }
            for attempt in attempts
        ],
        "attempt_2_available": len(attempts) == 2,
        "acceptance_rule": "Every contributing program exactly reproduces every transformed training output.",
        "ranking_rule": "Majority vote first; shortest exact-fit program breaks ties.",
        "second_attempt_rule": "Highest-voted different output with a different transformation-family signature.",
    }
