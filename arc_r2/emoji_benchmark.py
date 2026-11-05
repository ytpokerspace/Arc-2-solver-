"""Compare integer, emoji-grid, and emoji-object round-trip representations.

This is a codec benchmark, not an ARC solving benchmark. It measures only
representation conversion and exact reconstruction.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .emoji_codec import (
    decode_grid,
    decode_object_graph,
    encode_grid,
    encode_object_graph,
)
from .grid import Grid, as_grid


DEMO_GRIDS: tuple[Grid, ...] = (
    ((0, 0, 0, 0), (0, 2, 2, 0), (0, 0, 0, 0)),
    ((1, 1, 1), (1, 3, 1), (1, 1, 1)),
    ((0, 4, 0, 5, 0), (4, 4, 0, 5, 5), (0, 0, 0, 0)),
)


def load_grids(path: str, limit: int | None = None) -> tuple[Grid, ...]:
    """Read train input/output grids from a standard ARC JSON task file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = list(payload.values()) if isinstance(payload, dict) else payload
    grids: list[Grid] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("each ARC task must be a JSON object")
        for pair in task.get("train", []):
            if "input" in pair:
                grids.append(as_grid(pair["input"]))
            if "output" in pair:
                grids.append(as_grid(pair["output"]))
        if limit is not None and len(grids) >= limit:
            return tuple(grids[:limit])
    if not grids:
        raise ValueError("dataset contains no training input/output grids")
    return tuple(grids)


def _json_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def benchmark(grids: tuple[Grid, ...], repeats: int = 20) -> dict[str, object]:
    """Measure exact round-trip time and JSON character count for A/B/C."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    strategies: tuple[tuple[str, Callable[[Grid], object], Callable[[object], Grid],
                             Callable[[object], object]], ...] = (
        ("integer", lambda grid: grid, lambda value: as_grid(value),
         lambda value: [list(row) for row in value]),
        ("emoji_grid", lambda grid: encode_grid(grid), lambda value: decode_grid(value),
         lambda value: [list(row) for row in value.rows]),
        ("emoji_object_graph", lambda grid: encode_object_graph(grid),
         lambda value: decode_object_graph(value),
         lambda value: asdict(value)),
    )
    results: dict[str, object] = {
        "scope": "codec_round_trip_only_not_solver_accuracy_or_search_speed",
        "grid_count": len(grids),
        "repeats": repeats,
        "representations": {},
    }
    reports: dict[str, object] = {}
    for name, encoder, decoder, serializer in strategies:
        encoded_values = [encoder(grid) for grid in grids]
        exact = all(decoder(encoded) == original
                    for original, encoded in zip(grids, encoded_values))
        size_chars = sum(_json_size(serializer(value)) for value in encoded_values)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for original in grids:
                restored = decoder(encoder(original))
                if restored != original:
                    raise AssertionError(f"{name} failed exact reconstruction")
            samples.append(time.perf_counter_ns() - start)
        reports[name] = {
            "exact_reconstruction": exact,
            "total_serialized_json_chars": size_chars,
            "median_round_trip_ns_per_batch": int(statistics.median(samples)),
            "samples": len(samples),
        }
    results["representations"] = reports
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="optional local ARC JSON task file")
    parser.add_argument("--limit", type=int, default=100,
                        help="maximum number of train input/output grids")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()

    grids = load_grids(args.dataset, args.limit) if args.dataset else DEMO_GRIDS
    report = benchmark(grids, args.repeats)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
