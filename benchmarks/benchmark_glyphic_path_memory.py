"""Reproducible microbenchmark for Glyphic Path Optimization memory.

Measures write throughput, tag retrieval latency, hydration latency, serialized
storage, and exact evidence reconstruction. This is a memory-system benchmark,
not an ARC-AGI accuracy benchmark.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from arc_r2.glyphic_path_memory import GlyphicPathMemory, make_event


def run(paths: int = 250, queries: int = 100) -> dict:
    if paths < 1 or queries < 1:
        raise ValueError("paths and queries must be positive")
    with tempfile.TemporaryDirectory() as directory:
        file_path = Path(directory) / "gpo.json"
        memory = GlyphicPathMemory(file_path)
        expected = {}
        start_write = time.perf_counter()
        for index in range(paths):
            tags = [f"color:{index % 10}", f"shape:{index % 7}", f"rule:{index % 5}"]
            evidence = {
                "task_id": f"synthetic-{index:06d}",
                "input": [[(index + r + c) % 10 for c in range(5)] for r in range(5)],
                "output": [[(index + r + c + 1) % 10 for c in range(5)] for r in range(5)],
                "trace": ["observe", "extract", "transform", "verify"],
            }
            expected[f"path-{index:06d}"] = evidence
            memory.add_path(
                f"path-{index:06d}",
                [
                    make_event("observe", "inspect grid", task=index),
                    make_event("object", "extract components"),
                    make_event("relation", "compare structures"),
                    make_event("transform", "apply candidate"),
                    make_event("verify", "check exact output"),
                ],
                tags=tags,
                status="verified",
                source_evidence=evidence,
            )
        write_seconds = time.perf_counter() - start_write
        storage_bytes = file_path.stat().st_size

        retrieval_samples = []
        retrieved = []
        for index in range(queries):
            tag = f"color:{index % 10}"
            start = time.perf_counter()
            result = memory.retrieve([tag], limit=10)
            retrieval_samples.append(time.perf_counter() - start)
            retrieved.append(result)

        hydration_samples = []
        exact_reconstructions = 0
        for index in range(min(paths, queries)):
            path_id = f"path-{index:06d}"
            start = time.perf_counter()
            restored = memory.hydrate(path_id)
            hydration_samples.append(time.perf_counter() - start)
            exact_reconstructions += restored == expected[path_id]

        returned = sum(len(items) for items in retrieved)
        return {
            "benchmark": "glyphic_path_memory_microbenchmark",
            "scope": "synthetic memory workload; not ARC task accuracy",
            "paths_written": paths,
            "retrieval_queries": queries,
            "write_seconds": round(write_seconds, 6),
            "write_paths_per_second": round(paths / write_seconds, 3) if write_seconds else None,
            "storage_bytes": storage_bytes,
            "storage_bytes_per_path": round(storage_bytes / paths, 2),
            "retrieval_latency_ms_mean": round(statistics.mean(retrieval_samples) * 1000, 4),
            "retrieval_latency_ms_p50": round(statistics.median(retrieval_samples) * 1000, 4),
            "retrieval_results_returned": returned,
            "hydration_latency_ms_mean": round(statistics.mean(hydration_samples) * 1000, 4) if hydration_samples else 0,
            "hydration_exact_reconstructions": exact_reconstructions,
            "hydration_checks": len(hydration_samples),
            "hydration_exact_rate": exact_reconstructions / len(hydration_samples) if hydration_samples else 0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=250)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.paths, args.queries)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
