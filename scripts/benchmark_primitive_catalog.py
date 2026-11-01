"""Structural validation for the authoritative ARC primitive catalog.

This validates catalog integrity only. Runtime implementation coverage is
reported as unavailable until the project exposes a populated registry factory.
It is not an ARC task-solving accuracy benchmark.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


def benchmark(catalog_path: Path) -> dict:
    raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("groups"), dict):
        raise ValueError("INVALID_CATALOG_STRUCTURE")

    groups = raw["groups"]
    names: list[str] = []
    category_counts: dict[str, int] = {}
    for category, entry in groups.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("primitives"), list):
            raise ValueError(f"INVALID_CATEGORY:{category}")
        items = entry["primitives"]
        if not all(isinstance(item, str) and item for item in items):
            raise ValueError(f"INVALID_PRIMITIVE_NAME:{category}")
        declared = entry.get("count")
        if declared != len(items):
            raise ValueError(
                f"CATEGORY_COUNT_MISMATCH:{category}:declared={declared}:actual={len(items)}"
            )
        category_counts[category] = len(items)
        names.extend(items)

    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    expected = int(raw.get("expected_unique_primitive_count", 0))
    declared_total = int(raw.get("total_count", -1))
    if duplicates:
        raise ValueError(f"DUPLICATE_PRIMITIVES:{','.join(duplicates)}")
    if len(names) != expected or len(set(names)) != expected:
        raise ValueError(
            f"UNIQUE_COUNT_MISMATCH:expected={expected}:actual={len(set(names))}"
        )
    if declared_total != len(names):
        raise ValueError(f"TOTAL_COUNT_MISMATCH:declared={declared_total}:actual={len(names)}")

    return {
        "benchmark": "arc_primitive_catalog_structural",
        "catalog_id": raw.get("catalog_id"),
        "catalog_valid": True,
        "category_count": len(groups),
        "category_counts": category_counts,
        "catalog_primitive_count": len(names),
        "unique_primitive_count": len(set(names)),
        "duplicate_count": len(duplicates),
        "runtime_registry_status": "not_wired",
        "registered_executable_primitive_count": None,
        "registered_coverage_percent": None,
        "unregistered_catalog_entries": None,
        "interpretation": (
            "Catalog structure and uniqueness validated. Runtime implementation "
            "coverage is unavailable because no populated registry factory is "
            "connected to this benchmark. This is not an ARC-AGI-2 accuracy benchmark."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="research/primitive-catalog/original_200.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = benchmark(Path(args.catalog))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + chr(10), encoding="utf-8")


if __name__ == "__main__":
    main()
