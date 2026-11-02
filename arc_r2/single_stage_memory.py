"""Persistent single-stage memory for exact-fit ARC programs.

A record is scoped to the exact ordered training-pair set. Cached programs are
revalidated before reuse; stale records never block a fresh search.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuples(item) for key, item in value.items()}
    return value


class SingleStageMemory:
    """Store one confirmed exact-fit program per training-problem fingerprint."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self, fingerprint: str) -> tuple[tuple[str, tuple], ...] | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            record = data.get("records", {}).get(fingerprint)
            if not isinstance(record, dict) or record.get("stage") != 1:
                return None
            operations = record.get("operations")
            if not isinstance(operations, list):
                return None
            decoded = []
            for operation in operations:
                if not isinstance(operation, dict) or not isinstance(operation.get("name"), str):
                    return None
                args = _tuples(operation.get("args", []))
                if not isinstance(args, tuple):
                    return None
                decoded.append((operation["name"], args))
            return tuple(decoded)
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def save(self, fingerprint: str, operations: tuple) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "records": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("records"), dict):
                    data = loaded
            except (OSError, ValueError):
                pass
        encoded = []
        for name, args in operations:
            encoded.append({"name": name, "args": _json_value(args)})
        data["records"][fingerprint] = {
            "stage": 1,
            "operations": encoded,
            "validation": "exact_fit_on_all_training_pairs",
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value
