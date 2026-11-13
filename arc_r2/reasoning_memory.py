"""Structured interpretation records for ARC-R2.

This module stores traceable engineering context; it does not train a model or
claim that stored hypotheses are correct. Only validated records can be
retrieved as reusable verified patterns.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

STATUSES = {"proposed", "implemented", "tested", "measured", "validated"}
LIST_FIELDS = (
    "requirements", "assumptions", "unknowns", "representations",
    "hypotheses", "evidence", "tests", "tags", "sources",
)


def canonical_fingerprint(value: Any) -> str:
    """Return a stable SHA-256 fingerprint for JSON-compatible input."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary")
    record_id = record.get("id")
    title = record.get("title")
    intent = record.get("intent")
    if not all(isinstance(value, str) and value.strip()
               for value in (record_id, title, intent)):
        raise ValueError("record requires non-empty string fields: id, title, intent")

    status = record.get("status", "proposed")
    if status not in STATUSES:
        raise ValueError(f"status must be one of: {', '.join(sorted(STATUSES))}")

    clean = {
        "id": record_id.strip(),
        "title": title.strip(),
        "intent": intent.strip(),
        "status": status,
    }
    for field in LIST_FIELDS:
        value = record.get(field, [])
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list")
        clean[field] = value
    clean["outcome"] = record.get("outcome")
    clean["updated_at"] = record.get("updated_at")
    clean["fingerprint"] = canonical_fingerprint({
        key: value for key, value in clean.items()
        if key not in {"updated_at", "fingerprint"}
    })
    return clean


class ReasoningMemory:
    """Small, dependency-free, provenance-aware JSON memory store."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "records": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read reasoning memory: {exc}") from exc
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("unsupported or invalid reasoning-memory schema")
        if not isinstance(data.get("records"), dict):
            raise ValueError("reasoning-memory records must be an object")
        return data

    def put(self, record: dict[str, Any]) -> dict[str, Any]:
        """Insert or replace a record by ID; returns its normalized copy."""
        clean = _normalise_record(record)
        data = self._read()
        data["records"][clean["id"]] = clean
        self._write(data)
        return clean.copy()

    def get(self, record_id: str) -> dict[str, Any] | None:
        record = self._read()["records"].get(record_id)
        return record.copy() if isinstance(record, dict) else None

    def find(self, tags: Iterable[str] = (), *, status: str | None = None) -> list[dict[str, Any]]:
        """Find records containing every requested tag, optionally by status."""
        wanted = set(tags)
        if status is not None and status not in STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(STATUSES))}")
        found = []
        for record in self._read()["records"].values():
            if not isinstance(record, dict):
                continue
            if wanted.issubset(set(record.get("tags", []))):
                if status is None or record.get("status") == status:
                    found.append(record.copy())
        return sorted(found, key=lambda item: item["id"])

    def verified_patterns(self, tags: Iterable[str] = ()) -> list[dict[str, Any]]:
        """Return only validated records; proposed ideas never become memory facts."""
        return self.find(tags, status="validated")

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
