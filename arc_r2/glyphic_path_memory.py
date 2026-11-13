"""Glyphic semantic paths for reusable ARC-R2 reasoning memory.

Glyphs are compact semantic indexes, never substitutes for source evidence.
Only paths explicitly marked verified may be retrieved as verified knowledge.
Hybrid paths are always hypotheses until independently checked.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CODEBOOK_VERSION = 1
CODEBOOK = {
    "observe": "🔍",
    "object": "🟦",
    "relation": "🔗",
    "geometry": "📐",
    "transform": "🔄",
    "verify": "🎯",
    "hypothesis": "💭",
    "reject": "🚫",
    "memory": "🧠",
}
VALID_STATUSES = {"verified", "hypothesis", "rejected"}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GlyphEvent:
    kind: str
    label: str
    details: dict[str, Any]

    @property
    def glyph(self) -> str:
        return CODEBOOK.get(self.kind, "❔")

    def to_record(self) -> dict[str, Any]:
        return {"kind": self.kind, "glyph": self.glyph,
                "label": self.label, "details": self.details}


def make_event(kind: str, label: str, **details: Any) -> GlyphEvent:
    """Create a typed event; unknown kinds remain visible as ❔, not discarded."""
    return GlyphEvent(kind=kind, label=label, details=dict(details))


class GlyphicPathMemory:
    """Append/update semantic paths with evidence hashes and lazy hydration."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "codebook_version": CODEBOOK_VERSION, "paths": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("paths"), dict):
            raise ValueError("invalid glyphic memory document")
        if data.get("version") != 1:
            raise ValueError("unsupported glyphic memory version")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(_canonical(data) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def add_path(
        self,
        path_id: str,
        events: Iterable[GlyphEvent],
        *,
        tags: Iterable[str],
        status: str,
        source_evidence: Any,
    ) -> dict[str, Any]:
        """Persist one path and its original evidence; reject duplicate IDs."""
        if not path_id or not isinstance(path_id, str):
            raise ValueError("path_id must be a non-empty string")
        if status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        event_records = [event.to_record() for event in events]
        if not event_records:
            raise ValueError("a semantic path must contain at least one event")
        if not isinstance(source_evidence, (dict, list, str, int, float, bool, type(None))):
            raise TypeError("source_evidence must be JSON-serializable data")
        # Validate serializability before mutating persistent state.
        evidence_copy = json.loads(_canonical(source_evidence))
        data = self._read()
        if path_id in data["paths"]:
            raise ValueError(f"duplicate path_id: {path_id}")
        record = {
            "path_id": path_id,
            "status": status,
            "tags": sorted({str(tag) for tag in tags if str(tag)}),
            "events": event_records,
            "glyph_sequence": "".join(item["glyph"] for item in event_records),
            "source_sha256": _sha256(evidence_copy),
            "source_evidence": evidence_copy,
        }
        data["paths"][path_id] = record
        self._write(data)
        return {key: value for key, value in record.items() if key != "source_evidence"}

    def retrieve(
        self,
        query_tags: Iterable[str],
        *,
        limit: int = 10,
        statuses: Iterable[str] = ("verified", "hypothesis"),
    ) -> list[dict[str, Any]]:
        """Rank paths by tag Jaccard similarity; return metadata, not raw evidence."""
        if limit < 1:
            raise ValueError("limit must be positive")
        query = {str(tag) for tag in query_tags if str(tag)}
        allowed = set(statuses)
        data = self._read()
        ranked = []
        for record in data["paths"].values():
            if record["status"] not in allowed:
                continue
            tags = set(record.get("tags", []))
            union = query | tags
            score = len(query & tags) / len(union) if union else 0.0
            if query and not (query & tags):
                continue
            ranked.append((score, record["path_id"], record))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        results = []
        for score, _, record in ranked[:limit]:
            results.append({
                "path_id": record["path_id"],
                "status": record["status"],
                "tags": list(record["tags"]),
                "glyph_sequence": record["glyph_sequence"],
                "events": list(record["events"]),
                "similarity": score,
                "source_sha256": record["source_sha256"],
            })
        return results

    def hydrate(self, path_id: str) -> Any:
        """Return original evidence only after verifying its stored SHA-256."""
        data = self._read()
        record = data["paths"].get(path_id)
        if record is None:
            raise KeyError(path_id)
        evidence = record.get("source_evidence")
        if _sha256(evidence) != record.get("source_sha256"):
            raise ValueError(f"evidence integrity check failed for {path_id}")
        return evidence

    def hybridize(self, first_id: str, second_id: str) -> dict[str, Any]:
        """Compose two verified paths with maximal suffix/prefix overlap.

        The result is a candidate hypothesis, never an automatically verified rule.
        """
        data = self._read()
        first = data["paths"].get(first_id)
        second = data["paths"].get(second_id)
        if first is None or second is None:
            raise KeyError(first_id if first is None else second_id)
        if first["status"] != "verified" or second["status"] != "verified":
            raise ValueError("only verified source paths may be hybridized")
        left, right = first["events"], second["events"]
        overlap = 0
        for size in range(1, min(len(left), len(right)) + 1):
            if left[-size:] == right[:size]:
                overlap = size
        return {
            "status": "hypothesis",
            "source_paths": [first_id, second_id],
            "overlap_events": overlap,
            "events": left + right[overlap:],
            "glyph_sequence": "".join(item["glyph"] for item in left + right[overlap:]),
        }
