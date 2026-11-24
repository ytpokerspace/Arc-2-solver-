"""HCM-32: hierarchical, cross-linked, evidence-aware procedural memory.

The 32 stable regions organize knowledge; the sparse connection graph links
nodes and regions across any hierarchy level. Saved DSL records are declarative
data only. Runtime code must be registered separately by a trusted caller.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
RELATION_TYPES = frozenset({
    "similar", "composable", "inverse", "dependency", "equivalent",
    "contradictory", "generalizes", "derived", "sequential",
})
NODE_STATUSES = frozenset({"hypothesis", "supported", "verified", "rejected"})
CONNECTION_STATUSES = frozenset({"hypothesis", "supported", "verified", "rejected"})
PROCEDURE_STATUSES = frozenset({"draft", "tested", "verified", "deprecated"})
EVIDENCE_RESULTS = frozenset({"passed", "failed", "inconclusive"})
CANDIDATE_STATUSES = frozenset({"generated", "untested", "validated", "rejected"})
SUBREGION_ROLES = (
    ("S01", "identify"),
    ("S02", "represent"),
    ("S03", "evaluate_connect"),
    ("S04", "retain_retrieve"),
)

# Eight families x four stable primary regions = 32 regions.
_FAMILIES = (
    ("F01", "perception", (
        "object_identity", "shape_geometry", "color_attributes", "position_orientation")),
    ("F02", "structure", (
        "spatial_relations", "symmetry_patterns", "topology_connectivity", "constraints_boundaries")),
    ("F03", "combinatorial", (
        "pairwise_combinations", "multi_rule_combinations", "permutation_sequencing", "hybridization_composition")),
    ("F04", "time", (
        "operation_order", "temporal_progression", "learning_history_recency", "duration_cycles_repetition")),
    ("F05", "methodology", (
        "problem_decomposition", "hypothesis_formation", "experimental_design", "search_decision_strategies")),
    ("F06", "technique", (
        "dsl_primitives", "transformation_operators", "inverse_operations", "program_composition")),
    ("F07", "relationships", (
        "similarity_distance", "causal_functional_relations", "generalization_transfer", "alternatives_contradictions")),
    ("F08", "verification_memory", (
        "evidence_confidence", "counterexamples_failures", "reusable_procedures", "recall_refinement")),
)


def _build_regions() -> dict[str, dict[str, Any]]:
    regions: dict[str, dict[str, Any]] = {}
    index = 0
    for family_id, family_name, labels in _FAMILIES:
        for label in labels:
            index += 1
            qid = f"Q{index:02d}"
            regions[qid] = {
                "region_id": qid,
                "region_type": "quadrant",
                "family_id": family_id,
                "family": family_name,
                "label": label,
            }
            for suffix, role in SUBREGION_ROLES:
                sid = f"{qid}.{suffix}"
                regions[sid] = {
                    "region_id": sid,
                    "region_type": "subquadrant",
                    "parent_id": qid,
                    "family_id": family_id,
                    "family": family_name,
                    "label": label,
                    "role": role,
                }
    return regions


REGIONS = _build_regions()
PRIMARY_REGION_IDS = tuple(f"Q{i:02d}" for i in range(1, 33))
SUBREGION_IDS = tuple(f"Q{i:02d}.S{j:02d}" for i in range(1, 33) for j in range(1, 5))


class HCM32FormatError(ValueError):
    """Raised when persisted HCM-32 data is malformed or incompatible."""


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _json_copy(value: Any, field: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain JSON-safe values") from exc


def _empty_data() -> dict[str, Any]:
    return {
        "schema": "hcm32",
        "schema_version": SCHEMA_VERSION,
        "regions": REGIONS,
        "nodes": {},
        "connections": {},
        "procedures": {},
        "evidence": {},
        "candidates": {},
        "events": [],
    }


def _validate_endpoint(endpoint: Any) -> dict[str, str]:
    endpoint = _require_mapping(endpoint, "connection endpoint")
    kind = endpoint.get("kind")
    identifier = _require_text(endpoint.get("id"), "endpoint.id")
    if kind == "node":
        if not identifier.startswith("KN-"):
            raise ValueError("node endpoint IDs must start with KN-")
    elif kind == "region":
        if identifier not in REGIONS:
            raise ValueError(f"unknown region endpoint: {identifier}")
    else:
        raise ValueError("endpoint.kind must be 'node' or 'region'")
    return {"kind": kind, "id": identifier}


class HCM32Memory:
    """Persistent sparse memory with strict records and atomic JSON writes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = _empty_data()
        if self.path.exists():
            self.data = self._read_and_validate()

    @property
    def regions(self) -> Mapping[str, Mapping[str, Any]]:
        return self.data["regions"]

    def _read_and_validate(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise HCM32FormatError("memory root must be an object")
            if raw.get("schema") != "hcm32" or raw.get("schema_version") != SCHEMA_VERSION:
                raise HCM32FormatError("unsupported HCM-32 schema/version")
            if raw.get("regions") != REGIONS:
                raise HCM32FormatError("region catalog does not match this schema version")
            for key in ("nodes", "connections", "procedures", "evidence", "candidates"):
                if not isinstance(raw.get(key), dict):
                    raise HCM32FormatError(f"{key} must be an object")
            if not isinstance(raw.get("events"), list):
                raise HCM32FormatError("events must be a list")
            # Validate record identity and critical references before trusting loaded data.
            for key, record in raw["nodes"].items():
                if not isinstance(record, dict) or record.get("node_id") != key:
                    raise HCM32FormatError(f"invalid node record: {key}")
                if not isinstance(record.get("quadrant_refs"), list) or any(
                    ref not in REGIONS for ref in record["quadrant_refs"]
                ):
                    raise HCM32FormatError(f"invalid region references in node: {key}")
            for key, record in raw["connections"].items():
                if not isinstance(record, dict) or record.get("connection_id") != key:
                    raise HCM32FormatError(f"invalid connection record: {key}")
                _validate_endpoint(record.get("source"))
                _validate_endpoint(record.get("target"))
                if record["source"]["kind"] == "node" and record["source"]["id"] not in raw["nodes"]:
                    raise HCM32FormatError(f"missing source node in connection: {key}")
                if record["target"]["kind"] == "node" and record["target"]["id"] not in raw["nodes"]:
                    raise HCM32FormatError(f"missing target node in connection: {key}")
            for key, record in raw["procedures"].items():
                if not isinstance(record, dict) or record.get("procedure_id") != key:
                    raise HCM32FormatError(f"invalid procedure record: {key}")
                self._validate_procedure_record(record, raw["procedures"], allow_missing_inverse=True)
            for key, record in raw["evidence"].items():
                if not isinstance(record, dict) or record.get("evidence_id") != key:
                    raise HCM32FormatError(f"invalid evidence record: {key}")
            for key, record in raw["candidates"].items():
                if not isinstance(record, dict) or record.get("candidate_id") != key:
                    raise HCM32FormatError(f"invalid candidate record: {key}")
            return raw
        except HCM32FormatError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            raise HCM32FormatError(f"cannot safely load HCM-32 memory: {exc}") from exc

    @staticmethod
    def _validate_procedure_record(
        record: Mapping[str, Any],
        procedures: Mapping[str, Any],
        allow_missing_inverse: bool = False,
    ) -> None:
        if not isinstance(record.get("operations"), list) or not record["operations"]:
            raise ValueError("procedure operations must be a non-empty list")
        for operation in record["operations"]:
            operation = _require_mapping(operation, "operation")
            _require_text(operation.get("operator"), "operation.operator")
            _require_mapping(operation.get("parameters", {}), "operation.parameters")
        inverse = record.get("inverse_procedure_id")
        if inverse is not None and not isinstance(inverse, str):
            raise ValueError("inverse_procedure_id must be a string or null")
        if inverse and inverse != record.get("procedure_id") and not allow_missing_inverse:
            if inverse not in procedures:
                raise ValueError(f"unknown inverse procedure: {inverse}")
        if record.get("status") not in PROCEDURE_STATUSES:
            raise ValueError("invalid procedure status")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.data, sort_keys=True, indent=2, allow_nan=False) + "\n"
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def add_node(
        self, node_id: str, node_type: str, name: str, quadrant_refs: list[str],
        description: str = "", status: str = "hypothesis",
        procedure_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        _require_text(node_id, "node_id")
        if not node_id.startswith("KN-"):
            raise ValueError("node_id must start with KN-")
        if node_id in self.data["nodes"]:
            raise ValueError(f"duplicate node_id: {node_id}")
        if status != "hypothesis":
            raise ValueError("new nodes start as hypotheses; use set_node_status after recording evidence")
        if not isinstance(quadrant_refs, list) or not quadrant_refs:
            raise ValueError("quadrant_refs must be a non-empty list")
        if any(ref not in REGIONS for ref in quadrant_refs):
            raise ValueError("quadrant_refs contains an unknown region")
        record = {
            "node_id": node_id, "node_type": _require_text(node_type, "node_type"),
            "name": _require_text(name, "name"), "description": description,
            "quadrant_refs": list(dict.fromkeys(quadrant_refs)),
            "procedure_refs": list(procedure_refs or []), "evidence_refs": [], "status": status, "version": 1,
        }
        self.data["nodes"][node_id] = record
        return record

    def add_connection(
        self, connection_id: str, source: Mapping[str, str], target: Mapping[str, str],
        relationship_type: str, *, directed: bool = True, strength: float | None = None,
        strength_basis: str = "unmeasured", status: str = "hypothesis",
    ) -> dict[str, Any]:
        _require_text(connection_id, "connection_id")
        if connection_id in self.data["connections"]:
            raise ValueError(f"duplicate connection_id: {connection_id}")
        if relationship_type not in RELATION_TYPES:
            raise ValueError("unsupported relationship_type")
        if status != "hypothesis":
            raise ValueError("new connections start as hypotheses; use set_connection_status after evidence")
        src, dst = _validate_endpoint(source), _validate_endpoint(target)
        for endpoint in (src, dst):
            if endpoint["kind"] == "node" and endpoint["id"] not in self.data["nodes"]:
                raise ValueError(f"unknown node endpoint: {endpoint['id']}")
        if src == dst:
            raise ValueError("self-connections are not allowed")
        if strength is not None and (isinstance(strength, bool) or not isinstance(strength, (int, float)) or not 0 <= strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        record = {
            "connection_id": connection_id, "source": src, "target": dst,
            "relationship_type": relationship_type, "directed": bool(directed),
            "strength": strength, "strength_basis": _require_text(strength_basis, "strength_basis"),
            "status": status, "successful_uses": 0, "failed_uses": 0, "evidence_refs": [],
        }
        self.data["connections"][connection_id] = record
        return record

    def add_procedure(
        self, procedure_id: str, name: str, operations: list[Mapping[str, Any]],
        *, dsl_version: str = "1", input_constraints: list[str] | None = None,
        inverse_procedure_id: str | None = None, status: str = "draft",
        evidence_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        _require_text(procedure_id, "procedure_id")
        if procedure_id in self.data["procedures"]:
            raise ValueError(f"duplicate procedure_id: {procedure_id}")
        record = {
            "procedure_id": procedure_id, "name": _require_text(name, "name"),
            "dsl_version": _require_text(dsl_version, "dsl_version"),
            "operations": _json_copy(list(operations), "operations"),
            "input_constraints": list(input_constraints or []),
            "inverse_procedure_id": inverse_procedure_id, "status": status,
            "evidence_refs": list(evidence_refs or []), "reuse_successes": 0,
            "reuse_failures": 0, "version": 1,
        }
        if status != "draft":
            raise ValueError("new procedures start as drafts; use set_procedure_status after evidence")
        self._validate_procedure_record(record, self.data["procedures"])
        self.data["procedures"][procedure_id] = record
        return record

    def add_evidence(
        self, evidence_id: str, subject_type: str, subject_id: str,
        evidence_type: str, result: str, *, input_fingerprint: str | None = None,
        expected_fingerprint: str | None = None, actual_fingerprint: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        _require_text(evidence_id, "evidence_id")
        if evidence_id in self.data["evidence"]:
            raise ValueError(f"duplicate evidence_id: {evidence_id}")
        if subject_type not in {"node", "procedure", "candidate", "connection"}:
            raise ValueError("invalid evidence subject_type")
        collection = {"node": "nodes", "procedure": "procedures",
                      "candidate": "candidates", "connection": "connections"}[subject_type]
        if subject_id not in self.data[collection]:
            raise ValueError(f"unknown evidence subject: {subject_id}")
        if result not in EVIDENCE_RESULTS:
            raise ValueError("invalid evidence result")
        record = {
            "evidence_id": evidence_id, "subject_type": subject_type,
            "subject_id": subject_id, "evidence_type": _require_text(evidence_type, "evidence_type"),
            "result": result, "input_fingerprint": input_fingerprint,
            "expected_fingerprint": expected_fingerprint,
            "actual_fingerprint": actual_fingerprint,
            "details": _json_copy(dict(details or {}), "details"),
        }
        self.data["evidence"][evidence_id] = record
        subject = self.data[collection][subject_id]
        subject.setdefault("evidence_refs", []).append(evidence_id)
        return record

    @staticmethod
    def _require_evidence_result(
        record: Mapping[str, Any], evidence: Mapping[str, Any], result: str
    ) -> None:
        refs = record.get("evidence_refs", [])
        if not any(ref in evidence and evidence[ref].get("result") == result for ref in refs):
            raise ValueError(f"a {result} evidence record is required for this status")

    def set_node_status(self, node_id: str, status: str) -> None:
        if node_id not in self.data["nodes"] or status not in NODE_STATUSES:
            raise ValueError("unknown node or invalid status")
        if status in {"supported", "verified"}:
            self._require_evidence_result(self.data["nodes"][node_id], self.data["evidence"], "passed")
        elif status == "rejected":
            self._require_evidence_result(self.data["nodes"][node_id], self.data["evidence"], "failed")
        self.data["nodes"][node_id]["status"] = status

    def set_connection_status(self, connection_id: str, status: str) -> None:
        if connection_id not in self.data["connections"] or status not in CONNECTION_STATUSES:
            raise ValueError("unknown connection or invalid status")
        if status in {"supported", "verified"}:
            self._require_evidence_result(self.data["connections"][connection_id], self.data["evidence"], "passed")
        elif status == "rejected":
            self._require_evidence_result(self.data["connections"][connection_id], self.data["evidence"], "failed")
        self.data["connections"][connection_id]["status"] = status

    def set_procedure_status(self, procedure_id: str, status: str) -> None:
        if procedure_id not in self.data["procedures"] or status not in PROCEDURE_STATUSES:
            raise ValueError("unknown procedure or invalid status")
        if status in {"tested", "verified"}:
            self._require_evidence_result(self.data["procedures"][procedure_id], self.data["evidence"], "passed")
        elif status == "deprecated":
            self._require_evidence_result(self.data["procedures"][procedure_id], self.data["evidence"], "failed")
        self.data["procedures"][procedure_id]["status"] = status

    def set_candidate_status(self, candidate_id: str, status: str) -> None:
        if candidate_id not in self.data["candidates"] or status not in CANDIDATE_STATUSES:
            raise ValueError("unknown candidate or invalid status")
        if status == "validated":
            self._require_evidence_result(self.data["candidates"][candidate_id], self.data["evidence"], "passed")
        elif status == "rejected":
            self._require_evidence_result(self.data["candidates"][candidate_id], self.data["evidence"], "failed")
        self.data["candidates"][candidate_id]["status"] = status

    def add_candidate(
        self, candidate_id: str, input_procedures: list[str],
        operation_sequence: list[str], *, generation_method: str,
        status: str = "generated",
    ) -> dict[str, Any]:
        _require_text(candidate_id, "candidate_id")
        if candidate_id in self.data["candidates"]:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        if not input_procedures or any(pid not in self.data["procedures"] for pid in input_procedures):
            raise ValueError("candidate must reference existing procedures")
        if not operation_sequence or any(pid not in input_procedures for pid in operation_sequence):
            raise ValueError("operation_sequence must be a non-empty ordered subset of input_procedures")
        if status != "generated":
            raise ValueError("new candidates start as generated; use set_candidate_status after evidence")
        record = {
            "candidate_id": candidate_id, "input_procedures": list(input_procedures),
            "operation_sequence": list(operation_sequence),
            "generation_method": _require_text(generation_method, "generation_method"),
            "status": status, "evidence_refs": [],
        }
        self.data["candidates"][candidate_id] = record
        return record

    def record_application(self, procedure_id: str, result: str, *, task_ref: str = "") -> dict[str, Any]:
        if procedure_id not in self.data["procedures"]:
            raise ValueError(f"unknown procedure: {procedure_id}")
        if result not in {"passed", "failed", "inconclusive"}:
            raise ValueError("application result must be passed, failed, or inconclusive")
        procedure = self.data["procedures"][procedure_id]
        if result == "passed":
            procedure["reuse_successes"] += 1
        elif result == "failed":
            procedure["reuse_failures"] += 1
        event = {
            "event_number": len(self.data["events"]) + 1,
            "event_type": "procedure_application", "subject_id": procedure_id,
            "result": result, "task_ref": task_ref,
        }
        self.data["events"].append(event)
        return event

    def connections_for(self, endpoint: Mapping[str, str]) -> list[dict[str, Any]]:
        wanted = _validate_endpoint(endpoint)
        return [edge for edge in self.data["connections"].values()
                if edge["source"] == wanted or edge["target"] == wanted
                or (not edge["directed"] and (edge["source"] == wanted or edge["target"] == wanted))]

    def procedures_for_node(self, node_id: str) -> list[dict[str, Any]]:
        node = self.data["nodes"].get(node_id)
        if node is None:
            raise KeyError(node_id)
        return [self.data["procedures"][pid] for pid in node["procedure_refs"]
                if pid in self.data["procedures"]]

    def snapshot(self) -> dict[str, Any]:
        return _json_copy(self.data, "memory snapshot")
