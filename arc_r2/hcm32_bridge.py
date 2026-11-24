"""Narrow, evidence-gated bridge between HCM-32 and the ARC primitive solver.

Only grid-geometry primitives with equivalent HCM-32 DSL operators are stored.
HCM-32 retrieval is opt-in and remains separate from SingleStageMemory.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .hcm32 import HCM32Memory, HCM32Runtime
from .primitives import Primitive, Program, as_grid

REGION_ID = "Q02"  # Perception / shape-geometry
_INPUT_CONSTRAINT = "non_empty_rectangular_grid"


@dataclass(frozen=True)
class HCM32Match:
    procedure_id: str
    program: Program


def _encode_program(program: Program) -> list[dict[str, Any]] | None:
    operations = []
    for primitive in program.operations:
        if primitive.name == "FLIP_H" and not primitive.args:
            operator, parameters = "REFLECT", {"axis": "left-right"}
        elif primitive.name == "FLIP_V" and not primitive.args:
            operator, parameters = "REFLECT", {"axis": "top-bottom"}
        elif primitive.name == "ROTATE" and len(primitive.args) == 1:
            turns = primitive.args[0]
            if isinstance(turns, bool) or turns not in (1, 2, 3):
                return None
            operator, parameters = "ROTATE", {"quarter_turns": turns}
        elif primitive.name == "TRANSPOSE" and not primitive.args:
            operator, parameters = "TRANSPOSE", {}
        else:
            return None
        operations.append({"operator": operator, "parameters": parameters})
    return operations or None


def _decode_operations(operations: list[dict[str, Any]]) -> Program | None:
    decoded = []
    for operation in operations:
        name = operation.get("operator")
        parameters = operation.get("parameters", {})
        if name == "REFLECT" and parameters == {"axis": "left-right"}:
            decoded.append(Primitive("FLIP_H"))
        elif name == "REFLECT" and parameters == {"axis": "top-bottom"}:
            decoded.append(Primitive("FLIP_V"))
        elif name == "ROTATE" and set(parameters) == {"quarter_turns"}:
            turns = parameters["quarter_turns"]
            if isinstance(turns, bool) or turns not in (1, 2, 3):
                return None
            decoded.append(Primitive("ROTATE", (turns,)))
        elif name == "TRANSPOSE" and not parameters:
            decoded.append(Primitive("TRANSPOSE"))
        else:
            return None
    return Program(tuple(decoded)) if decoded else None


def _fingerprint(examples) -> str:
    payload = [
        [[list(row) for row in source], [list(row) for row in target]]
        for source, target in examples
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HCM32Bridge:
    """Retrieve only unique all-example exact fits; save only after a CV run."""

    def __init__(self, path):
        self.memory = HCM32Memory(path)
        self.runtime = HCM32Runtime(self.memory)

    def find_program(self, examples) -> HCM32Match | None:
        pairs = tuple((as_grid(source), as_grid(target)) for source, target in examples)
        if not pairs:
            return None
        matches = []
        for node in self.memory.nodes_in_region(REGION_ID):
            for procedure in self.memory.procedures_for_node(node["node_id"]):
                if procedure.get("status") != "verified":
                    continue
                procedure_id = procedure["procedure_id"]
                try:
                    exact = all(
                        self.runtime.apply_procedure(procedure_id, source) == target
                        for source, target in pairs
                    )
                except (KeyError, TypeError, ValueError, IndexError):
                    exact = False
                if exact:
                    program = _decode_operations(procedure["operations"])
                    if program is not None:
                        matches.append(HCM32Match(procedure_id, program))
        return matches[0] if len(matches) == 1 else None

    def store_program(self, program: Program, examples) -> str | None:
        pairs = tuple((as_grid(source), as_grid(target)) for source, target in examples)
        if not pairs or not all(program.apply(source) == target for source, target in pairs):
            return None
        operations = _encode_program(program)
        if operations is None:
            return None
        digest = hashlib.sha256(
            json.dumps(operations, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        procedure_id = "PR-ARC-" + digest
        node_id = "KN-ARC-" + digest
        if procedure_id in self.memory.data["procedures"]:
            return None
        fingerprint = _fingerprint(pairs)
        self.memory.add_procedure(
            procedure_id,
            str(program),
            operations,
            input_constraints=[_INPUT_CONSTRAINT],
        )
        self.memory.add_evidence(
            "EV-ARC-" + digest,
            "procedure",
            procedure_id,
            "exact_fit_on_visible_training_pairs",
            "passed",
            input_fingerprint=fingerprint,
            expected_fingerprint=fingerprint,
            actual_fingerprint=fingerprint,
            details={"training_pair_count": len(pairs), "scope": "fit pairs only"},
        )
        self.memory.set_procedure_status(procedure_id, "verified")
        self.memory.add_node(
            node_id,
            "reusable_grid_transformation",
            str(program),
            [REGION_ID],
            description="ARC primitive procedure; verified only against recorded fit pairs.",
            procedure_refs=[procedure_id],
        )
        self.memory.add_evidence(
            "EVN-ARC-" + digest,
            "node",
            node_id,
            "linked_verified_procedure",
            "passed",
            input_fingerprint=fingerprint,
            expected_fingerprint=fingerprint,
            actual_fingerprint=fingerprint,
            details={"procedure_id": procedure_id},
        )
        self.memory.set_node_status(node_id, "supported")
        self.memory.save()
        return procedure_id
