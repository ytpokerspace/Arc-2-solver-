"""Canonical emoji aliases for the ARC-2 engine's human-readable trace.

Symbols are documentation/telemetry labels, not executable instructions.
Canonical IDs remain the stable machine-readable identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineSymbol:
    symbol: str
    canonical_id: str
    description: str


SYMBOLS = (
    EngineSymbol("📥", "ARC2.INPUT", "Load task and visible training pairs"),
    EngineSymbol("🔐", "ARC2.PRESERVE", "Preserve source grids and provenance"),
    EngineSymbol("👁️", "ARC2.PERCEPTION", "Extract colors, objects, and geometry"),
    EngineSymbol("🕸️", "ARC2.SCENE_GRAPH", "Represent spatial relationships"),
    EngineSymbol("⚙️", "ARC2.DSL", "Represent typed transformation programs"),
    EngineSymbol("🧩", "ARC2.PRIMITIVE_ENUMERATION", "Enumerate elementary program compositions"),
    EngineSymbol("🌳", "ARC2.PREDICTIVE_MCTS", "Explore bounded hybrid programs"),
    EngineSymbol("💾", "ARC2.MEMORY", "Reuse fingerprint-scoped programs after revalidation"),
    EngineSymbol("✅", "ARC2.EXACT_FIT", "Require exact equality on every visible training pair"),
    EngineSymbol("⚖️", "ARC2.SELECT", "Choose the simplest exact-fit candidate found"),
    EngineSymbol("🛑", "ARC2.ABSTAIN", "Decline to predict when no exact-fit program is found"),
    EngineSymbol("📤", "ARC2.TEST_PREDICTION", "Apply selected program to test input grids"),
    EngineSymbol("📊", "ARC2.TELEMETRY", "Record counts, runtime, and validation evidence"),
)

BY_CANONICAL_ID = {item.canonical_id: item for item in SYMBOLS}


def render_trace(canonical_ids: tuple[str, ...] | list[str]) -> str:
    """Render known canonical stage IDs as a compact emoji trace."""
    unknown = [item for item in canonical_ids if item not in BY_CANONICAL_ID]
    if unknown:
        raise ValueError(f"unknown ARC-2 engine stage IDs: {', '.join(unknown)}")
    return " → ".join(BY_CANONICAL_ID[item].symbol for item in canonical_ids)
