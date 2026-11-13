"""Small, offline index connecting research concepts to existing ARC-R2 modules.

This is a retrieval/catalog layer, not an autonomous learner. It stores concise
metadata and mappings; it does not reproduce copyrighted book content or claim
that linked books have been fully read.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeEntry:
    key: str
    discipline: str
    concept: str
    arc_use: str
    existing_modules: tuple[str, ...]
    source_title: str
    source_url: str
    evidence_status: str = "bibliographic/source metadata verified; concept mapping is an engineering hypothesis"


_ENTRIES = (
    KnowledgeEntry(
        "symbolic-ai-programs", "program synthesis",
        "Symbolic representations, rule-based programs, search, unification, and constraint-based scene labeling.",
        "Inform the typed DSL, candidate program generation, and explicit rule representations.",
        ("arc_r2/dsl.py", "arc_r2/search.py", "arc_r2/primitives.py"),
        "Paradigms of Artificial Intelligence Programming: Case Studies in Common Lisp — Peter Norvig",
        "https://www.norvig.com/paip.html",
    ),
    KnowledgeEntry(
        "constraint-processing", "constraint satisfaction",
        "Constraint networks, consistency enforcement, look-ahead/look-back search, and hybrid inference.",
        "Treat each training pair as a constraint on candidate transformations; reject any program that fails an example.",
        ("arc_r2/primitive_benchmark.py", "arc_r2/primitives.py", "arc_r2/object_solver.py"),
        "Constraint Processing — Rina Dechter",
        "https://ics.uci.edu/~dechter/books/",
    ),
    KnowledgeEntry(
        "csp-foundations", "constraint satisfaction",
        "Formal CSP definitions, domains, constraints, problem reduction, and propagation.",
        "A future typed constraint layer could represent color, object, geometry, and relation requirements.",
        ("arc_r2/objects.py", "arc_r2/relations.py", "arc_r2/object_solver.py"),
        "Foundations of Constraint Satisfaction — Edward Tsang",
        "https://www.bracil.net/edward/FCS.html",
    ),
    KnowledgeEntry(
        "mathematical-morphology", "computer vision",
        "Shape and region analysis using operations such as dilation, erosion, and morphological filtering.",
        "Candidate future operators for boundary, connectivity, hole, and region transformations; not currently implemented by this catalog.",
        ("arc_r2/objects.py", "arc_r2/object_ops.py", "arc_r2/primitives.py"),
        "Morphological Image Analysis: Principles and Applications — Pierre Soille",
        "https://link.springer.com/book/10.1007/978-3-662-05088-0",
    ),
    KnowledgeEntry(
        "graph-structures", "graph theory",
        "Graph connectivity, matching, coloring, flows, and combinatorial structure.",
        "Extend object relation graphs toward graph matching and structural correspondence between examples.",
        ("arc_r2/objects.py", "arc_r2/relations.py", "arc_r2/object_solver.py"),
        "Modern Graph Theory — Béla Bollobás",
        "https://link.springer.com/book/10.1007/978-1-4612-0619-4",
    ),
    KnowledgeEntry(
        "combinatorial-csp", "combinatorial algorithms",
        "Backtracking, satisfiability, constraint satisfaction, graph labeling, and variable-ordering heuristics.",
        "Guide bounded candidate enumeration and future constraint pruning; keep the active primitive benchmark deterministic.",
        ("arc_r2/primitive_benchmark.py", "arc_r2/primitives.py", "arc_r2/search.py"),
        "The Art of Computer Programming, Volume 4, Fascicle 7: Constraint Satisfaction — Donald E. Knuth",
        "https://www.oreilly.com/library/view/art-of-computer/9780135328262/",
    ),
    KnowledgeEntry(
        "vector-symbolic-arc", "vector symbolic architectures",
        "Object-centric program synthesis with vector symbolic algebra representations.",
        "Research candidate for compact object/relation encoding; evaluate separately before integrating.",
        ("arc_r2/objects.py", "arc_r2/relations.py", "arc_r2/object_dsl.py"),
        "Vector Symbolic Algebras for the Abstraction and Reasoning Corpus — Isaac Joffe and Chris Eliasmith",
        "https://arxiv.org/abs/2511.08747",
    ),
)


def iter_entries() -> tuple[KnowledgeEntry, ...]:
    """Return the immutable catalog in stable order."""
    return _ENTRIES


def get_entry(key: str) -> KnowledgeEntry:
    """Return one catalog entry by key; raise KeyError when absent."""
    for entry in _ENTRIES:
        if entry.key == key:
            return entry
    raise KeyError(key)


def search_knowledge(query: str) -> tuple[KnowledgeEntry, ...]:
    """Case-insensitive search over discipline, concept, use, and source title."""
    terms = tuple(term for term in query.casefold().split() if term)
    if not terms:
        return ()
    matches = []
    for entry in _ENTRIES:
        haystack = " ".join((
            entry.discipline, entry.concept, entry.arc_use, entry.source_title,
            entry.key, " ".join(entry.existing_modules),
        )).casefold()
        if all(term in haystack for term in terms):
            matches.append(entry)
    return tuple(matches)
