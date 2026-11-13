import tempfile
import unittest
from pathlib import Path

from arc_r2.glyphic_path_memory import GlyphicPathMemory, make_event


class GlyphicPathMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.memory = GlyphicPathMemory(Path(self.temp.name) / "paths.json")

    def tearDown(self):
        self.temp.cleanup()

    def test_codebook_is_stable_and_sequence_is_readable(self):
        record = self.memory.add_path(
            "p1",
            [make_event("observe", "find object"), make_event("transform", "flip"),
             make_event("verify", "exact fit")],
            tags=["object", "flip"],
            status="verified",
            source_evidence={"input": [[0, 1]], "output": [[1, 0]]},
        )
        self.assertEqual(record["glyph_sequence"], "🔍🔄🎯")
        self.assertEqual(record["events"][0]["glyph"], "🔍")

    def test_retrieval_ranks_matching_tags_and_excludes_rejected_by_default(self):
        common = [make_event("observe", "inspect")]
        self.memory.add_path("a", common, tags=["flip", "geometry"],
                             status="verified", source_evidence={"n": 1})
        self.memory.add_path("b", common, tags=["flip"], status="hypothesis",
                             source_evidence={"n": 2})
        self.memory.add_path("c", common, tags=["fill"], status="rejected",
                             source_evidence={"n": 3})
        found = self.memory.retrieve(["flip", "geometry"])
        self.assertEqual([row["path_id"] for row in found], ["a", "b"])
        self.assertEqual(found[0]["similarity"], 1.0)

    def test_hydration_reconstructs_exact_source_and_checks_hash(self):
        source = {"grid": [[0, 1], [2, 0]], "trace": ["detect", "flip"]}
        self.memory.add_path("exact", [make_event("observe", "grid")],
                             tags=["grid"], status="verified",
                             source_evidence=source)
        self.assertEqual(self.memory.hydrate("exact"), source)

    def test_corrupt_evidence_is_rejected(self):
        self.memory.add_path("bad", [make_event("observe", "grid")],
                             tags=["grid"], status="verified",
                             source_evidence={"grid": [[0, 1]]})
        data = self.memory._read()
        data["paths"]["bad"]["source_evidence"]["grid"][0][0] = 9
        self.memory._write(data)
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.memory.hydrate("bad")

    def test_duplicate_path_ids_are_rejected(self):
        args = dict(events=[make_event("observe", "grid")], tags=["grid"],
                    status="verified", source_evidence={"x": 1})
        self.memory.add_path("same", **args)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.memory.add_path("same", **args)

    def test_hybridization_deduplicates_overlap_and_remains_hypothesis(self):
        self.memory.add_path(
            "left", [make_event("observe", "detect"), make_event("object", "object")],
            tags=["detect"], status="verified", source_evidence={"left": 1})
        self.memory.add_path(
            "right", [make_event("object", "object"), make_event("transform", "move"),
                      make_event("verify", "check")],
            tags=["move"], status="verified", source_evidence={"right": 1})
        result = self.memory.hybridize("left", "right")
        self.assertEqual(result["status"], "hypothesis")
        self.assertEqual(result["overlap_events"], 1)
        self.assertEqual(result["glyph_sequence"], "🔍🟦🔄🎯")

    def test_hypothesis_paths_cannot_be_treated_as_verified_hybrid_sources(self):
        event = [make_event("observe", "detect")]
        self.memory.add_path("hyp", event, tags=["x"], status="hypothesis",
                             source_evidence={"x": 1})
        self.memory.add_path("verified", event, tags=["x"], status="verified",
                             source_evidence={"x": 2})
        with self.assertRaisesRegex(ValueError, "only verified"):
            self.memory.hybridize("hyp", "verified")


if __name__ == "__main__":
    unittest.main()
