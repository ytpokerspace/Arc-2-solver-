import json
import tempfile
import unittest
from pathlib import Path

from arc_r2.reasoning_memory import ReasoningMemory, canonical_fingerprint


class ReasoningMemoryTests(unittest.TestCase):
    def test_round_trip_and_stable_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = ReasoningMemory(Path(directory) / "memory.json")
            record = {
                "id": "visual-boundaries",
                "title": "Visual boundary hypotheses",
                "intent": "Represent alternative object boundaries without losing the source grid",
                "requirements": ["preserve exact dimensions"],
                "assumptions": ["connected components are candidate objects"],
                "unknowns": ["whether diagonal contact is meaningful"],
                "representations": ["pixel grid", "object graph"],
                "hypotheses": ["test 4- and 8-connectivity"],
                "evidence": [],
                "tests": ["compare both segmentations on all training pairs"],
                "tags": ["vision", "segmentation"],
                "sources": ["https://example.org/reference"],
                "status": "proposed",
            }
            saved = memory.put(record)
            loaded = memory.get("visual-boundaries")
            self.assertEqual(saved, loaded)
            self.assertEqual(saved["fingerprint"], canonical_fingerprint({
                key: value for key, value in saved.items()
                if key not in {"updated_at", "fingerprint"}
            }))

    def test_only_validated_records_are_reusable_patterns(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = ReasoningMemory(Path(directory) / "memory.json")
            base = {
                "id": "candidate-a",
                "title": "Candidate rule",
                "intent": "Test a candidate rule",
                "tags": ["reflection"],
                "status": "proposed",
            }
            memory.put(base)
            self.assertEqual(memory.verified_patterns(["reflection"]), [])
            base["status"] = "validated"
            base["outcome"] = "Exact output equality on all recorded validation cases"
            memory.put(base)
            patterns = memory.verified_patterns(["reflection"])
            self.assertEqual([item["id"] for item in patterns], ["candidate-a"])

    def test_rejects_missing_core_fields_and_invalid_status(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = ReasoningMemory(Path(directory) / "memory.json")
            with self.assertRaises(ValueError):
                memory.put({"id": "missing-intent", "title": "Incomplete"})
            with self.assertRaises(ValueError):
                memory.put({
                    "id": "bad-status", "title": "Bad status",
                    "intent": "Reject unsupported state", "status": "guaranteed",
                })

    def test_preserves_multiple_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "memory.json"
            memory = ReasoningMemory(path)
            for record_id in ("one", "two"):
                memory.put({
                    "id": record_id, "title": record_id,
                    "intent": "Keep independent records",
                })
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(data["records"]), {"one", "two"})


if __name__ == "__main__":
    unittest.main()
