import json
import tempfile
import unittest
from pathlib import Path

from arc_r2.hcm32 import (
    HCM32FormatError,
    HCM32Memory,
    PRIMARY_REGION_IDS,
    REGIONS,
    SUBREGION_IDS,
)


class HCM32MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "hcm32.json"
        self.memory = HCM32Memory(self.path)

    def _reflection_procedure(self):
        return self.memory.add_procedure(
            "PR-REFLECT-H",
            "horizontal_reflection",
            [{"operator": "REFLECT", "parameters": {"axis": "left-right"}}],
            inverse_procedure_id="PR-REFLECT-H",
            status="draft",
        )

    def test_catalog_has_32_primary_and_128_subregions(self):
        self.assertEqual(len(PRIMARY_REGION_IDS), 32)
        self.assertEqual(len(SUBREGION_IDS), 128)
        self.assertEqual(len(REGIONS), 160)
        for region_id in PRIMARY_REGION_IDS:
            self.assertEqual(
                sum(item["parent_id"] == region_id for item in REGIONS.values()
                    if item["region_type"] == "subquadrant"),
                4,
            )

    def test_cross_family_and_cross_level_connections(self):
        self.memory.add_node(
            "KN-REFLECTION", "transformation_rule", "horizontal reflection",
            ["Q22.S02"], status="hypothesis",
        )
        self.memory.add_node(
            "KN-SYMMETRY", "visual_concept", "bilateral symmetry",
            ["Q06.S01"], status="hypothesis",
        )
        edge = self.memory.add_connection(
            "CN-1",
            {"kind": "node", "id": "KN-REFLECTION"},
            {"kind": "node", "id": "KN-SYMMETRY"},
            "similar",
            strength=0.9,
            strength_basis="measured_on_fixture_set",
        )
        self.assertEqual(edge["status"], "hypothesis")
        region_edge = self.memory.add_connection(
            "CN-2",
            {"kind": "region", "id": "Q22.S02"},
            {"kind": "region", "id": "Q06"},
            "dependency",
        )
        self.assertEqual(region_edge["target"]["id"], "Q06")
        node_region_edge = self.memory.add_connection(
            "CN-3",
            {"kind": "node", "id": "KN-REFLECTION"},
            {"kind": "region", "id": "Q29.S01"},
            "dependency",
        )
        self.assertEqual(node_region_edge["target"]["kind"], "region")
        self.assertEqual(len(self.memory.connections_for(
            {"kind": "node", "id": "KN-REFLECTION"})), 2)

    def test_rejects_unknown_regions_and_nodes(self):
        with self.assertRaises(ValueError):
            self.memory.add_node("KN-BAD", "concept", "bad", ["Q99"])
        with self.assertRaises(ValueError):
            self.memory.add_connection(
                "CN-BAD", {"kind": "node", "id": "KN-MISSING"},
                {"kind": "region", "id": "Q01"}, "similar",
            )

    def test_reflection_procedure_and_candidate_are_persistent(self):
        self._reflection_procedure()
        self.memory.add_candidate(
            "CB-1", ["PR-REFLECT-H"], ["PR-REFLECT-H"],
            generation_method="single_rule_reuse",
        )
        self.memory.add_evidence(
            "EV-1", "procedure", "PR-REFLECT-H",
            "round_trip_test", "passed",
            input_fingerprint="sha256:input",
            expected_fingerprint="sha256:original",
            actual_fingerprint="sha256:original",
        )
        self.memory.set_procedure_status("PR-REFLECT-H", "verified")
        self.memory.add_evidence("EV-2", "candidate", "CB-1", "held_out_validation", "passed")
        self.memory.set_candidate_status("CB-1", "validated")
        self.memory.record_application("PR-REFLECT-H", "passed", task_ref="task-B")
        self.memory.save()

        restored = HCM32Memory(self.path)
        procedure = restored.data["procedures"]["PR-REFLECT-H"]
        self.assertEqual(procedure["inverse_procedure_id"], "PR-REFLECT-H")
        self.assertEqual(procedure["reuse_successes"], 1)
        self.assertEqual(restored.data["candidates"]["CB-1"]["status"], "validated")
        self.assertEqual(restored.data["evidence"]["EV-1"]["result"], "passed")
        self.assertEqual(restored.data["events"][0]["event_number"], 1)

    def test_verified_status_requires_passed_evidence(self):
        self._reflection_procedure()
        with self.assertRaises(ValueError):
            self.memory.set_procedure_status("PR-REFLECT-H", "verified")
        self.memory.add_candidate(
            "CB-1", ["PR-REFLECT-H"], ["PR-REFLECT-H"],
            generation_method="single_rule_reuse",
        )
        with self.assertRaises(ValueError):
            self.memory.set_candidate_status("CB-1", "validated")

    def test_node_and_connection_promotion_requires_evidence(self):
        self.memory.add_node("KN-A", "concept", "A", ["Q01"])
        self.memory.add_node("KN-B", "concept", "B", ["Q02.S01"])
        self.memory.add_connection(
            "CN-A-B", {"kind": "node", "id": "KN-A"},
            {"kind": "node", "id": "KN-B"}, "similar",
        )
        with self.assertRaises(ValueError):
            self.memory.set_connection_status("CN-A-B", "verified")
        self.memory.add_evidence("EV-N", "node", "KN-A", "training_fit", "passed")
        self.memory.set_node_status("KN-A", "supported")
        self.memory.add_evidence("EV-C", "connection", "CN-A-B", "transfer_test", "passed")
        self.memory.set_connection_status("CN-A-B", "verified")
        self.assertEqual(self.memory.data["nodes"]["KN-A"]["status"], "supported")
        self.assertEqual(self.memory.data["connections"]["CN-A-B"]["status"], "verified")

    def test_failed_application_is_preserved_without_erasing_success(self):
        self._reflection_procedure()
        self.memory.record_application("PR-REFLECT-H", "passed")
        self.memory.record_application("PR-REFLECT-H", "failed", task_ref="counterexample")
        record = self.memory.data["procedures"]["PR-REFLECT-H"]
        self.assertEqual(record["reuse_successes"], 1)
        self.assertEqual(record["reuse_failures"], 1)
        self.assertEqual(len(self.memory.data["events"]), 2)

    def test_duplicate_ids_and_invalid_strength_are_rejected(self):
        self.memory.add_node("KN-A", "concept", "A", ["Q01"])
        with self.assertRaises(ValueError):
            self.memory.add_node("KN-A", "concept", "duplicate", ["Q02"])
        with self.assertRaises(ValueError):
            self.memory.add_connection(
                "CN-A", {"kind": "node", "id": "KN-A"},
                {"kind": "region", "id": "Q02"}, "similar", strength=1.5,
            )

    def test_unverified_candidate_does_not_become_verified_automatically(self):
        self._reflection_procedure()
        candidate = self.memory.add_candidate(
            "CB-UNTESTED", ["PR-REFLECT-H"], ["PR-REFLECT-H"],
            generation_method="composition",
        )
        self.assertEqual(candidate["status"], "generated")
        self.assertNotEqual(candidate["status"], "validated")

    def test_corrupt_or_incompatible_memory_fails_closed(self):
        self.path.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(HCM32FormatError):
            HCM32Memory(self.path)

        self.path.write_text(json.dumps({
            "schema": "hcm32", "schema_version": 999,
        }), encoding="utf-8")
        with self.assertRaises(HCM32FormatError):
            HCM32Memory(self.path)

    def test_loader_rejects_verified_record_without_evidence(self):
        self.memory.add_node("KN-A", "concept", "A", ["Q01"])
        self.memory.save()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        raw["nodes"]["KN-A"]["status"] = "verified"
        self.path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(HCM32FormatError):
            HCM32Memory(self.path)

    def test_rejected_status_requires_failed_evidence(self):
        self.memory.add_node("KN-A", "concept", "A", ["Q01"])
        with self.assertRaises(ValueError):
            self.memory.set_node_status("KN-A", "rejected")
        self.memory.add_evidence("EV-FAIL", "node", "KN-A", "counterexample", "failed")
        self.memory.set_node_status("KN-A", "rejected")
        self.assertEqual(self.memory.data["nodes"]["KN-A"]["status"], "rejected")

    def test_snapshot_is_detached_from_live_memory(self):
        self.memory.add_node("KN-A", "concept", "A", ["Q01"])
        snapshot = self.memory.snapshot()
        snapshot["nodes"]["KN-A"]["name"] = "mutated"
        self.assertEqual(self.memory.data["nodes"]["KN-A"]["name"], "A")


if __name__ == "__main__":
    unittest.main()
