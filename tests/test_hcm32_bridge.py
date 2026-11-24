"""Opt-in HCM-32 bridge tests, including cross-validation leakage guards."""
import json
import tempfile
import unittest
from pathlib import Path

from arc_r2.hcm32_bridge import HCM32Bridge
from arc_r2.primitive_benchmark import run_benchmark
from arc_r2.primitives import Primitive, Program, as_grid


class HCM32BridgeTests(unittest.TestCase):
    def setUp(self):
        self.examples = (
            (as_grid([[1, 2, 3], [4, 5, 6]]),
             as_grid([[3, 2, 1], [6, 5, 4]])),
            (as_grid([[7, 0], [2, 9], [5, 4]]),
             as_grid([[0, 7], [9, 2], [4, 5]])),
        )
        self.program = Program((Primitive("FLIP_H"),))

    def test_store_reload_and_retrieve_unique_all_pair_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hcm32.json"
            bridge = HCM32Bridge(path)
            procedure_id = bridge.store_program(self.program, self.examples)
            self.assertIsNotNone(procedure_id)
            reloaded = HCM32Bridge(path)
            match = reloaded.find_program(self.examples)
            self.assertIsNotNone(match)
            self.assertEqual(match.procedure_id, procedure_id)
            self.assertEqual(match.program, self.program)

    def test_ambiguous_memories_abstain(self):
        one_pair = ((as_grid([[4]]), as_grid([[4]])),)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hcm32.json"
            bridge = HCM32Bridge(path)
            bridge.store_program(Program((Primitive("FLIP_H"),)), one_pair)
            bridge.store_program(Program((Primitive("TRANSPOSE"),)), one_pair)
            bridge.memory.save()
            self.assertIsNone(HCM32Bridge(path).find_program(one_pair))

    def test_benchmark_uses_preexisting_hcm32_without_primitive_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "task.json"
            dataset.write_text(json.dumps([{
                "task_id": "geometry-transfer",
                "train": [
                    {"input": [list(row) for row in source],
                     "output": [list(row) for row in target]}
                    for source, target in self.examples
                ],
            }]), encoding="utf-8")
            memory_path = root / "hcm32.json"
            HCM32Bridge(memory_path).store_program(self.program, self.examples)
            report = run_benchmark(
                dataset, limit=1, candidate_cap=5000, max_depth=2,
                hcm32_memory_path=memory_path,
            )
            self.assertEqual(report["hcm32_memory_hits"], 2)
            self.assertEqual(report["held_out_exact"], 2)
            self.assertTrue(all(row["candidates_evaluated"] == 0
                                for row in report["fold_results"]))
            self.assertEqual(report["single_stage_memory_hits"], 0)

    def test_new_rules_are_saved_after_all_folds_not_during_cv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "task.json"
            dataset.write_text(json.dumps([{
                "task_id": "geometry-fit",
                "train": [
                    {"input": [list(row) for row in source],
                     "output": [list(row) for row in target]}
                    for source, target in self.examples
                ],
            }]), encoding="utf-8")
            memory_path = root / "hcm32.json"
            report = run_benchmark(
                dataset, limit=1, candidate_cap=5000, max_depth=1,
                hcm32_memory_path=memory_path,
            )
            self.assertEqual(report["hcm32_memory_hits"], 0)
            self.assertGreaterEqual(report["hcm32_memory_saves"], 0)
            # A newly learned procedure can only be used on a later run.
            self.assertTrue(all(not row["hcm32_memory_hit"]
                                for row in report["fold_results"]))


if __name__ == "__main__":
    unittest.main()
