import json
import tempfile
import unittest
from pathlib import Path

from arc_r2.cli import main


class CliReasoningMemoryIntegrationTests(unittest.TestCase):
    def test_demo_run_persists_reasoning_record_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            exit_code = main([
                "--demo", "--output", str(output),
                "--candidate-cap", "40", "--mcts-iterations", "0",
            ])
            self.assertEqual(exit_code, 0)
            memory_path = output / "reasoning_memory.json"
            self.assertTrue(memory_path.is_file())
            memory = json.loads(memory_path.read_text(encoding="utf-8"))
            self.assertEqual(memory["schema_version"], 1)
            records = list(memory["records"].values())
            self.assertEqual(len(records), 1)
            self.assertIn(records[0]["status"], {"tested", "implemented"})
            self.assertEqual(records[0]["outcome"]["total_training_examples"], 2)
            log = output / "baseline_run.jsonl"
            self.assertTrue(log.is_file())
            logged = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(logged["reasoning_memory_status"], records[0]["status"])
            self.assertEqual(logged["verified_patterns_available"], 0)


if __name__ == "__main__":
    unittest.main()
