"""Regression tests for the standalone primitive catalog benchmark."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPOSITORY_ROOT / "scripts" / "benchmark_primitive_catalog.py"


class PrimitiveCatalogBenchmarkTests(unittest.TestCase):
    def test_runs_outside_repository_and_does_not_claim_empty_registry_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary)
            catalog = workdir / "catalog.yaml"
            catalog.write_text(
                "expected_unique_primitive_count: 2\n"
                "total_count: 2\n"
                "catalog_id: test-catalog\n"
                "groups:\n"
                "  geometry:\n"
                "    count: 2\n"
                "    primitives: [flip_h, rot90]\n",
                encoding="utf-8",
            )
            output = workdir / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARK),
                    "--catalog",
                    str(catalog),
                    "--output",
                    str(output),
                ],
                cwd=workdir,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["catalog_valid"])
            self.assertEqual(report["unique_primitive_count"], 2)
            self.assertEqual(report["runtime_registry_status"], "not_wired")
            self.assertIsNone(report["registered_executable_primitive_count"])
            self.assertIsNone(report["registered_coverage_percent"])
            self.assertIsNone(report["unregistered_catalog_entries"])


if __name__ == "__main__":
    unittest.main()
