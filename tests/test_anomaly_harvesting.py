import unittest

from arc_r2.anomaly_harvesting import AnomalyHarvestingEngine, BoundedDSLRegistry, compute_residual
from arc_r2.primitives import Primitive, Program, as_grid


class ResidualTests(unittest.TestCase):
    def test_residual_mask_and_norm(self):
        mask, norm = compute_residual(as_grid([[1, 0], [0, 2]]), as_grid([[1, 2], [0, 2]]))
        self.assertEqual(mask, ((0, 1), (0, 0)))
        self.assertEqual(norm, 1)

    def test_shape_mismatch(self):
        mask, norm = compute_residual(as_grid([[1]]), as_grid([[1, 2], [3, 4]]))
        self.assertEqual(mask, ((1, 1), (1, 1)))
        self.assertEqual(norm, 5)


class RegistryTests(unittest.TestCase):
    def test_budget_and_rejection(self):
        registry = BoundedDSLRegistry(1024)
        self.assertTrue(registry.register(Program((Primitive("FLIP_H"),))))
        self.assertLessEqual(registry.memory_usage, registry.max_memory_bytes)
        tiny = BoundedDSLRegistry(1)
        self.assertFalse(tiny.register(Program((Primitive("FLIP_H"),))))
        self.assertEqual(tiny.memory_usage, 0)


class HarvestingTests(unittest.TestCase):
    def test_baseline_exact_fit_skips_harvesting(self):
        source = as_grid([[0, 1], [0, 0]])
        target = as_grid([[1, 0], [0, 0]])
        result = AnomalyHarvestingEngine().solve("flip", ((source, target),))
        self.assertTrue(result.exact_fit)
        self.assertEqual(result.traces_collected, 0)
        self.assertEqual(result.probes_tested, 0)

    def test_any_retained_program_fits_all_pairs(self):
        source = as_grid([[0, 1, 0], [0, 0, 0]])
        target = as_grid([[0, 0, 0], [0, 0, 1]])
        pairs = ((source, target),)
        result = AnomalyHarvestingEngine(max_probes_per_cycle=64).solve("compose", pairs)
        if result.program is not None:
            self.assertTrue(all(result.program.apply(a) == b for a, b in pairs))
        self.assertLessEqual(result.probes_tested, 64)
        self.assertLessEqual(result.memory_bytes, 512 * 1024)

    def test_conflicting_pairs_never_return_partial_fit(self):
        source = as_grid([[0, 1], [0, 0]])
        pairs = ((source, as_grid([[1, 0], [0, 0]])), (source, source))
        result = AnomalyHarvestingEngine(max_probes_per_cycle=32).solve("conflict", pairs)
        if result.program is not None:
            self.assertTrue(all(result.program.apply(a) == b for a, b in pairs))
        self.assertLessEqual(result.probes_tested, 32)


if __name__ == "__main__":
    unittest.main()
