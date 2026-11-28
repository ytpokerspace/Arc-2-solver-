import unittest

from arc_r2.ahpse import TrainingExample
from arc_r2.ahpse_hmse import (
    build_ahpse_procedure,
    register_ahpse_methods,
    run_ahpse_with_hmse,
)
from arc_r2.hmse import MethodRegistry, ProcedureError, execute_procedure


class AHPSEHMSEIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.examples = (
            TrainingExample.from_values(
                [[0, 1], [2, 0]],
                [[3, 0], [0, 4]],
            ),
            TrainingExample.from_values(
                [[0, 1, 0], [2, 0, 0]],
                [[3, 0], [0, 4], [0, 0]],
            ),
        )

    def test_explicit_opt_in_runs_through_hmse(self):
        result, execution = run_ahpse_with_hmse(
            self.examples, max_depth=1, candidate_cap=20
        )
        self.assertFalse(result.abstained)
        self.assertGreater(result.exact_fit_count, 0)
        self.assertEqual(execution.procedure_id, "arc.ahpse.opt-in.v1")
        self.assertEqual(execution.total_evaluations, 1)
        self.assertEqual(
            execution.stages[0].selected_method, "arc.ahpse.synthesize"
        )
        self.assertEqual(
            execution.state["ahpse_result"].exact_fit_count,
            result.exact_fit_count,
        )

    def test_hmse_records_exact_fit_evidence(self):
        registry = MethodRegistry()
        register_ahpse_methods(registry)
        procedure = build_ahpse_procedure(max_depth=1, candidate_cap=20)
        execution = execute_procedure(
            procedure, {"training_examples": self.examples}, registry
        )
        evidence = execution.stages[0].candidates[0].evidence
        self.assertGreater(evidence["exact_fit_count"], 0)
        self.assertFalse(evidence["abstained"])
        self.assertIn("constraint_checks", evidence)
        self.assertIn("constraint_rejections", evidence)

    def test_no_fit_abstains_and_hmse_does_not_select_it(self):
        contradictory = (
            TrainingExample.from_values([[1]], [[2]]),
            TrainingExample.from_values([[1]], [[3]]),
        )
        with self.assertRaises(ProcedureError):
            run_ahpse_with_hmse(contradictory, max_depth=0)

    def test_missing_training_examples_is_rejected(self):
        registry = MethodRegistry()
        register_ahpse_methods(registry)
        procedure = build_ahpse_procedure()
        with self.assertRaises((ValueError, ProcedureError)):
            execute_procedure(procedure, {}, registry)


if __name__ == "__main__":
    unittest.main()
