import unittest

from arc_r2.ahpse import TrainingExample
from arc_r2.full_orchestration import build_full_procedure, run_full_orchestration


class FullOrchestrationTests(unittest.TestCase):
    def test_runs_perception_synthesis_and_consensus_stages(self):
        examples = (
            TrainingExample.from_values([[1, 0], [0, 2]], [[1, 0], [0, 2]]),
            TrainingExample.from_values([[0, 3], [4, 0]], [[0, 3], [4, 0]]),
        )
        test_input = [[5, 0], [0, 6]]
        result = run_full_orchestration(examples, (test_input,), max_depth=0, candidate_cap=40)
        self.assertEqual(
            [stage.stage_id for stage in result.execution.stages],
            ["perception", "candidate-synthesis", "consensus-and-abstention"],
        )
        self.assertEqual(result.execution.total_evaluations, 3)
        self.assertEqual(result.decision.predictions, (((5, 0), (0, 6)),))
        self.assertEqual(result.decision.abstentions, 0)
        self.assertEqual(result.radar_reports["schema"], "arc-full-perception/v1")
        self.assertEqual(len(result.radar_reports["training_outputs"]), 2)
        self.assertEqual(result.candidate_pool.family_evidence["perception"]["training_pairs_scanned"], 2)
        evidence = result.candidate_pool.family_evidence
        self.assertEqual(set(evidence["family_timing_seconds"]),
                         {"ahpse", "primitive", "registered_rules", "object"})
        self.assertEqual(set(evidence["family_candidate_counts"]),
                         {"ahpse", "primitive", "registered_rules", "object"})
        self.assertTrue(all(value >= 0 for value in evidence["family_timing_seconds"].values()))

    def test_disagreement_abstains_instead_of_selecting_first_candidate(self):
        examples = (
            TrainingExample.from_values([[1, 1], [1, 1]], [[1, 1], [1, 1]]),
        )
        test_input = [[1, 0], [0, 0]]
        result = run_full_orchestration(examples, (test_input,), max_depth=1, candidate_cap=80)
        self.assertIsNone(result.decision.predictions[0])
        self.assertTrue(result.decision.ambiguous[0])
        self.assertEqual(result.decision.abstentions, 1)

    def test_no_exact_fit_is_a_recorded_abstention_not_pipeline_failure(self):
        examples = (
            TrainingExample.from_values([[1]], [[2]]),
            TrainingExample.from_values([[1]], [[3]]),
        )
        result = run_full_orchestration(examples, ([[4]],), max_depth=0, candidate_cap=20)
        self.assertEqual(result.decision.predictions, (None,))
        self.assertEqual(result.decision.candidate_count, 0)
        self.assertEqual(result.execution.total_evaluations, 3)


    def test_registered_rule_families_are_reused_by_full_orchestration(self):
        examples = (
            TrainingExample.from_values([[0, 1], [0, 0]], [[1, 0], [0, 0]]),
            TrainingExample.from_values([[0, 2], [0, 0]], [[2, 0], [0, 0]]),
        )
        result = run_full_orchestration(
            examples, ([[0, 3], [0, 0]],), max_depth=1, candidate_cap=100
        )
        rules = result.candidate_pool.family_evidence["registered_rules"]
        reflection = next(item for item in rules if item["rule_id"] == "reflection")
        self.assertTrue(reflection["accepted"])
        self.assertEqual(result.decision.predictions[0], ((3, 0), (0, 0)))
        self.assertIn("skipped", result.candidate_pool.family_evidence["object"])
        self.assertIn(
            "every requested test input already has an applicable",
            result.candidate_pool.family_evidence["object"]["skipped"],
        )
        self.assertEqual(
            result.candidate_pool.family_evidence["object"]["runtime_seconds"], 0.0
        )

    def test_procedure_is_three_stage_and_budgeted(self):
        procedure = build_full_procedure(max_depth=2, candidate_cap=100)
        self.assertEqual(procedure.estimated_steps, 3)
        self.assertEqual(procedure.search_budget, 3)


if __name__ == "__main__":
    unittest.main()
