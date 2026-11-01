import unittest

from arc_r2.dsl import Program, Instruction
from arc_r2.grid import as_grid
from arc_r2.research_systems import (
    AbstractionLibrary, CandidateLedger, IndependentVerifier, JSONLTelemetry,
    SafeDSLExecutor, SearchBudget, TaskDataManager, TaskRecord, build_scene_graph,
    calibration_metrics, cegis_refine, disagreement_report,
)
from arc_r2.search import SearchLimits
from arc_r2.solver import Example, HybridSolver


class TaskIsolationTests(unittest.TestCase):
    def test_task_manager_only_keeps_test_inputs(self):
        raw = {"train": [{"input": [[0, 1]], "output": [[1, 0]]}],
               "test": [{"input": [[0, 1]]}]}
        manager = TaskDataManager()
        manager.add(TaskRecord.from_mapping("t1", raw))
        self.assertEqual(manager.prediction_inputs("t1"), (as_grid([[0, 1]]),))
        self.assertFalse(manager.summary()["hidden_test_outputs_loaded"])

    def test_rejects_test_outputs_and_duplicate_ids(self):
        raw = {"train": [{"input": [[0]], "output": [[0]]}],
               "test": [{"input": [[0]], "output": [[0]]}]}
        with self.assertRaises(ValueError):
            TaskRecord.from_mapping("leak", raw)
        clean = {"train": [{"input": [[0]], "output": [[0]]}], "test": [{"input": [[0]]}]}
        manager = TaskDataManager()
        manager.add(TaskRecord.from_mapping("same", clean))
        with self.assertRaises(ValueError):
            manager.add(TaskRecord.from_mapping("same", clean))


class ExecutionAndVerificationTests(unittest.TestCase):
    def test_safe_executor_enforces_depth_and_approved_ops(self):
        executor = SafeDSLExecutor(max_depth=1)
        with self.assertRaises(ValueError):
            executor.execute(Program((Instruction("REVERSE.HORIZONTAL"), Instruction("ROTATE.180"))),
                             as_grid([[0, 1]]))
        with self.assertRaises(ValueError):
            executor.execute(Program((Instruction("MAGIC"),)), as_grid([[0, 1]]))

    def test_verifier_requires_all_pairs_exact(self):
        examples = ((as_grid([[0, 1]]), as_grid([[1, 0]])),
                    (as_grid([[0, 0]]), as_grid([[1, 1]])))
        result = IndependentVerifier().verify(Program((Instruction("REVERSE.HORIZONTAL"),)), examples)
        self.assertFalse(result.passed)
        self.assertEqual(result.counterexample_indices, (1,))


class ProvenanceAndBudgetTests(unittest.TestCase):
    def test_ledger_deduplicates_and_records_rejection(self):
        ledger = CandidateLedger()
        program = Program((Instruction("REVERSE.HORIZONTAL"),))
        first = ledger.add(program, origin="beam")
        second = ledger.add(program, origin="mcts")
        self.assertEqual(first.candidate_id, second.candidate_id)
        ledger.reject(first.candidate_id, "counterexample:0")
        self.assertEqual(ledger.export()[0]["rejection_reason"], "counterexample:0")

    def test_budget_stops_at_caps(self):
        budget = SearchBudget(max_expansions=1, max_executions=1, max_program_depth=2,
                              max_seconds=10, max_hybrids=1, max_memory_bytes=2048)
        self.assertTrue(budget.consume_expansion())
        self.assertFalse(budget.consume_expansion())
        self.assertEqual(budget.stopped_reason, "expansion_budget")
        self.assertTrue(budget.consume_memory(1024))
        self.assertFalse(budget.consume_memory(1024 * 1024))
        self.assertEqual(budget.stopped_reason, "estimated_memory_budget")
        self.assertTrue(budget.consume_hybrid())
        self.assertFalse(budget.consume_hybrid())


class LearningAndTelemetryTests(unittest.TestCase):
    def test_cegis_rejects_then_accepts_exact_candidate(self):
        examples = ((as_grid([[0, 1]]), as_grid([[1, 0]])),)
        identity = Program()
        reverse = Program((Instruction("REVERSE.HORIZONTAL"),))
        result = cegis_refine(examples, [identity, reverse])
        self.assertTrue(result.solved)
        self.assertEqual(result.program, reverse)
        self.assertEqual(result.rejection_count, 1)

    def test_cegis_refiner_receives_counterexamples(self):
        examples = ((as_grid([[0, 1]]), as_grid([[1, 0]])),)
        identity = Program()
        reverse = Program((Instruction("REVERSE.HORIZONTAL"),))
        seen = []
        result = cegis_refine(examples, [identity],
                              refine=lambda program, bad: (seen.append(bad) or [reverse]))
        self.assertTrue(result.solved)
        self.assertEqual(seen, [(0,)])

    def test_disagreement_and_calibration(self):
        report = disagreement_report([as_grid([[0]]), as_grid([[1]]), as_grid([[1]])])
        self.assertTrue(report["uncertain"])
        self.assertAlmostEqual(report["agreement"], 2/3)
        metrics = calibration_metrics([(0.9, True), (0.8, False)])
        self.assertEqual(metrics["count"], 2)
        self.assertGreater(metrics["brier_score"], 0)

    def test_abstraction_requires_multiple_validation_tasks(self):
        library = AbstractionLibrary(min_validation_tasks=3, min_exact_rate=0.8)
        program = Program((Instruction("REVERSE.HORIZONTAL"),))
        self.assertFalse(library.propose(program, [("task-a", True), ("task-b", True)]))
        self.assertTrue(library.propose(program, [("task-a", True), ("task-b", True), ("task-c", True), ("task-d", True)]))
        with self.assertRaises(ValueError):
            library.propose(program, [("task-a", True), ("task-a", True), ("task-b", True)])
        item_id = library.export()[0]["abstraction_id"]
        library.record_reuse(item_id)
        self.assertEqual(library.export()[0]["reuse_count"], 1)

    def test_spatial_scene_graph_contains_object_attributes(self):
        graph = build_scene_graph(as_grid([[0, 1, 0, 2], [0, 1, 0, 2]]))
        self.assertEqual(len(graph.nodes), 2)
        self.assertIn("LEFT_OF", graph.edges[0][2])
        self.assertTrue(all(node.size > 0 for node in graph.nodes))


    def test_main_solver_returns_verification_and_provenance(self):
        examples = (Example(as_grid([[0, 1]]), as_grid([[1, 0]])),)
        result = HybridSolver(SearchLimits(max_depth=1, candidate_cap=40, mcts_iterations=5),
                              strategy="beam").solve(examples)
        self.assertTrue(result.verified_exact_fit)
        self.assertTrue(result.verification["passed"])
        self.assertTrue(result.candidate_provenance)
        self.assertIn("elapsed_seconds", result.search_budget)

    def test_main_solver_honors_zero_wall_clock_budget(self):
        examples = (Example(as_grid([[0, 1]]), as_grid([[1, 0]])),)
        result = HybridSolver(SearchLimits(max_depth=2, candidate_cap=40, mcts_iterations=5,
                                           max_seconds=0.0), strategy="beam").solve(examples)
        self.assertFalse(result.verified_exact_fit)
        self.assertTrue(result.search_budget["beam"]["stopped_by_time"])

    def test_jsonl_telemetry_is_parseable(self):
        telemetry = JSONLTelemetry()
        telemetry.emit("candidate", candidate_id="abc", exact=False)
        self.assertEqual(telemetry.to_jsonl().count("\n"), 1)
        self.assertEqual(telemetry.summary()["event_count"], 1)


if __name__ == "__main__":
    unittest.main()
