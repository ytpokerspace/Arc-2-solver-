import tempfile
import unittest
from pathlib import Path

from arc_r2.hmse import (
    MethodLibrary,
    MethodRegistry,
    MethodSpec,
    Procedure,
    ProcedureError,
    SearchPartition,
    Stage,
    ValidationResult,
    compose_procedures,
    execute_procedure,
    replace_stage,
)


class HierarchicalMethodEngineTests(unittest.TestCase):
    def setUp(self):
        self.registry = MethodRegistry()

    def test_partitions_are_searched_fairly_and_best_validated_candidate_wins(self):
        calls = []
        self.registry.register_method(
            "weak", lambda state, params: calls.append("weak") or 1
        )
        self.registry.register_method(
            "strong", lambda state, params: calls.append("strong") or 5
        )
        self.registry.register_method(
            "unused", lambda state, params: calls.append("unused") or 4
        )

        def score(stage, state, output, spec):
            return ValidationResult(output > 0, output / 5)

        self.registry.register_validator("score", score)
        procedure = Procedure("fair-search", (
            Stage(
                "choose", "choice",
                partitions=(
                    SearchPartition("geometry", (
                        MethodSpec("weak"), MethodSpec("unused")
                    )),
                    SearchPartition("relations", (MethodSpec("strong"),)),
                ),
                validator_id="score",
                max_evaluations=2,
                stop_score=1.0,
            ),
        ))

        result = execute_procedure(procedure, {}, self.registry)
        self.assertEqual(calls, ["weak", "strong"])
        self.assertEqual(result.state["choice"], 5)
        self.assertEqual(result.stages[0].selected_partition, "relations")
        self.assertEqual(result.stages[0].evaluations, 2)

    def test_sequential_stages_receive_prior_outputs(self):
        self.registry.register_method(
            "detect", lambda state, params: {"count": state["grid_size"]}
        )
        self.registry.register_method(
            "classify", lambda state, params: state["objects"]["count"] + params["offset"]
        )
        procedure = Procedure("two-stage", (
            Stage("detect", "objects", (
                SearchPartition("perception", (MethodSpec("detect"),)),
            ), requires=("grid_size",)),
            Stage("classify", "class", (
                SearchPartition("abstraction", (
                    MethodSpec("classify", {"offset": 2}, ("objects",)),
                )),
            ), requires=("objects",)),
        ), input_keys=("grid_size",))

        result = execute_procedure(procedure, {"grid_size": 3}, self.registry)
        self.assertEqual(result.state["class"], 5)
        self.assertEqual(result.total_evaluations, 2)

    def test_stage_budget_is_enforced(self):
        calls = []
        for index in range(5):
            method_id = f"m{index}"
            self.registry.register_method(
                method_id, lambda state, params, i=index: calls.append(i) or i
            )
        self.registry.register_validator(
            "never", lambda stage, state, output, spec: ValidationResult(False, 0.0)
        )
        procedure = Procedure("bounded", (
            Stage("search", "answer", (
                SearchPartition("p", tuple(MethodSpec(f"m{i}") for i in range(5))),
            ), validator_id="never", max_evaluations=2),
        ))
        with self.assertRaises(ProcedureError):
            execute_procedure(procedure, {}, self.registry)
        self.assertEqual(len(calls), 2)

    def test_unsatisfied_dependency_is_rejected_before_execution(self):
        procedure = Procedure("invalid", (
            Stage("later", "result", (
                SearchPartition("p", (MethodSpec("anything"),)),
            ), requires=("missing",)),
        ))
        with self.assertRaises(ValueError):
            execute_procedure(procedure, {}, self.registry)

    def test_method_library_round_trip_and_stage_replacement(self):
        original = Procedure("saved", (
            Stage("first", "one", (
                SearchPartition("p", (MethodSpec("add", {"n": 2}),)),
            )),
        ), description="Reusable test procedure")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "methods" / "saved.json"
            MethodLibrary.save(original, path)
            restored = MethodLibrary.load(path)
        self.assertEqual(restored, original)

        replacement = Stage("first", "one", (
            SearchPartition("alternate", (MethodSpec("multiply", {"n": 3}),)),
        ))
        updated = replace_stage(original, "first", replacement)
        self.assertEqual(updated.version, 2)
        self.assertEqual(updated.stages[0], replacement)
        self.assertEqual(original.stages[0], original.stages[0])

    def test_procedures_compose_in_dependency_order(self):
        first = Procedure("one", (
            Stage("a", "x", (SearchPartition("p", (MethodSpec("a"),)),)),
        ))
        second = Procedure("two", (
            Stage("b", "y", (SearchPartition("p", (MethodSpec("b"),)),),
                  requires=("x",)),
        ))
        combined = compose_procedures("combined", first, second)
        self.assertEqual([stage.stage_id for stage in combined.stages], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
