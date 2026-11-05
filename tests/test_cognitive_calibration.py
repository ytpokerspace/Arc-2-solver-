import unittest
from arc_r2.cognitive_calibration import CalibrationQuestion, Evaluation, GeneratedAnswer, run_calibration, run_calibrated_session, weighted_quality

def grade(correct=True):
    return Evaluation(correct, {
        "factual_correctness": 1.0, "completeness": 1.0, "reasoning": 1.0,
        "relevance": 1.0, "consistency": 1.0,
    })

class CognitiveCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.questions = tuple(CalibrationQuestion(f"Q{i}", "science", f"Explain concept {i} in detail.") for i in range(1, 11))

    def test_stops_after_three_consecutive_correct(self):
        seen = []
        def generate(prompt):
            seen.append(prompt)
            return GeneratedAnswer("A complete reasoned answer.", 90.0)
        report = run_calibration(self.questions, generate, lambda q, a: grade())
        self.assertEqual(report.questions_used, 3)
        self.assertTrue(report.stopped_early)
        self.assertEqual(report.final_score, 165)
        self.assertEqual(len(seen), 3)

    def test_incorrect_resets_streak_and_decrements_score(self):
        outcomes = iter([True, False, True, True, True])
        report = run_calibration(self.questions, lambda p: GeneratedAnswer("Answer", 80.0), lambda q, a: grade(next(outcomes)))
        self.assertEqual(report.questions_used, 5)
        self.assertEqual(report.final_score, 165)
        self.assertEqual(report.records[1].consecutive_correct, 0)

    def test_confidence_adjustment_is_separate(self):
        report = run_calibration(self.questions, lambda p: GeneratedAnswer("Answer", 40.0), lambda q, a: grade())
        self.assertEqual(report.final_score, 165)
        self.assertEqual(report.confidence_adjustment, 15.0)

    def test_generator_receives_prompt_only(self):
        calls = []
        def generate(prompt):
            self.assertIsInstance(prompt, str)
            calls.append(prompt)
            return GeneratedAnswer("Answer", 70.0)
        run_calibration(self.questions, generate, lambda q, a: grade())
        self.assertEqual(len(calls), 3)

    def test_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "DUPLICATE"):
            run_calibration((self.questions[0], self.questions[0]), lambda p: GeneratedAnswer("A", 50), lambda q, a: grade())

    def test_weighted_rubric(self):
        self.assertAlmostEqual(weighted_quality(grade()), 1.0)

    def test_brainstorming_runs_only_after_calibration(self):
        events = []
        def generate(prompt):
            events.append("generate")
            return GeneratedAnswer("Answer", 90.0)
        def evaluate(question, answer):
            events.append("evaluate")
            return grade()
        def brainstorm(question, report):
            events.append("brainstorm")
            self.assertEqual(question, "Solve this task")
            self.assertEqual(report.questions_used, 3)
            return "started"
        report, result = run_calibrated_session("Solve this task", self.questions, generate, evaluate, brainstorm)
        self.assertEqual(result, "started")
        self.assertEqual(events[-1], "brainstorm")
        self.assertEqual(events.count("generate"), 3)
        self.assertEqual(events.count("evaluate"), 3)

if __name__ == "__main__":
    unittest.main()
