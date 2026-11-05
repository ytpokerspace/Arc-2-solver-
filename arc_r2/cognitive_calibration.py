"""Adaptive Cognitive Calibration Layer for QBS.

The generator receives only a question. A separately supplied evaluator scores
the answer against a reference key that is never passed to the generator.
This protocol does not claim that calibration measures IQ.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, TypeVar

RUBRIC_WEIGHTS = {
    "factual_correctness": 0.35, "completeness": 0.25, "reasoning": 0.20,
    "relevance": 0.10, "consistency": 0.10,
}

@dataclass(frozen=True)
class CalibrationQuestion:
    question_id: str
    domain: str
    prompt: str

@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    confidence: float
    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("EMPTY_GENERATED_ANSWER")
        if not 0.0 <= self.confidence <= 100.0:
            raise ValueError("CONFIDENCE_MUST_BE_0_TO_100")

@dataclass(frozen=True)
class Evaluation:
    correct: bool
    rubric: Mapping[str, float]
    evidence_note: str = ""
    def __post_init__(self):
        if set(self.rubric) != set(RUBRIC_WEIGHTS):
            raise ValueError("RUBRIC_MUST_CONTAIN_ALL_FIVE_CRITERIA")
        if any(not 0.0 <= value <= 1.0 for value in self.rubric.values()):
            raise ValueError("RUBRIC_SCORES_MUST_BE_0_TO_1")

@dataclass(frozen=True)
class CalibrationRecord:
    question_id: str
    domain: str
    answer: str
    confidence: float
    evaluation: Evaluation
    quality_score: float
    score_after: int
    confidence_adjustment_after: float
    consecutive_correct: int

@dataclass(frozen=True)
class CalibrationReport:
    records: tuple[CalibrationRecord, ...]
    initial_score: int
    final_score: int
    confidence_adjustment: float
    stopped_early: bool
    stop_reason: str
    @property
    def questions_used(self) -> int:
        return len(self.records)
    @property
    def accuracy(self) -> float:
        return sum(r.evaluation.correct for r in self.records) / len(self.records) if self.records else 0.0

AnswerGenerator = Callable[[str], GeneratedAnswer]
AnswerEvaluator = Callable[[CalibrationQuestion, GeneratedAnswer], Evaluation]

def weighted_quality(evaluation: Evaluation) -> float:
    return sum(RUBRIC_WEIGHTS[k] * evaluation.rubric[k] for k in RUBRIC_WEIGHTS)

def run_calibration(
    questions: Sequence[CalibrationQuestion],
    generate: AnswerGenerator,
    evaluate: AnswerEvaluator,
    *,
    initial_score: int = 150,
    minimum_score: int = 0,
    maximum_score: int = 300,
    score_step: int = 5,
    stop_after_consecutive_correct: int = 3,
    confidence_tolerance: float = 10.0,
    confidence_step: float = 5.0,
) -> CalibrationReport:
    """Run at most ten questions; the evaluator is called after generation."""
    if not questions:
        raise ValueError("CALIBRATION_REQUIRES_QUESTIONS")
    if not 0 <= minimum_score <= initial_score <= maximum_score:
        raise ValueError("INVALID_CALIBRATION_SCORE_BOUNDS")
    if score_step <= 0 or stop_after_consecutive_correct <= 0:
        raise ValueError("INVALID_CALIBRATION_STEP_OR_STOP_THRESHOLD")
    if len({q.question_id for q in questions}) != len(questions):
        raise ValueError("DUPLICATE_CALIBRATION_QUESTION_ID")

    score, adjustment, streak = initial_score, 0.0, 0
    records = []
    stopped_early = False
    stop_reason = "QUESTION_BANK_EXHAUSTED"
    for question in questions[:10]:
        generated = generate(question.prompt)
        evaluation = evaluate(question, generated)
        quality = weighted_quality(evaluation)
        score = min(maximum_score, score + score_step) if evaluation.correct else max(minimum_score, score - score_step)
        adjusted_confidence = min(100.0, max(0.0, generated.confidence + adjustment))
        observed_quality = quality * 100.0
        if adjusted_confidence - observed_quality > confidence_tolerance:
            adjustment = max(-100.0, adjustment - confidence_step)
        elif observed_quality - adjusted_confidence > confidence_tolerance:
            adjustment = min(100.0, adjustment + confidence_step)
        streak = streak + 1 if evaluation.correct else 0
        records.append(CalibrationRecord(
            question.question_id, question.domain, generated.text,
            generated.confidence, evaluation, quality, score, adjustment, streak
        ))
        if streak >= stop_after_consecutive_correct:
            stopped_early = len(records) < min(10, len(questions))
            stop_reason = "CONSECUTIVE_CORRECT_THRESHOLD_REACHED"
            break
    return CalibrationReport(tuple(records), initial_score, score, adjustment, stopped_early, stop_reason)

ResultT = TypeVar("ResultT")
Brainstorm = Callable[[str, CalibrationReport], ResultT]

def run_calibrated_session(
    user_question: str,
    calibration_questions: Sequence[CalibrationQuestion],
    generate: AnswerGenerator,
    evaluate: AnswerEvaluator,
    brainstorm: Brainstorm[ResultT],
) -> tuple[CalibrationReport, ResultT]:
    """Mandatory orchestration entry point: calibrate first, then brainstorm.

    The user's actual question is not passed to the brainstorming callback
    until the pre-test finishes. The report is passed alongside it so the
    studio can adjust confidence handling without weakening exact-match gates.
    """
    if not user_question.strip():
        raise ValueError("USER_QUESTION_REQUIRED")
    report = run_calibration(calibration_questions, generate, evaluate)
    result = brainstorm(user_question, report)
    return report, result
