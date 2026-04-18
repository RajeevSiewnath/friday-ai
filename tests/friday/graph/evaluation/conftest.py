import pytest
import tempfile
import json
from friday.core.evaluation import Evaluation
from friday.graph.evaluation.states.questions_state import QuestionsState
from friday.graph.evaluation.states.scores_state import ScoresState


@pytest.fixture
def sample_evaluations():
    return [
        Evaluation(
            question="What is Python?",
            keywords=["programming", "language"],
            answer="Python is a high-level programming language.",
            category="programming",
        ),
        Evaluation(
            question="What is machine learning?",
            keywords=["ML", "AI"],
            answer="Machine learning is a subset of artificial intelligence.",
            category="ai",
        ),
        Evaluation(
            question="What is data science?",
            keywords=["data", "analysis"],
            answer="Data science is an interdisciplinary field.",
            category="data",
        ),
    ]


@pytest.fixture
def evaluation_json_file(sample_evaluations):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump([e.model_dump() for e in sample_evaluations], f)
        return f.name


@pytest.fixture
def questions_state(sample_evaluations):
    return QuestionsState(evaluation_questions=sample_evaluations)


@pytest.fixture
def scores_state():
    return ScoresState(
        evaluation_scores={
            "accuracy": [0.85, 0.90, 0.88],
            "completeness": [0.80, 0.85, 0.82],
        }
    )


@pytest.fixture
def empty_questions_state():
    return QuestionsState(evaluation_questions=[])


@pytest.fixture
def empty_scores_state():
    return ScoresState(evaluation_scores={})
