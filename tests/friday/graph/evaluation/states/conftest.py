import pytest
from friday.core.evaluation import Evaluation
from friday.graph.evaluation.states.questions_state import QuestionsState
from friday.graph.evaluation.states.scores_state import ScoresState


@pytest.fixture
def sample_evaluation():
    return Evaluation(
        question="What is AI?",
        answer="AI is artificial intelligence",
        category="technology",
        keywords=["AI", "intelligence"],
    )


@pytest.fixture
def multiple_evaluations():
    return [
        Evaluation(
            question="Q1?",
            answer="A1",
            category="cat1",
            keywords=["k1", "k2"],
        ),
        Evaluation(
            question="Q2?",
            answer="A2",
            category="cat2",
            keywords=["k3"],
        ),
    ]


@pytest.fixture
def sample_questions_state(sample_evaluation):
    return QuestionsState(evaluation_questions=[sample_evaluation])


@pytest.fixture
def sample_scores_dict():
    return {
        "accuracy": [0.8, 0.85],
        "completeness": [0.75, 0.80],
        "relevance": [0.90, 0.85],
    }
