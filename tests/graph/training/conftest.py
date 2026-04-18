import pytest
import tempfile
import json
from friday.core.evaluation import Evaluation
from friday.graph.training.states.training_state import TrainingState
from friday.graph.training.states.statistics_state import StatisticsState


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
        Evaluation(
            question="What is deep learning?",
            keywords=["neural", "networks"],
            answer="Deep learning uses neural networks with multiple layers.",
            category="ai",
        ),
        Evaluation(
            question="What is NLP?",
            keywords=["language", "processing"],
            answer="Natural language processing deals with text and language.",
            category="nlp",
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
def training_state(sample_evaluations):
    return TrainingState(
        full_data_set=sample_evaluations,
        training_data_set=sample_evaluations[:3],
        validating_data_set=sample_evaluations[3:4],
        testing_data_set=sample_evaluations[4:],
    )


@pytest.fixture
def statistics_state():
    return StatisticsState(data_set_statistics={"accuracy": 0.85, "precision": 0.90})
