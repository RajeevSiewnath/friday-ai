import pytest
from friday.core.evaluation import Evaluation
from friday.graph.training.states.training_state import TrainingState
from friday.graph.training.states.statistics_state import StatisticsState


@pytest.fixture
def sample_evaluation():
    return Evaluation(
        question="What is test?",
        keywords=["test"],
        answer="A test is a procedure to verify functionality.",
        category="testing",
    )


@pytest.fixture
def empty_training_state():
    return TrainingState(
        full_data_set=[],
        training_data_set=[],
        validating_data_set=[],
        testing_data_set=[],
    )
