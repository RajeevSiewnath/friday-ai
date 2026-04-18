import pytest
from typing import TypedDict
from friday.core.evaluation import Evaluation
from friday.graph.training.states.training_state import TrainingState


class TestTrainingState:
    def test_training_state_creation(self, training_state: TrainingState):
        assert "full_data_set" in training_state
        assert "training_data_set" in training_state
        assert "validating_data_set" in training_state
        assert "testing_data_set" in training_state

    def test_training_state_full_data_set(self, training_state: TrainingState):
        assert isinstance(training_state["full_data_set"], list)
        assert len(training_state["full_data_set"]) > 0
        assert all(isinstance(e, Evaluation) for e in training_state["full_data_set"])

    def test_training_state_training_data_set(self, training_state: TrainingState):
        assert isinstance(training_state["training_data_set"], list)
        assert len(training_state["training_data_set"]) > 0
        assert all(
            isinstance(e, Evaluation) for e in training_state["training_data_set"]
        )

    def test_training_state_validating_data_set(self, training_state: TrainingState):
        assert isinstance(training_state["validating_data_set"], list)
        assert all(
            isinstance(e, Evaluation) for e in training_state["validating_data_set"]
        )

    def test_training_state_testing_data_set(self, training_state: TrainingState):
        assert isinstance(training_state["testing_data_set"], list)
        assert all(isinstance(e, Evaluation) for e in training_state["testing_data_set"])

    def test_training_state_has_evaluation_attributes(self, training_state: TrainingState):
        evaluation = training_state["full_data_set"][0]
        assert hasattr(evaluation, "question")
        assert hasattr(evaluation, "answer")
        assert hasattr(evaluation, "category")
        assert hasattr(evaluation, "keywords")

    def test_training_state_annotations(self):
        assert hasattr(TrainingState, "__annotations__")
        expected_keys = {"full_data_set", "training_data_set", "validating_data_set", "testing_data_set"}
        assert expected_keys == set(TrainingState.__annotations__.keys())

    def test_training_state_data_sets_are_lists_of_evaluations(self, training_state):
        for key in ["full_data_set", "training_data_set", "validating_data_set", "testing_data_set"]:
            assert isinstance(training_state[key], list)
            if training_state[key]:
                for item in training_state[key]:
                    assert isinstance(item, Evaluation)
