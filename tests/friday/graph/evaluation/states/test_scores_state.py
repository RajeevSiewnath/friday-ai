import pytest
from friday.graph.evaluation.states.scores_state import ScoresState


class TestScoresState:
    def test_scores_state_creation(self, scores_state: ScoresState):
        assert "evaluation_scores" in scores_state
        assert isinstance(scores_state["evaluation_scores"], dict)

    def test_scores_state_has_evaluation_scores_key(self, scores_state: ScoresState):
        assert "evaluation_scores" in scores_state

    def test_scores_state_evaluation_scores_is_dict(self, scores_state: ScoresState):
        assert isinstance(scores_state["evaluation_scores"], dict)

    def test_scores_state_empty(self, empty_scores_state: ScoresState):
        assert empty_scores_state["evaluation_scores"] == {}

    def test_scores_state_annotations(self):
        assert hasattr(ScoresState, "__annotations__")
        assert "evaluation_scores" in ScoresState.__annotations__

    def test_scores_state_contains_lists(self, scores_state: ScoresState):
        scores = scores_state["evaluation_scores"]
        for key, values in scores.items():
            assert isinstance(key, str)
            assert isinstance(values, list)

    def test_scores_state_multiple_metrics(self, scores_state: ScoresState):
        scores = scores_state["evaluation_scores"]
        assert len(scores) > 0
        for key, values in scores.items():
            assert isinstance(values, list)
            for value in values:
                assert isinstance(value, (int, float))

    def test_scores_state_single_metric(self):
        single_metric: ScoresState = ScoresState(evaluation_scores={"accuracy": [0.9]})
        assert len(single_metric["evaluation_scores"]) == 1
        assert single_metric["evaluation_scores"]["accuracy"] == [0.9]

    def test_scores_state_can_be_modified(self, scores_state: ScoresState):
        original_keys = set(scores_state["evaluation_scores"].keys())
        scores_state["evaluation_scores"]["new_metric"] = [0.75, 0.80]
        new_keys = set(scores_state["evaluation_scores"].keys())
        assert len(new_keys) > len(original_keys)

    def test_scores_state_numeric_values(self, scores_state: ScoresState):
        scores = scores_state["evaluation_scores"]
        for values in scores.values():
            for value in values:
                assert isinstance(value, (int, float))

    def test_scores_state_with_empty_lists(self):
        empty_lists: ScoresState = ScoresState(evaluation_scores={"metric1": []})
        assert empty_lists["evaluation_scores"]["metric1"] == []

    def test_scores_state_with_many_values(self):
        many_values: ScoresState = ScoresState(
            evaluation_scores={
                "accuracy": [0.8, 0.85, 0.9, 0.95, 0.88, 0.92]
            }
        )
        assert len(many_values["evaluation_scores"]["accuracy"]) == 6
