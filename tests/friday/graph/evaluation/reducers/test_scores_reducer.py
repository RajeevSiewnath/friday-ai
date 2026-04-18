import pytest
from friday.graph.evaluation.reducers.scores_reducer import scores_reducer


class TestScoresReducer:
    def test_reducer_returns_dict(self, scores_dict_1, empty_scores_dict):
        result = scores_reducer(scores_dict_1, empty_scores_dict)
        assert isinstance(result, dict)

    def test_reducer_with_empty_right(self, scores_dict_1, empty_scores_dict):
        left = scores_dict_1
        right = empty_scores_dict
        result = scores_reducer(left, right)
        assert isinstance(result, dict)

    def test_reducer_with_empty_left_and_right(self, empty_scores_dict):
        left = empty_scores_dict
        right = empty_scores_dict
        result = scores_reducer(left, right)
        assert isinstance(result, dict)

    def test_reducer_preserves_left(self, scores_dict_1):
        left = scores_dict_1
        right = {}
        result = scores_reducer(left, right)
        assert result == left

    def test_reducer_deep_copies_left(self, scores_dict_1):
        left = scores_dict_1
        right = {}
        result = scores_reducer(left, right)
        # Modify result to verify deep copy
        if result:
            first_key = list(result.keys())[0]
            result[first_key].append(999)
            assert 999 not in left[first_key]

    def test_reducer_with_empty_dicts(self):
        left = {}
        right = {}
        result = scores_reducer(left, right)
        assert result == {}

    def test_reducer_single_metric(self, single_metric_dict):
        left = single_metric_dict
        right = {}
        result = scores_reducer(left, right)
        assert "metric" in result
        assert result["metric"] == [0.75]

    def test_reducer_returns_state_not_modified_left(self, scores_dict_1):
        left = scores_dict_1.copy()
        right = {}
        result = scores_reducer(left, right)
        # Result should be a deepcopy of left
        assert result == left

    def test_reducer_type_returned(self, scores_dict_1, empty_scores_dict):
        result = scores_reducer(scores_dict_1, empty_scores_dict)
        assert isinstance(result, dict)

    def test_reducer_preserves_all_keys_from_left(self, scores_dict_1):
        left = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
        right = {}
        result = scores_reducer(left, right)
        assert set(result.keys()) == {"a", "b", "c"}

    def test_reducer_preserves_list_contents(self, scores_dict_1):
        left = {"metric": [0.1, 0.2, 0.3]}
        right = {}
        result = scores_reducer(left, right)
        assert result["metric"] == [0.1, 0.2, 0.3]

    def test_reducer_with_complex_dicts(self):
        left = {
            "accuracy": [0.8, 0.85, 0.9],
            "completeness": [0.75, 0.80],
            "relevance": [0.88],
        }
        right = {}
        result = scores_reducer(left, right)
        assert len(result) == 3
        assert len(result["accuracy"]) == 3

    def test_reducer_empty_lists_in_dict(self):
        left = {"metric": []}
        right = {}
        result = scores_reducer(left, right)
        assert result["metric"] == []

    def test_reducer_numeric_values_preserved(self):
        left = {"scores": [1, 2.5, 3.14, 4]}
        right = {}
        result = scores_reducer(left, right)
        assert result["scores"] == [1, 2.5, 3.14, 4]

    def test_reducer_maintains_order(self):
        left = {"z": [1], "a": [2], "m": [3]}
        right = {}
        result = scores_reducer(left, right)
        # Dict maintains insertion order in Python 3.7+
        assert list(result.keys()) == ["z", "a", "m"]
