import pytest
from typing import TypedDict
from friday.graph.training.states.statistics_state import StatisticsState


class TestStatisticsState:
    def test_statistics_state_creation(self, statistics_state: StatisticsState):
        assert "data_set_statistics" in statistics_state
        assert isinstance(statistics_state["data_set_statistics"], dict)

    def test_statistics_state_has_data_set_statistics_key(self, statistics_state: StatisticsState):
        assert "data_set_statistics" in statistics_state
        stats = statistics_state["data_set_statistics"]
        assert isinstance(stats, dict)

    def test_statistics_state_values_are_floats(self, statistics_state: StatisticsState):
        stats = statistics_state["data_set_statistics"]
        for key, value in stats.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float))

    def test_statistics_state_can_have_empty_statistics(self):
        empty_stats: StatisticsState = StatisticsState(data_set_statistics={})
        assert empty_stats["data_set_statistics"] == {}

    def test_statistics_state_can_have_multiple_metrics(self):
        metrics: StatisticsState = StatisticsState(
            data_set_statistics={
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.88,
                "f1_score": 0.90,
            }
        )
        assert len(metrics["data_set_statistics"]) == 4
        assert metrics["data_set_statistics"]["accuracy"] == 0.95

    def test_statistics_state_annotations(self):
        assert hasattr(StatisticsState, "__annotations__")
        assert "data_set_statistics" in StatisticsState.__annotations__

    def test_statistics_state_single_metric(self):
        single_metric: StatisticsState = StatisticsState(data_set_statistics={"loss": 0.23})
        assert single_metric["data_set_statistics"]["loss"] == 0.23

    def test_statistics_state_numeric_values(self, statistics_state: StatisticsState):
        stats = statistics_state["data_set_statistics"]
        for value in stats.values():
            assert value >= 0  # Assuming metrics are non-negative
