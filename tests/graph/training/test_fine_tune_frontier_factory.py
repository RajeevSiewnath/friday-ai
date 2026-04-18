import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from friday.core.evaluation import Evaluation
from friday.graph.training.states.training_state import TrainingState


class TestFineTuneFrontierFactory:
    def test_factory_can_be_called(self):
        from friday.graph.training.fine_tune_frontier_factory import (
            fine_tune_frontier_factory,
        )

        result = fine_tune_frontier_factory()
        assert result is None

    def test_factory_accepts_all_parameters(self):
        from friday.graph.training.fine_tune_frontier_factory import (
            fine_tune_frontier_factory,
        )

        result = fine_tune_frontier_factory(
            suffix="test_suffix",
            model="gpt-4",
            n_epochs=3,
            batch_size=32,
            learning_rate_multiplier=0.5,
            wait_for_fine_tune_to_complete=True,
        )
        assert result is None

    def test_factory_with_default_parameters(self):
        from friday.graph.training.fine_tune_frontier_factory import (
            fine_tune_frontier_factory,
        )

        result = fine_tune_frontier_factory()
        assert result is None

    def test_factory_with_none_model(self):
        from friday.graph.training.fine_tune_frontier_factory import (
            fine_tune_frontier_factory,
        )

        result = fine_tune_frontier_factory(model=None)
        assert result is None

    def test_factory_with_custom_suffix(self):
        from friday.graph.training.fine_tune_frontier_factory import (
            fine_tune_frontier_factory,
        )

        result = fine_tune_frontier_factory(suffix="custom_suffix")
        assert result is None

    def test_factory_with_custom_model(self):
        from friday.graph.training.fine_tune_frontier_factory import (
            fine_tune_frontier_factory,
        )

        result = fine_tune_frontier_factory(model="gpt-4-turbo")
        assert result is None

    def test_factory_with_training_hyperparameters(self):
        from friday.graph.training.fine_tune_frontier_factory import (
            fine_tune_frontier_factory,
        )

        result = fine_tune_frontier_factory(
            n_epochs=5, batch_size=16, learning_rate_multiplier=0.3
        )
        assert result is None

    def test_factory_with_wait_enabled(self):
        from friday.graph.training.fine_tune_frontier_factory import (
            fine_tune_frontier_factory,
        )

        result = fine_tune_frontier_factory(wait_for_fine_tune_to_complete=True)
        assert result is None

    def test_factory_with_wait_disabled(self):
        from friday.graph.training.fine_tune_frontier_factory import (
            fine_tune_frontier_factory,
        )

        result = fine_tune_frontier_factory(wait_for_fine_tune_to_complete=False)
        assert result is None

    def test_factory_returns_same_type_for_different_calls(self):
        from friday.graph.training.fine_tune_frontier_factory import (
            fine_tune_frontier_factory,
        )

        result1 = fine_tune_frontier_factory()
        result2 = fine_tune_frontier_factory(suffix="test")
        assert type(result1) == type(result2)
        assert result1 is None
        assert result2 is None

    def test_factory_with_all_custom_parameters(self):
        from friday.graph.training.fine_tune_frontier_factory import (
            fine_tune_frontier_factory,
        )

        result = fine_tune_frontier_factory(
            suffix="test",
            model="test-model",
            n_epochs=2,
            batch_size=8,
            learning_rate_multiplier=0.2,
            wait_for_fine_tune_to_complete=True,
        )
        assert result is None

    def test_factory_creates_independent_calls(self):
        from friday.graph.training.fine_tune_frontier_factory import (
            fine_tune_frontier_factory,
        )

        result1 = fine_tune_frontier_factory(suffix="suffix1")
        result2 = fine_tune_frontier_factory(suffix="suffix2")
        assert result1 is result2
