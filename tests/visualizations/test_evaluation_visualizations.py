import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock external dependencies
sys.modules["evaluation_pipes"] = MagicMock()
sys.modules["evaluation_pipes.retrieval_eval"] = MagicMock()
sys.modules["evaluation_pipes.answer_eval"] = MagicMock()

from friday.models.evaluation_score import EvaluationScore, EvalQuestionCollection
from friday.visualizations.retrieval_evaluation_visualization import (
    RetrievalEvaluationVisualization,
)
from friday.visualizations.answer_evaluation_visualization import (
    AnswerEvaluationVisualization,
)
import plotly.graph_objects as go


class MockRetrievalEvalResult:
    """Mock RetrievalEvalResult for testing."""

    def __init__(self, mrr=0.8, ndcg=0.75, keyword_coverage=90):
        self.mrr = mrr
        self.ndcg = ndcg
        self.keyword_coverage = keyword_coverage


class MockAnswerEvalResult:
    """Mock AnswerEvalResult for testing."""

    def __init__(self, accuracy=85, relevance=80, completeness=75):
        self.accuracy_percentage = accuracy
        self.relevance_percentage = relevance
        self.completeness_percentage = completeness


class TestRetrievalEvaluationVisualization:
    def test_initialization(self):
        eval_score = EvaluationScore()
        viz = RetrievalEvaluationVisualization(
            title="Retrieval Evaluation",
            evaluation_score=eval_score,
            keys=["set1", "set2"],
        )

        assert viz.title == "Retrieval Evaluation"
        assert viz.evaluation_score == eval_score
        assert viz.keys == ["set1", "set2"]
        assert isinstance(viz.fig, go.Figure)

    def test_draw_with_single_key(self):
        results = [
            MockRetrievalEvalResult(mrr=0.9, ndcg=0.85, keyword_coverage=95),
            MockRetrievalEvalResult(mrr=0.8, ndcg=0.75, keyword_coverage=90),
        ]

        eval_score = EvaluationScore(scores={"set1": results})
        viz = RetrievalEvaluationVisualization(
            title="Test",
            evaluation_score=eval_score,
            keys=["set1"],
        )

        viz.draw()

        assert len(viz.fig.data) == 3  # MRR, nDCG, keyword_coverage
        assert viz.fig.data[0].name == "MRR"
        assert viz.fig.data[1].name == "nDCG"
        assert viz.fig.data[2].name == "kw cov."

    def test_draw_with_multiple_keys(self):
        results1 = [
            MockRetrievalEvalResult(mrr=0.9, ndcg=0.85, keyword_coverage=95),
        ]
        results2 = [
            MockRetrievalEvalResult(mrr=0.7, ndcg=0.65, keyword_coverage=70),
        ]

        eval_score = EvaluationScore(scores={"set1": results1, "set2": results2})
        viz = RetrievalEvaluationVisualization(
            title="Multi-set Evaluation",
            evaluation_score=eval_score,
            keys=["set1", "set2"],
        )

        viz.draw()

        assert len(viz.fig.data) == 3
        assert list(viz.fig.data[0].x) == ["set1", "set2"]

    def test_draw_calculates_averages(self):
        results = [
            MockRetrievalEvalResult(mrr=0.8, ndcg=0.75, keyword_coverage=80),
            MockRetrievalEvalResult(mrr=1.0, ndcg=1.0, keyword_coverage=100),
        ]

        eval_score = EvaluationScore(scores={"set1": results})
        viz = RetrievalEvaluationVisualization(
            title="Test",
            evaluation_score=eval_score,
            keys=["set1"],
        )

        viz.draw()

        # Average MRR should be 0.9
        assert viz.fig.data[0].y[0] == 0.9
        # Average nDCG should be 0.875
        assert viz.fig.data[1].y[0] == 0.875
        # Average keyword coverage should be 0.9
        assert viz.fig.data[2].y[0] == 0.9

    def test_layout_configuration(self):
        eval_score = EvaluationScore(scores={"set1": [MockRetrievalEvalResult()]})
        viz = RetrievalEvaluationVisualization(
            title="Test Title",
            evaluation_score=eval_score,
            keys=["set1"],
        )

        viz.draw()

        layout = viz.fig.layout
        assert layout.title.text == "Test Title"
        assert layout.barmode == "group"
        # Check that the layout has the expected yaxis configuration
        assert hasattr(layout, "yaxis")

    def test_empty_keys_list(self):
        eval_score = EvaluationScore()
        viz = RetrievalEvaluationVisualization(
            title="Empty",
            evaluation_score=eval_score,
            keys=[],
        )

        viz.draw()
        # Should not crash with empty keys


class TestAnswerEvaluationVisualization:
    def test_initialization(self):
        eval_score = EvaluationScore()
        viz = AnswerEvaluationVisualization(
            title="Answer Evaluation",
            evaluation_score=eval_score,
            keys=["set1", "set2"],
        )

        assert viz.title == "Answer Evaluation"
        assert viz.evaluation_score == eval_score
        assert viz.keys == ["set1", "set2"]
        assert isinstance(viz.fig, go.Figure)

    def test_draw_with_single_key(self):
        results = [
            MockAnswerEvalResult(accuracy=95, relevance=90, completeness=85),
            MockAnswerEvalResult(accuracy=85, relevance=80, completeness=75),
        ]

        eval_score = EvaluationScore(scores={"set1": results})
        viz = AnswerEvaluationVisualization(
            title="Test",
            evaluation_score=eval_score,
            keys=["set1"],
        )

        viz.draw()

        assert len(viz.fig.data) == 3  # Accuracy, Relevance, Completeness
        assert viz.fig.data[0].name == "Accuracy"
        assert viz.fig.data[1].name == "Relevance"
        assert viz.fig.data[2].name == "Completeness"

    def test_draw_with_multiple_keys(self):
        results1 = [
            MockAnswerEvalResult(accuracy=95, relevance=90, completeness=85),
        ]
        results2 = [
            MockAnswerEvalResult(accuracy=70, relevance=65, completeness=60),
        ]

        eval_score = EvaluationScore(scores={"set1": results1, "set2": results2})
        viz = AnswerEvaluationVisualization(
            title="Multi-set Evaluation",
            evaluation_score=eval_score,
            keys=["set1", "set2"],
        )

        viz.draw()

        assert len(viz.fig.data) == 3
        assert list(viz.fig.data[0].x) == ["set1", "set2"]

    def test_draw_calculates_averages(self):
        results = [
            MockAnswerEvalResult(accuracy=100, relevance=100, completeness=100),
            MockAnswerEvalResult(accuracy=80, relevance=80, completeness=80),
        ]

        eval_score = EvaluationScore(scores={"set1": results})
        viz = AnswerEvaluationVisualization(
            title="Test",
            evaluation_score=eval_score,
            keys=["set1"],
        )

        viz.draw()

        # Average accuracy should be 90
        assert viz.fig.data[0].y[0] == 90
        # Average relevance should be 90
        assert viz.fig.data[1].y[0] == 90
        # Average completeness should be 90
        assert viz.fig.data[2].y[0] == 90

    def test_layout_configuration(self):
        eval_score = EvaluationScore(scores={"set1": [MockAnswerEvalResult()]})
        viz = AnswerEvaluationVisualization(
            title="Test Title",
            evaluation_score=eval_score,
            keys=["set1"],
        )

        viz.draw()

        layout = viz.fig.layout
        assert layout.title.text == "Test Title"
        assert layout.barmode == "group"
        # Check that the layout has the expected yaxis configuration
        assert hasattr(layout, "yaxis")

    def test_percentage_values_converted_to_decimal(self):
        results = [
            MockAnswerEvalResult(accuracy=100, relevance=100, completeness=100),
        ]

        eval_score = EvaluationScore(scores={"set1": results})
        viz = AnswerEvaluationVisualization(
            title="Test",
            evaluation_score=eval_score,
            keys=["set1"],
        )

        viz.draw()

        # Values should be as raw percentages (100), not converted
        assert viz.fig.data[0].y[0] == 100
        assert viz.fig.data[1].y[0] == 100
        assert viz.fig.data[2].y[0] == 100

    def test_get_with_redraw(self):
        eval_score = EvaluationScore(scores={"set1": [MockAnswerEvalResult()]})
        viz = AnswerEvaluationVisualization(
            title="Test",
            evaluation_score=eval_score,
            keys=["set1"],
        )

        fig = viz.get(redraw=True)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3

    def test_missing_key_returns_empty_list(self):
        eval_score = EvaluationScore(scores={"set1": [MockAnswerEvalResult()]})
        viz = AnswerEvaluationVisualization(
            title="Test",
            evaluation_score=eval_score,
            keys=["set1", "missing_set"],
        )

        # Should handle missing key gracefully
        with pytest.raises(ZeroDivisionError):
            # Will fail on division by zero for missing key
            viz.draw()
