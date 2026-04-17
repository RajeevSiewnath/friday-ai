import pytest
from unittest.mock import MagicMock
from friday.visualizations.abstract_visualization import AbstractVisualization
import plotly.graph_objects as go


class ConcreteVisualization(AbstractVisualization):
    """Concrete implementation for testing abstract class."""

    def draw(self):
        self.fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))


class TestAbstractVisualization:
    def test_initialization(self):
        viz = ConcreteVisualization(title="Test Visualization")

        assert viz.title == "Test Visualization"
        assert isinstance(viz.fig, go.Figure)

    def test_draw_called_on_get(self):
        viz = ConcreteVisualization(title="Test")
        fig = viz.get(redraw=True)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_get_without_redraw(self):
        viz = ConcreteVisualization(title="Test")
        viz.draw()
        initial_trace_count = len(viz.fig.data)

        fig = viz.get(redraw=False)

        assert len(fig.data) == initial_trace_count

    def test_get_with_redraw_clears_data(self):
        viz = ConcreteVisualization(title="Test")
        viz.draw()
        assert len(viz.fig.data) > 0

        fig = viz.get(redraw=True)

        # After redraw, should have redrawn traces
        assert isinstance(fig, go.Figure)

    def test_get_returns_figure(self):
        viz = ConcreteVisualization(title="Test")
        result = viz.get()

        assert isinstance(result, go.Figure)
        assert result is viz.fig

    def test_must_implement_draw_method(self):
        # Test that the draw method is marked as abstract
        assert hasattr(AbstractVisualization.draw, "__isabstractmethod__")

    def test_multiple_redraws(self):
        viz = ConcreteVisualization(title="Test")

        fig1 = viz.get(redraw=True)
        fig2 = viz.get(redraw=True)

        assert fig1 is fig2  # Same figure object
        assert isinstance(fig1, go.Figure)

    def test_title_with_special_characters(self):
        special_title = "Test: <Special> & Characters!"
        viz = ConcreteVisualization(title=special_title)

        assert viz.title == special_title

    def test_empty_title(self):
        viz = ConcreteVisualization(title="")

        assert viz.title == ""
        fig = viz.get()
        assert isinstance(fig, go.Figure)
