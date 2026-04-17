import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from friday.visualizations.vector_db_tsne_visualization import (
    VectorDBTSNEVisualization,
)
import plotly.graph_objects as go


class TestVectorDBTSNEVisualization:
    @pytest.fixture
    def mock_collection(self):
        """Create a mock ChromaDB collection."""
        collection = MagicMock()
        collection.get.return_value = {
            "ids": ["doc1", "doc2", "doc3"],
            "embeddings": [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.4, 0.5, 0.6],
            ],
            "metadatas": [
                {"doc_type": "type1"},
                {"doc_type": "type1"},
                {"doc_type": "type2"},
            ],
            "documents": ["Document 1 content", "Document 2 content", "Document 3"],
        }
        return collection

    def test_initialization(self, mock_collection):
        viz = VectorDBTSNEVisualization(
            title="TSNE Visualization",
            collection=mock_collection,
            is_3d=False,
            perplexity=5,
            show_question=True,
        )

        assert viz.title == "TSNE Visualization"
        assert viz.collection == mock_collection
        assert viz.is_3d is False
        assert viz.perplexity == 5
        assert viz.show_question is True
        assert viz.highlight_ids == []
        assert viz.question is None

    def test_initialization_defaults(self, mock_collection):
        viz = VectorDBTSNEVisualization(
            title="Test",
            collection=mock_collection,
        )

        assert viz.is_3d is False
        assert viz.perplexity == 5
        assert viz.show_question is True

    def test_highlight_ids_property(self, mock_collection):
        viz = VectorDBTSNEVisualization(
            title="Test",
            collection=mock_collection,
        )

        viz.highlight_ids = ["doc1", "doc2"]
        assert viz.highlight_ids == ["doc1", "doc2"]

    def test_question_property(self, mock_collection):
        viz = VectorDBTSNEVisualization(
            title="Test",
            collection=mock_collection,
        )

        question_data = ("What is AI?", [0.1, 0.2, 0.3, 0.4])
        viz.question = question_data
        assert viz.question == question_data

    @patch("friday.visualizations.vector_db_tsne_visualization.TSNE")
    def test_draw_2d(self, mock_tsne, mock_collection):
        # Mock TSNE to return predictable 2D coordinates
        mock_tsne_instance = MagicMock()
        mock_tsne_instance.fit_transform.return_value = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
        )
        mock_tsne.return_value = mock_tsne_instance

        viz = VectorDBTSNEVisualization(
            title="Test",
            collection=mock_collection,
            is_3d=False,
        )

        viz.draw()

        assert len(viz.fig.data) == 1
        assert isinstance(viz.fig.data[0], go.Scatter)

    @patch("friday.visualizations.vector_db_tsne_visualization.TSNE")
    def test_draw_3d(self, mock_tsne, mock_collection):
        # Mock TSNE to return predictable 3D coordinates
        mock_tsne_instance = MagicMock()
        mock_tsne_instance.fit_transform.return_value = np.array(
            [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
        )
        mock_tsne.return_value = mock_tsne_instance

        viz = VectorDBTSNEVisualization(
            title="Test",
            collection=mock_collection,
            is_3d=True,
        )

        viz.draw()

        assert len(viz.fig.data) == 1
        assert isinstance(viz.fig.data[0], go.Scatter3d)

    @patch("friday.visualizations.vector_db_tsne_visualization.TSNE")
    def test_draw_with_question(self, mock_tsne, mock_collection):
        mock_tsne_instance = MagicMock()
        mock_tsne_instance.fit_transform.return_value = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
        )
        mock_tsne.return_value = mock_tsne_instance

        viz = VectorDBTSNEVisualization(
            title="Test",
            collection=mock_collection,
            is_3d=False,
            show_question=True,
        )

        viz.question = ("What is AI?", [0.1, 0.2, 0.3, 0.4])
        viz.draw()

        # Check that the trace has points for docs + question
        assert len(viz.fig.data[0].x) == 4

    @patch("friday.visualizations.vector_db_tsne_visualization.TSNE")
    def test_draw_with_question_hidden(self, mock_tsne, mock_collection):
        mock_tsne_instance = MagicMock()
        mock_tsne_instance.fit_transform.return_value = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
        )
        mock_tsne.return_value = mock_tsne_instance

        viz = VectorDBTSNEVisualization(
            title="Test",
            collection=mock_collection,
            show_question=False,
        )

        viz.question = ("What is AI?", [0.1, 0.2, 0.3, 0.4])
        viz.draw()

        # Should not include question point
        assert len(viz.fig.data[0].x) == 3

    @patch("friday.visualizations.vector_db_tsne_visualization.TSNE")
    def test_highlight_ids_change_symbols(self, mock_tsne, mock_collection):
        mock_tsne_instance = MagicMock()
        mock_tsne_instance.fit_transform.return_value = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
        )
        mock_tsne.return_value = mock_tsne_instance

        viz = VectorDBTSNEVisualization(
            title="Test",
            collection=mock_collection,
        )

        viz.highlight_ids = ["doc1"]
        viz.draw()

        symbols = viz.fig.data[0].marker.symbol
        # doc1 should be square, others should be circle
        assert symbols[0] == "square"
        assert symbols[1] == "circle"
        assert symbols[2] == "circle"

    @patch("friday.visualizations.vector_db_tsne_visualization.TSNE")
    def test_tsne_parameters(self, mock_tsne, mock_collection):
        mock_tsne_instance = MagicMock()
        mock_tsne_instance.fit_transform.return_value = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
        )
        mock_tsne.return_value = mock_tsne_instance

        viz = VectorDBTSNEVisualization(
            title="Test",
            collection=mock_collection,
            is_3d=False,
            perplexity=10,
        )

        viz.draw()

        # Verify TSNE was called with correct parameters
        mock_tsne.assert_called_with(
            n_components=2,
            random_state=42,
            perplexity=10,
        )

    @patch("friday.visualizations.vector_db_tsne_visualization.TSNE")
    def test_unique_doc_types_colored(self, mock_tsne, mock_collection):
        mock_tsne_instance = MagicMock()
        mock_tsne_instance.fit_transform.return_value = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
        )
        mock_tsne.return_value = mock_tsne_instance

        viz = VectorDBTSNEVisualization(
            title="Test",
            collection=mock_collection,
        )

        viz.draw()

        # Should have colors assigned to unique doc_types
        colors = viz.fig.data[0].marker.color
        assert len(colors) == 3

    @patch("friday.visualizations.vector_db_tsne_visualization.TSNE")
    def test_collection_get_called_correctly(self, mock_tsne, mock_collection):
        mock_tsne_instance = MagicMock()
        mock_tsne_instance.fit_transform.return_value = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
        )
        mock_tsne.return_value = mock_tsne_instance

        viz = VectorDBTSNEVisualization(
            title="Test",
            collection=mock_collection,
        )

        viz.draw()

        # Verify collection.get was called with correct include parameters
        mock_collection.get.assert_called_with(
            include=["embeddings", "metadatas", "documents"]
        )

    @patch("friday.visualizations.vector_db_tsne_visualization.TSNE")
    def test_title_includes_dimensionality(self, mock_tsne, mock_collection):
        # Test 2D
        mock_tsne_instance_2d = MagicMock()
        mock_tsne_instance_2d.fit_transform.return_value = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
        )

        # Test 3D
        mock_tsne_instance_3d = MagicMock()
        mock_tsne_instance_3d.fit_transform.return_value = np.array(
            [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
        )

        viz_2d = VectorDBTSNEVisualization(
            title="My Visualization",
            collection=mock_collection,
            is_3d=False,
        )

        mock_tsne.return_value = mock_tsne_instance_2d
        viz_2d.draw()

        viz_3d = VectorDBTSNEVisualization(
            title="My Visualization",
            collection=mock_collection,
            is_3d=True,
        )

        mock_tsne.return_value = mock_tsne_instance_3d
        viz_3d.draw()

        assert "(2D)" in viz_2d.fig.layout.title.text
        assert "(3D)" in viz_3d.fig.layout.title.text

    @patch("friday.visualizations.vector_db_tsne_visualization.TSNE")
    def test_hover_text_includes_document_content(self, mock_tsne, mock_collection):
        mock_tsne_instance = MagicMock()
        mock_tsne_instance.fit_transform.return_value = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
        )
        mock_tsne.return_value = mock_tsne_instance

        viz = VectorDBTSNEVisualization(
            title="Test",
            collection=mock_collection,
        )

        viz.draw()

        texts = viz.fig.data[0].text
        assert "Document 1 content" in texts[0]
        assert "Document 2 content" in texts[1]

    def test_get_method(self, mock_collection):
        with patch(
            "friday.visualizations.vector_db_tsne_visualization.TSNE"
        ) as mock_tsne:
            mock_tsne_instance = MagicMock()
            mock_tsne_instance.fit_transform.return_value = np.array(
                [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
            )
            mock_tsne.return_value = mock_tsne_instance

            viz = VectorDBTSNEVisualization(
                title="Test",
                collection=mock_collection,
            )

            fig = viz.get(redraw=True)

            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0
