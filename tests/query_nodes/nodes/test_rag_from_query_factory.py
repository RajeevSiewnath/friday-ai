import pytest
from unittest.mock import MagicMock
from friday.query_nodes.nodes.rag_from_query_factory import rag_from_query_factory


def test_rag_from_query_factory_creates_function():
    func = rag_from_query_factory("documents")
    assert callable(func)


def test_rag_from_query_uses_collection_key():
    mock_query_result = MagicMock()
    mock_collection = MagicMock()
    mock_collection.query.return_value = mock_query_result

    mock_vector_db = {"documents": mock_collection}

    mock_runtime = MagicMock()
    mock_runtime.context.vector_db = mock_vector_db

    state = {"messages": [{"content": "test query"}]}

    rag_from_query = rag_from_query_factory("documents")
    result = rag_from_query(state, mock_runtime)

    assert "rag_data" in result
    mock_collection.query.assert_called_once_with("test query")


def test_rag_from_query_uses_custom_state_key():
    mock_query_result = MagicMock()
    mock_collection = MagicMock()
    mock_collection.query.return_value = mock_query_result

    mock_vector_db = {"documents": mock_collection}

    mock_runtime = MagicMock()
    mock_runtime.context.vector_db = mock_vector_db

    state = {"messages": [{"content": "test"}]}

    rag_from_query = rag_from_query_factory("documents", state_key="custom_key")
    result = rag_from_query(state, mock_runtime)

    assert "custom_key" in result["rag_data"]
    assert result["rag_data"]["custom_key"] == mock_query_result


def test_rag_from_query_defaults_state_key_to_collection_key():
    mock_query_result = MagicMock()
    mock_collection = MagicMock()
    mock_collection.query.return_value = mock_query_result

    mock_vector_db = {"docs": mock_collection}

    mock_runtime = MagicMock()
    mock_runtime.context.vector_db = mock_vector_db

    state = {"messages": [{"content": "query"}]}

    rag_from_query = rag_from_query_factory("docs")
    result = rag_from_query(state, mock_runtime)

    assert "docs" in result["rag_data"]
