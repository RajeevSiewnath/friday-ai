import pytest
from unittest.mock import MagicMock
from friday.query_nodes.nodes.rag_context_injector_factory import (
    rag_context_injector_factory,
)


def test_rag_context_injector_factory_creates_function():
    func = rag_context_injector_factory("documents")
    assert callable(func)


def test_rag_context_injector_formats_with_documents():
    mock_result1 = MagicMock()
    mock_result1.document = "Document 1 content"

    mock_result2 = MagicMock()
    mock_result2.document = "Document 2 content"

    state = {
        "rag_data": {"documents": [mock_result1, mock_result2]},
        "system_prompt": "",
    }

    injector = rag_context_injector_factory("documents")
    result = injector(state)

    assert "system_prompt" in result
    assert "Context:" in result["system_prompt"]
    assert "Document 1 content" in result["system_prompt"]
    assert "Document 2 content" in result["system_prompt"]


def test_rag_context_injector_uses_custom_label():
    mock_result = MagicMock()
    mock_result.document = "Content"

    state = {
        "rag_data": {"data": [mock_result]},
        "system_prompt": "",
    }

    injector = rag_context_injector_factory("data", context_label="Retrieved Info")
    result = injector(state)

    assert "Retrieved Info:" in result["system_prompt"]


def test_rag_context_injector_empty_results():
    state = {
        "rag_data": {"documents": []},
        "system_prompt": "",
    }

    injector = rag_context_injector_factory("documents")
    result = injector(state)

    assert "Context:" in result["system_prompt"]
    assert result["system_prompt"] == "Context:"


def test_rag_context_injector_formats_documents_with_newlines():
    mock_result1 = MagicMock()
    mock_result1.document = "First doc"

    mock_result2 = MagicMock()
    mock_result2.document = "Second doc"

    state = {
        "rag_data": {"docs": [mock_result1, mock_result2]},
        "system_prompt": "",
    }

    injector = rag_context_injector_factory("docs")
    result = injector(state)

    assert "Context:\nFirst doc\nSecond doc" in result["system_prompt"]
