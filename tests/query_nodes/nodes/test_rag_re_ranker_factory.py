import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from friday.query_nodes.nodes.rag_re_ranker_factory import rag_re_ranker_factory


def test_rag_re_ranker_factory_creates_function():
    func = rag_re_ranker_factory("documents")
    assert callable(func)


@pytest.mark.asyncio
async def test_rag_re_ranker_calls_llm_with_ranking_prompt():
    mock_result1 = MagicMock()
    mock_result1.document = "Document 1"

    mock_result2 = MagicMock()
    mock_result2.document = "Document 2"

    state = {
        "messages": [{"content": "What is your experience?"}],
        "rag_data": {"documents": [mock_result1, mock_result2]},
    }

    mock_response = MagicMock()
    mock_response.output_parsed.order = [2, 1]

    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    mock_runtime = MagicMock()
    mock_runtime.context.llm = mock_llm

    with patch("friday.query_nodes.nodes.rag_re_ranker_factory.RagReducerReplaceAction"):
        re_ranker = rag_re_ranker_factory("documents")
        await re_ranker(state, mock_runtime)

    mock_llm.invoke.assert_called_once()
    call_args = mock_llm.invoke.call_args
    assert call_args.kwargs["response_format"] is not None


@pytest.mark.asyncio
async def test_rag_re_ranker_reorders_documents():
    mock_result1 = MagicMock()
    mock_result1.document = "Document 1"

    mock_result2 = MagicMock()
    mock_result2.document = "Document 2"

    state = {
        "messages": [{"content": "test question"}],
        "rag_data": {"documents": [mock_result1, mock_result2]},
    }

    mock_response = MagicMock()
    mock_response.output_parsed.order = [2, 1]

    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    mock_runtime = MagicMock()
    mock_runtime.context.llm = mock_llm

    with patch(
        "friday.query_nodes.nodes.rag_re_ranker_factory.RagReducerReplaceAction"
    ) as mock_action:
        re_ranker = rag_re_ranker_factory("documents")
        result = await re_ranker(state, mock_runtime)

    assert "rag_data" in result
    assert "documents" in result["rag_data"]


@pytest.mark.asyncio
async def test_rag_re_ranker_includes_question_in_prompt():
    mock_result = MagicMock()
    mock_result.document = "Content"

    state = {
        "messages": [{"content": "specific question about something"}],
        "rag_data": {"documents": [mock_result]},
    }

    mock_response = MagicMock()
    mock_response.output_parsed.order = [1]

    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = mock_response

    mock_runtime = MagicMock()
    mock_runtime.context.llm = mock_llm

    with patch("friday.query_nodes.nodes.rag_re_ranker_factory.RagReducerReplaceAction"):
        re_ranker = rag_re_ranker_factory("documents")
        await re_ranker(state, mock_runtime)

    call_args = mock_llm.invoke.call_args
    messages = call_args.kwargs.get("input", [])
    prompt_text = str(messages)
    assert "specific question about something" in prompt_text
