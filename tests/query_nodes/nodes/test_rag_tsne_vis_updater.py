import pytest
from unittest.mock import AsyncMock, MagicMock
from friday.query_nodes.nodes.rag_tsne_vis_updater import rag_tsne_vis_updater_factory


def test_rag_tsne_vis_updater_factory_creates_function():
    mock_vis = MagicMock()
    func = rag_tsne_vis_updater_factory("documents", mock_vis)
    assert callable(func)


@pytest.mark.asyncio
async def test_rag_tsne_vis_updater_sets_highlight_ids():
    mock_context1 = MagicMock()
    mock_context1.id = "id_1"

    mock_context2 = MagicMock()
    mock_context2.id = "id_2"

    state = {
        "messages": [{"content": "test query"}],
        "rag_data": {"documents": [mock_context1, mock_context2]},
    }

    mock_vis = MagicMock()
    mock_vis.highlight_ids = None

    mock_llm = AsyncMock()
    mock_llm.embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

    mock_runtime = MagicMock()
    mock_runtime.context.llm = mock_llm

    updater = rag_tsne_vis_updater_factory("documents", mock_vis)
    await updater(state, mock_runtime)

    assert mock_vis.highlight_ids == ["id_1", "id_2"]


@pytest.mark.asyncio
async def test_rag_tsne_vis_updater_sets_question():
    mock_context = MagicMock()
    mock_context.id = "id_1"

    state = {
        "messages": [{"content": "What is Python?"}],
        "rag_data": {"documents": [mock_context]},
    }

    mock_vis = MagicMock()
    mock_embedding = [0.1, 0.2, 0.3]

    mock_llm = AsyncMock()
    mock_llm.embedding = AsyncMock(return_value=mock_embedding)

    mock_runtime = MagicMock()
    mock_runtime.context.llm = mock_llm

    updater = rag_tsne_vis_updater_factory("documents", mock_vis)
    await updater(state, mock_runtime)

    assert mock_vis.question[0] == "What is Python?"
    assert mock_vis.question[1] == mock_embedding


@pytest.mark.asyncio
async def test_rag_tsne_vis_updater_calls_embedding():
    mock_context = MagicMock()
    mock_context.id = "id_1"

    state = {
        "messages": [{"content": "test"}],
        "rag_data": {"documents": [mock_context]},
    }

    mock_vis = MagicMock()

    mock_llm = AsyncMock()
    mock_llm.embedding = AsyncMock(return_value=[0.5, 0.6])

    mock_runtime = MagicMock()
    mock_runtime.context.llm = mock_llm

    updater = rag_tsne_vis_updater_factory("documents", mock_vis)
    await updater(state, mock_runtime)

    mock_llm.embedding.assert_called_once_with("test")


@pytest.mark.asyncio
async def test_rag_tsne_vis_updater_returns_empty_dict():
    mock_context = MagicMock()
    mock_context.id = "id_1"

    state = {
        "messages": [{"content": "query"}],
        "rag_data": {"documents": [mock_context]},
    }

    mock_vis = MagicMock()

    mock_llm = AsyncMock()
    mock_llm.embedding = AsyncMock(return_value=[0.1])

    mock_runtime = MagicMock()
    mock_runtime.context.llm = mock_llm

    updater = rag_tsne_vis_updater_factory("documents", mock_vis)
    result = await updater(state, mock_runtime)

    assert result == {}
