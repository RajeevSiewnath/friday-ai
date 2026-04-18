import pytest
from unittest.mock import AsyncMock, MagicMock
from friday.core.llm import LLM
from friday.core.vector_db import VectorQueryOutput
from friday.graph.query.nodes.rag_tsne_vis_updater import rag_tsne_vis_updater_factory


def test_rag_tsne_vis_updater_factory_creates_function():
    mock_vis = MagicMock()
    func = rag_tsne_vis_updater_factory("cv-rajeev-siewnath", mock_vis)
    assert callable(func)


@pytest.mark.asyncio
async def test_rag_tsne_vis_updater_sets_highlight_ids(
    make_runtime, llm: LLM, vector_db_documents: list[VectorQueryOutput]
):
    runtime = make_runtime({"llm": llm})

    state = {
        "messages": [{"role": "user", "content": "give me anything"}],
        "rag_data": {"cv-rajeev-siewnath": vector_db_documents},
    }

    mock_vis = MagicMock()
    mock_vis.highlight_ids = None

    mock_llm = AsyncMock()
    mock_llm.embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

    updater = rag_tsne_vis_updater_factory("cv-rajeev-siewnath", mock_vis)
    await updater(state, runtime)

    assert len(mock_vis.highlight_ids) > 0
