import pytest
from friday.core.vector_db import VectorQueryOutput
from friday.query_nodes.nodes.rag_context_injector_factory import (
    rag_context_injector_factory,
)


def test_rag_context_injector_factory_creates_function():
    func = rag_context_injector_factory("cv-rajeev-siewnath")
    assert callable(func)


@pytest.mark.asyncio
async def test_rag_context_injector_formats_with_documents(
    vector_db_documents: list[VectorQueryOutput],
):
    state = {
        "rag_data": {"cv-rajeev-siewnath": vector_db_documents},
        "system_prompt": "",
    }

    node = rag_context_injector_factory("cv-rajeev-siewnath")
    result = node(state)

    assert "system_prompt" in result
    assert "Context:" in result["system_prompt"]
