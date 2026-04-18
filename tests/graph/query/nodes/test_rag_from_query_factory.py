import pytest
from friday.core.llm import LLM
from friday.core.vector_db import VectorDB
from friday.graph.query.nodes.rag_from_query_factory import rag_from_query_factory


def test_rag_from_query_factory_creates_function():
    func = rag_from_query_factory("cv-rajeev-siewnath")
    assert callable(func)


@pytest.mark.asyncio
async def test_rag_from_query_uses_collection_key(
    make_runtime, llm: LLM, vector_db: VectorDB
):
    runtime = make_runtime({"vector_db": vector_db, "llm": llm})

    state = {
        "messages": [{"role": "user", "content": "give me anything"}],
        "rag_data": {},
    }

    rag_from_query = rag_from_query_factory("cv-rajeev-siewnath")
    result = await rag_from_query(state, runtime)

    assert "rag_data" in result
