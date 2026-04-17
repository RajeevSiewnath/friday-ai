import pytest
from friday.core.llm import LLM
from friday.core.vector_db import VectorQueryOutput
from friday.query_nodes.nodes.rag_re_ranker_factory import rag_re_ranker_factory


def test_rag_re_ranker_factory_creates_function():
    func = rag_re_ranker_factory("cv-rajeev-siewnath")
    assert callable(func)


@pytest.mark.asyncio
async def test_rag_re_ranker_calls_llm_with_ranking_prompt(
    make_runtime, llm: LLM, vector_db_documents: list[VectorQueryOutput]
):
    runtime = make_runtime({"llm": llm})
    state = {
        "messages": [{"content": "What is your experience?"}],
        "rag_data": {"cv-rajeev-siewnath": vector_db_documents},
    }

    re_ranker = rag_re_ranker_factory("cv-rajeev-siewnath")
    await re_ranker(state, runtime)
