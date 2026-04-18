import pytest
from unittest.mock import AsyncMock, MagicMock
from friday.core.llm import LLM
from friday.graph.query.nodes.llm_invoke import llm_invoke


@pytest.mark.asyncio
async def test_llm_invoke_returns_messages(make_runtime, llm: LLM):
    runtime = make_runtime({"llm": llm})
    state = {"messages": [{"role": "user", "content": "hello"}]}

    result = await llm_invoke(state, runtime)

    assert "messages" in result
    assert isinstance(result["messages"], list)
    assert len(result["messages"]) == 1
