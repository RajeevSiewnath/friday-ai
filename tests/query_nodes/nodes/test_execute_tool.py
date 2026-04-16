import pytest
from unittest.mock import AsyncMock, MagicMock
from friday.core.llm import LLM
from friday.query_nodes.nodes.execute_tool import execute_tool


@pytest.mark.asyncio
async def test_execute_tool_returns_function_output(make_runtime, llm: LLM):
    runtime = make_runtime({"llm": llm})
    state = {
        "messages": [
            {
                "name": "send_contact_request",
                "arguments": {"message": "hi"},
                "call_id": "call_123",
            }
        ]
    }

    result = await execute_tool(state, runtime)

    assert "messages" in result
    assert result["messages"][0]["type"] == "function_call_output"
    assert result["messages"][0]["output"] == "True"
    assert result["messages"][0]["call_id"] == "call_123"
