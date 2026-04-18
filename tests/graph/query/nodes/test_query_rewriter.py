import pytest
from unittest.mock import AsyncMock, MagicMock
from friday.core.llm import LLM
from friday.graph.query.nodes.query_rewriter import query_rewriter


@pytest.mark.asyncio
async def test_query_rewriter_calls_llm(make_runtime, llm: LLM):
    runtime = make_runtime(
        {
            "llm": llm,
            "user_context": "CV",
            "user": "John Doe",
        }
    )

    state = {
        "messages": [
            {"role": "user", "content": "Tell me more about Python"},
        ]
    }

    result = await query_rewriter(state, runtime)
    assert "messages" in result


# @pytest.mark.asyncio
# async def test_query_rewriter_includes_context_in_prompt():
#     captured_prompt = None

#     async def capture_invoke(msg):
#         nonlocal captured_prompt
#         captured_prompt = msg
#         return MagicMock(output_text="result")

#     mock_llm = AsyncMock()
#     mock_llm.invoke.side_effect = capture_invoke

#     mock_context = MagicMock()
#     mock_context.llm = mock_llm
#     mock_context.user_context = "My CV"
#     mock_context.user = "John"

#     mock_runtime = MagicMock()
#     mock_runtime.context = mock_context

#     state = {
#         "messages": [
#             {"role": "user", "content": "old"},
#             {"role": "user", "content": "current question"},
#         ]
#     }

#     await query_rewriter(state, mock_runtime)

#     assert captured_prompt["role"] == "system"
#     assert "My CV" in captured_prompt["content"]
#     assert "current question" in captured_prompt["content"]
