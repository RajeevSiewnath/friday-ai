import pytest
from unittest.mock import MagicMock, patch
from friday.core.llm import LLM
from friday.graph.query.nodes.llm_stream import llm_stream


@pytest.mark.asyncio
async def test_llm_stream_collects_events(make_runtime, llm: LLM):
    runtime = make_runtime({"llm": llm})

    state = {"messages": [{"role": "user", "content": "test"}]}

    with patch("friday.graph.query.nodes.llm_stream.get_stream_writer") as mock_writer:
        mock_writer.return_value = MagicMock()
        result = await llm_stream(state, runtime)

    assert "messages" in result
    assert len(result["messages"]) > 0
