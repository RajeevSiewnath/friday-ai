import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from friday.query_nodes.nodes.llm_stream import llm_stream


@pytest.mark.asyncio
async def test_llm_stream_collects_events():
    events = [
        {"role": "assistant", "content": "hello"},
        {"role": "assistant", "content": " world"},
    ]

    async def mock_stream(*args, **kwargs):
        for event in events:
            yield event

    mock_llm = MagicMock()
    mock_llm.stream = mock_stream

    mock_runtime = MagicMock()
    mock_runtime.context.llm = mock_llm

    state = {"messages": [{"role": "user", "content": "test"}]}

    with patch("langgraph.config.get_stream_writer") as mock_writer:
        mock_writer.return_value = MagicMock()
        result = await llm_stream(state, mock_runtime)

    assert "messages" in result
    assert len(result["messages"]) == 2
    assert result["messages"] == events


@pytest.mark.asyncio
async def test_llm_stream_writes_events():
    events = [{"role": "assistant", "content": "test"}]

    async def mock_stream(*args, **kwargs):
        for event in events:
            yield event

    mock_llm = MagicMock()
    mock_llm.stream = mock_stream

    mock_runtime = MagicMock()
    mock_runtime.context.llm = mock_llm

    state = {"messages": []}

    with patch("langgraph.config.get_stream_writer") as mock_writer_fn:
        mock_writer = MagicMock()
        mock_writer_fn.return_value = mock_writer
        await llm_stream(state, mock_runtime)
        mock_writer.assert_called()


@pytest.mark.asyncio
async def test_llm_stream_empty_events():
    async def mock_stream(*args, **kwargs):
        return
        yield

    mock_llm = MagicMock()
    mock_llm.stream = mock_stream

    mock_runtime = MagicMock()
    mock_runtime.context.llm = mock_llm

    state = {"messages": []}

    with patch("langgraph.config.get_stream_writer") as mock_writer:
        mock_writer.return_value = MagicMock()
        result = await llm_stream(state, mock_runtime)

    assert result["messages"] == []
