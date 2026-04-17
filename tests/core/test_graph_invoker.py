import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import TypedDict
from friday.core.graph_invoker import GraphInvoker


class MockState(TypedDict):
    value: str
    counter: int


def add_to_counter(old, new):
    return (old or 0) + new


class TestGraphInvoker:
    @pytest.fixture
    def mock_graph(self):
        graph_mock = AsyncMock()
        graph_mock.builder = MagicMock()
        graph_mock.builder.state_schema = MockState
        return graph_mock

    def test_initialization(self, mock_graph):
        invoker = GraphInvoker(mock_graph)
        assert invoker.graph == mock_graph
        assert invoker.on_effect is None
        assert invoker.on_node_change is None
        assert invoker.context is None
        assert invoker.update_state_with_effect is True

    def test_initialization_with_callbacks(self, mock_graph):
        effect_callback = lambda x: None
        node_callback = lambda x: None

        invoker = GraphInvoker(
            mock_graph,
            on_effect=effect_callback,
            on_node_change=node_callback,
            context={"test": "context"},
            update_state_with_effect=False,
        )

        assert invoker.on_effect == effect_callback
        assert invoker.on_node_change == node_callback
        assert invoker.context == {"test": "context"}
        assert invoker.update_state_with_effect is False

    @pytest.mark.asyncio
    async def test_stream_with_values_chunk(self, mock_graph):
        effect_mock = AsyncMock()
        node_change_mock = AsyncMock()
        invoker = GraphInvoker(mock_graph, on_effect=effect_mock, on_node_change=node_change_mock)

        test_state = {"value": "initial", "counter": 0}
        expected_output = {"value": "processed", "counter": 1}

        chunks = [{"type": "values", "data": expected_output}]

        async def mock_astream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_graph.astream = mock_astream

        results = []
        async for result in invoker.stream(test_state):
            results.append(result)

        assert len(results) == 1
        assert results[0] == expected_output

    @pytest.mark.asyncio
    async def test_stream_with_updates_chunk(self, mock_graph):
        node_change_mock = MagicMock()
        invoker = GraphInvoker(mock_graph, on_node_change=node_change_mock)

        test_state = {"value": "initial", "counter": 0}

        chunks = [
            {
                "type": "updates",
                "data": {"counter": 1},
            }
        ]

        async def mock_astream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_graph.astream = mock_astream

        async for _ in invoker.stream(test_state):
            pass

        node_change_mock.assert_called_once_with("counter")

    @pytest.mark.asyncio
    async def test_stream_with_custom_chunk(self, mock_graph):
        effect_mock = MagicMock()
        invoker = GraphInvoker(
            mock_graph, on_effect=effect_mock, update_state_with_effect=True
        )

        test_state = {"value": "initial", "counter": 0}

        chunks = [{"type": "custom", "data": {"counter": 1}}]

        async def mock_astream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_graph.astream = mock_astream

        results = []
        async for result in invoker.stream(test_state):
            results.append(result)

        effect_mock.assert_called_once_with({"counter": 1})
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_invoke_returns_final_state(self, mock_graph):
        invoker = GraphInvoker(mock_graph)

        test_state = {"value": "initial", "counter": 0}
        expected_final = {"value": "final", "counter": 3}

        chunks = [
            {"type": "values", "data": {"value": "processing", "counter": 1}},
            {"type": "values", "data": {"value": "processing", "counter": 2}},
            {"type": "values", "data": expected_final},
        ]

        async def mock_astream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_graph.astream = mock_astream

        result = await invoker.invoke(test_state)
        assert result == expected_final

    @pytest.mark.asyncio
    async def test_stream_with_context(self, mock_graph):
        context = {"user_id": "123"}
        invoker = GraphInvoker(mock_graph, context=context)

        test_state = {"value": "test", "counter": 0}

        async def mock_astream(*args, **kwargs):
            assert kwargs.get("context") == context
            yield {"type": "values", "data": test_state}

        mock_graph.astream = mock_astream

        results = []
        async for result in invoker.stream(test_state):
            results.append(result)

        assert len(results) == 1
        assert results[0] == test_state
