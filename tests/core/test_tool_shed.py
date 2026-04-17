import pytest
from friday.core.tool_shed import ToolShed, ToolDefinition


class TestToolShed:
    @pytest.fixture
    def simple_tool(self):
        def test_tool(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        return test_tool

    @pytest.fixture
    def async_tool(self):
        async def async_test_tool(x: int) -> int:
            """Square a number asynchronously."""
            return x * x

        return async_test_tool

    def test_initialization_empty(self):
        shed = ToolShed()
        assert len(shed.tools) == 0

    def test_initialization_with_tools(self, simple_tool, async_tool):
        shed = ToolShed(simple_tool, async_tool)
        assert len(shed.tools) == 2

    def test_add_single_tool(self, simple_tool):
        shed = ToolShed()
        result = shed.add(simple_tool)

        assert len(shed.tools) == 1
        assert shed.tools[0].name == "test_tool"
        assert result == shed  # Check it returns self for chaining

    def test_add_multiple_tools(self, simple_tool, async_tool):
        shed = ToolShed()
        shed.add(simple_tool, async_tool)

        assert len(shed.tools) == 2
        assert shed.tools[0].name == "test_tool"
        assert shed.tools[1].name == "async_test_tool"

    def test_add_returns_self_for_chaining(self, simple_tool, async_tool):
        shed = ToolShed()
        result = shed.add(simple_tool).add(async_tool)

        assert result == shed
        assert len(shed.tools) == 2

    def test_remove_tool(self, simple_tool, async_tool):
        shed = ToolShed(simple_tool, async_tool)
        tool_to_remove = shed.tools[0]

        result = shed.remove(tool_to_remove)

        assert len(shed.tools) == 1
        assert shed.tools[0].name == "async_test_tool"
        assert result == shed  # Check it returns self

    def test_remove_multiple_tools(self, simple_tool, async_tool):
        shed = ToolShed(simple_tool, async_tool)
        tools_to_remove = [shed.tools[0], shed.tools[1]]

        shed.remove(*tools_to_remove)

        assert len(shed.tools) == 0

    def test_remove_nonexistent_tool(self, simple_tool):
        shed = ToolShed(simple_tool)
        fake_tool = ToolDefinition(name="fake", callable=lambda: None, definition={})

        shed.remove(fake_tool)
        assert len(shed.tools) == 1  # Original tool still there

    @pytest.mark.asyncio
    async def test_call_sync_tool(self, simple_tool):
        shed = ToolShed(simple_tool)
        result = await shed.call("test_tool", '{"x": 2, "y": 3}')

        assert result == 5

    @pytest.mark.asyncio
    async def test_call_async_tool(self, async_tool):
        shed = ToolShed(async_tool)
        result = await shed.call("async_test_tool", '{"x": 5}')

        assert result == 25

    @pytest.mark.asyncio
    async def test_call_nonexistent_tool(self, simple_tool):
        shed = ToolShed(simple_tool)

        with pytest.raises(TypeError):
            await shed.call("nonexistent", '{}')

    @pytest.mark.asyncio
    async def test_call_with_invalid_json_args(self, simple_tool):
        shed = ToolShed(simple_tool)

        with pytest.raises(ValueError):
            await shed.call("test_tool", "invalid json")

    def test_definitions_property(self, simple_tool, async_tool):
        shed = ToolShed(simple_tool, async_tool)
        definitions = shed.definitions

        assert len(definitions) == 2
        assert all(isinstance(d, dict) for d in definitions)
        assert "name" in definitions[0]
        assert "parameters" in definitions[0]

    def test_definitions_empty(self):
        shed = ToolShed()
        definitions = shed.definitions

        assert definitions == []

    @pytest.mark.asyncio
    async def test_call_tool_with_complex_args(self):
        def process_data(data: dict, count: int) -> dict:
            """Process data dictionary."""
            return {"result": data, "count": count}

        shed = ToolShed(process_data)
        result = await shed.call("process_data", '{"data": {"key": "value"}, "count": 42}')

        assert result == {"result": {"key": "value"}, "count": 42}

    def test_tool_definition_structure(self, simple_tool):
        shed = ToolShed(simple_tool)
        tool_def = shed.tools[0]

        assert isinstance(tool_def, ToolDefinition)
        assert tool_def.name == "test_tool"
        assert tool_def.callable == simple_tool
        assert isinstance(tool_def.definition, dict)
