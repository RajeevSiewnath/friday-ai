import inspect
import json
from typing import Any, Callable, Self
from friday.core.mcp_tool_box import MCPToolBox
from friday.core.tool_definition import ToolDefinition
from friday.utils.function_definition_creator import function_definition_creator


class ToolShed:
    def __init__(self, *tools: Callable):
        self.tools: list[ToolDefinition] = []
        self.add(*tools)

    def add(self, *tools: Callable) -> Self:
        tool_defs = []
        for t in tools:
            definition = function_definition_creator(t)
            tool_defs.append(
                ToolDefinition(
                    name=definition["name"],
                    definition=definition,
                    callable=t,
                )
            )
        self.tools.extend(tool_defs)
        return self

    async def mcp_tool_box(self, toolbox: MCPToolBox) -> Self:
        for tool_def in await toolbox.list_tools():
            self.tools.extend(tool_def)
        return self

    async def mcp_tool_box_isolated(self, toolbox: MCPToolBox) -> Self:
        for tool_def in await toolbox.list_tools_isolated():
            self.tools.extend(tool_def)
        return self

    def remove(self, *tools: ToolDefinition) -> Self:
        self.tools = [t for t in self.tools if t not in tools]
        return self

    async def call(self, name: str, args: str) -> Any:
        tool = next((t for t in self.tools if t.name == name), None)
        if tool:
            result = tool.callable(**json.loads(args))
            if inspect.isawaitable(result):
                return await result
            else:
                return result
        else:
            raise f"tool not defined: '{name}'"

    @property
    def definitions(self) -> list[Any]:
        return [t.definition for t in self.tools]
