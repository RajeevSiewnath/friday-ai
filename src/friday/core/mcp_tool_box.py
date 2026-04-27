from typing import Self
from mcp.types import Tool
from agents.mcp import MCPServerStdio, MCPServerStdioParams

from friday.core.tool_definition import ToolDefinition


class MCPToolBox:
    def __init__(self, *stdio_params: MCPServerStdioParams):
        self.stdio_params: list[MCPServerStdioParams] = stdio_params
        self.mcp_servers: list[MCPServerStdio] = []

    async def open(self) -> Self:
        for params in self.stdio_params:
            mcp_server = MCPServerStdio(params=params)
            await mcp_server.connect()
            self.mcp_servers.append(mcp_server)
        return self

    async def close(self) -> Self:
        for server in self.mcp_servers:
            await server.cleanup()
        self.mcp_servers = []
        return self

    def add(self, *stdio_params: MCPServerStdioParams) -> Self:
        self.stdio_params.extend(stdio_params)
        return self

    def remove(self, *stdio_params: MCPServerStdioParams) -> Self:
        self.stdio_params = [t for t in self.stdio_params if t not in stdio_params]
        return self

    def call_tool(self, server: str, tool_name: str, args: dict):
        mcp_server = next((s for s in self.mcp_servers if s.name == server), None)
        if mcp_server:
            return mcp_server.call_tool(tool_name, args)
        else:
            raise f"server not found: '{server}'"

    def to_tool_definition(self, server: str, tool: list[Tool]) -> list[ToolDefinition]:
        tool_definitions = []
        for t in tool:
            definition = {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,
            }
            callable = lambda **kwargs: self.call_tool(server, t.name, kwargs)
            tool_definitions.append(
                ToolDefinition(
                    name=t.name,
                    definition=definition,
                    callable=callable,
                )
            )
        return tool_definitions

    async def list_tools(self) -> list[ToolDefinition]:
        tools = {}
        for server in self.mcp_servers:
            tools[server.name] = await server.list_tools()
        return [self.to_tool_definition(key, tool) for key, tool in tools.items()]

    async def list_tools_isolated(self) -> list[ToolDefinition]:
        tools = {}
        for params in self.stdio_params:
            async with MCPServerStdio(params=params) as server:
                tools[server.name] = await server.list_tools()
        return [self.to_tool_definition(key, tool) for key, tool in tools.items()]
