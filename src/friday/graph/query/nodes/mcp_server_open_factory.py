from typing import Any

from langgraph.runtime import Runtime
from friday.core.mcp_tool_box import MCPToolBox
from friday.graph.query.contexts.llm_context import LLMContext
from friday.loggers.logger import Logger


def mcp_server_open_factory(*mcp_tool_boxes: MCPToolBox, add_to_tool_shed: bool = True):
    async def mcp_server_open(state: Any, runtime: Runtime[LLMContext]):
        logger = Logger.get_logger("node.mcp_server_open")
        logger.info("opening mcp servers")
        for box in mcp_tool_boxes:
            await box.open()

        if add_to_tool_shed:
            logger.debug(
                "adding tools to shed: %s",
                lambda: [
                    mcp_server.name
                    for box in mcp_tool_boxes
                    for mcp_server in box.mcp_servers
                ],
            )
            for box in mcp_tool_boxes:
                await runtime.context.llm.tool_shed.mcp_tool_box(box)

        logger.debug(
            "servers: %s",
            lambda: [
                mcp_server.name
                for box in mcp_tool_boxes
                for mcp_server in box.mcp_servers
            ],
        )
        return {}

    return mcp_server_open
