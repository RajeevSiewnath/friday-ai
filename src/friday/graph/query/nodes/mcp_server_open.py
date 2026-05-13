from typing import Any
from langgraph.runtime import Runtime
from friday.graph.query.contexts.llm_context import LLMContext
from friday.loggers.logger import Logger


async def mcp_server_open(state: Any, runtime: Runtime[LLMContext]):
    logger = Logger.get_logger("node.mcp_server_open")
    logger.debug("opening mcp servers")
    await runtime.context.llm.tool_shed.mcp_tool_box.open()

    logger.trace(
        "servers: %s",
        lambda: [
            mcp_server.name
            for mcp_server in runtime.context.llm.tool_shed.mcp_tool_box.mcp_servers
        ],
    )
    return {}
