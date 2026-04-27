from typing import Any

from friday.core.mcp_tool_box import MCPToolBox
from friday.loggers.logger import Logger


def mcp_server_close_factory(*mcp_tool_boxes: MCPToolBox):
    async def mcp_server_close(state: Any):
        logger = Logger.get_logger("node.mcp_server_close")
        logger.info("closing mcp servers")
        logger.debug(
            "servers: %s",
            lambda: [
                mcp_server.name
                for box in mcp_tool_boxes
                for mcp_server in box.mcp_servers
            ],
        )

        for box in mcp_tool_boxes:
            await box.close()

        return {}

    return mcp_server_close
