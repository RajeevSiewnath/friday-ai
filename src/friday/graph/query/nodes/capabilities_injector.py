from langgraph.runtime import Runtime
from friday.graph.query.contexts.llm_context import LLMContext
from friday.graph.query.states.messages_state import MessagesState
from friday.loggers.logger import Logger


def capabilities_injector(state: MessagesState, runtime: Runtime[LLMContext]):
    if len(runtime.context.llm.tool_shed.tools) > 0:
        logger = Logger.get_logger("node.capabilities_injector")
        logger.info("injecting capabilities into system prompt")
        logger.debug("tools: %s", lambda: runtime.context.llm.tool_shed.tools)
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "\n\nCapabilities:\n"
                    + "\n".join(
                        f"- {tool.name}: {tool.definition["description"]}"
                        for tool in runtime.context.llm.tool_shed.tools
                    ),
                }
            ]
        }
    else:
        return {}
