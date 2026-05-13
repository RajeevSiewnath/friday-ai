from langgraph.runtime import Runtime
from friday.graph.query.states.messages_state import MessagesState
from friday.graph.query.contexts.llm_context import LLMContext
from friday.loggers.logger import Logger


async def execute_tool(state: MessagesState, runtime: Runtime[LLMContext]):
    logger = Logger.get_logger("node.execute_tool")
    logger.debug("executing tool")

    tool_call = state["messages"][-1]
    logger.trace("call: %s", lambda: tool_call)

    result = await runtime.context.llm.tool_shed.call(
        tool_call["name"], tool_call["arguments"]
    )
    logger.trace("result: %s", lambda: result)

    return {
        "messages": [
            {
                "type": "function_call_output",
                "call_id": tool_call["call_id"],
                "output": str(result),
            }
        ]
    }
