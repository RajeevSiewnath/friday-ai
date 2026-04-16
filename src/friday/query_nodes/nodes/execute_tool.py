from langgraph.runtime import Runtime
from friday.query_nodes.states.messages_state import MessagesState
from friday.query_nodes.contexts.llm_context import LLMContext


async def execute_tool(state: MessagesState, runtime: Runtime[LLMContext]):
    tool_call = state["messages"][-1]
    result = await runtime.context.llm.tool_shed.call(
        tool_call["name"], tool_call["arguments"]
    )
    return {
        "messages": [
            {
                "type": "function_call_output",
                "call_id": tool_call["call_id"],
                "output": str(result),
            }
        ]
    }
